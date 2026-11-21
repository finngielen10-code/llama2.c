/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Portability switches. The fast paths need: AVX2+FMA (-march=native on x86) for the
// SIMD kernels, and C11 atomics + threads (gcc/clang) for the thread pool. Builds
// without them (plain `make`, MSVC, ARM) use the plain-C kernels, and without the pool
// run tasks through OpenMP if enabled, else on the calling thread (see pool.h).
#if defined __AVX2__ && defined __FMA__
    #define USE_AVX2 1
#else
    #define USE_AVX2 0
#endif
// thread pool (USE_POOL, pool_init, pool_parallel_for; also pulls in <immintrin.h> on x86)
#include "pool.h"
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    float *rope_cos, *rope_sin; // RoPE factors for the current position (head_size/2,)
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope_cos = calloc(p->dim / p->n_heads / 2 + 1, sizeof(float));
    s->rope_sin = calloc(p->dim / p->n_heads / 2 + 1, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->rope_cos || !s->rope_sin) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ----------------------------------------------------------------------------
// matmul: AVX2/FMA kernel + a tiny persistent spin-wait thread pool.
// libgomp's fork/join on Windows costs tens of microseconds per parallel region,
// which is more than a whole small matmul; a spinning pool syncs in ~1us.
// Each output row is computed exactly as in the serial kernel, so results are
// bit-identical regardless of how rows are distributed across threads.

#ifndef POOL_MIN_WORK
#define POOL_MIN_WORK 32768   // d*n below this runs on the calling thread only
#endif
#ifndef POOL_CHUNK_ROWS
#define POOL_CHUNK_ROWS 64    // rows grabbed per work-steal step (multiple of 4)
#endif

#if USE_AVX2
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// computes rows [r0, r1) of W (d,n) @ x; r0 must be a multiple of 4
static void matmul_rows(float* xout, const float* x, const float* w, int n, int r0, int r1) {
    int i = r0;
    for (; i + 4 <= r1; i += 4) {
        const float* w0 = w + (size_t)i * n;
        const float* w1 = w0 + n;
        const float* w2 = w1 + n;
        const float* w3 = w2 + n;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xv, a3);
        }
        float s0 = hsum256(a0), s1 = hsum256(a1), s2 = hsum256(a2), s3 = hsum256(a3);
        for (; j < n; j++) {
            s0 += w0[j] * x[j]; s1 += w1[j] * x[j];
            s2 += w2[j] * x[j]; s3 += w3[j] * x[j];
        }
        xout[i] = s0; xout[i+1] = s1; xout[i+2] = s2; xout[i+3] = s3;
    }
#else
// computes rows [r0, r1) of W (d,n) @ x (plain C, same order as the original)
static void matmul_rows(float* xout, const float* x, const float* w, int n, int r0, int r1) {
    int i = r0;
#endif
    for (; i < r1; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[(size_t)i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Optional 8-bit mode (-q 1). Off by default, so the default output is unchanged.
// At load, the weights are quantized in groups of Q8_GS values: int8 in [-127,127]
// plus one float scale per group ("Q8_0"). The int8 copy mirrors the float weight
// region element by element, so a float weight pointer maps to its int8 copy by
// offset. The input vector of each matmul is quantized the same way, the dot
// products run in int8/int32, and each group's sum is scaled back to float.
// About 3.5x less weight memory traffic per token; output is close to, but not
// bit-identical with, the float path.

#ifndef Q8_GS
#define Q8_GS 32
#endif
#if Q8_GS != 32
#error "the AVX2 int8 kernel handles exactly one 32-byte group per step: Q8_GS must be 32"
#endif

static int g_quant = 0;                // set by -q 1
static const float* q8_fbase;          // start of the float weights
static int8_t* q8_w;                   // int8 weights, same layout as q8_fbase
static float* q8_ws;                   // one scale per Q8_GS weights
static int8_t* q8_x;                   // current matmul input, quantized
static float* q8_xs;                   // its scales

static void q8_quantize(int8_t* q, float* s, const float* x, size_t n) {
    for (size_t g = 0; g < n / Q8_GS; g++) {
        const float* xg = x + g * Q8_GS;
        float amax = 0.0f;
        for (int i = 0; i < Q8_GS; i++) { float a = fabsf(xg[i]); if (a > amax) amax = a; }
        float scale = amax / 127.0f;
        float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
        s[g] = scale;
        for (int i = 0; i < Q8_GS; i++) q[g * Q8_GS + i] = (int8_t)lrintf(xg[i] * inv);
    }
}

// quantize all float weights (n_floats of them, starting at base); returns 0 if unsupported
static int q8_init(const float* base, size_t n_floats, int dim, int hidden_dim) {
    if (dim % Q8_GS || hidden_dim % Q8_GS) return 0;
    n_floats -= n_floats % Q8_GS;
    q8_fbase = base;
    q8_w = malloc(n_floats);
    q8_ws = malloc(n_floats / Q8_GS * sizeof(float));
    int maxn = dim > hidden_dim ? dim : hidden_dim;
    q8_x = malloc(maxn);
    q8_xs = malloc(maxn / Q8_GS * sizeof(float));
    if (!q8_w || !q8_ws || !q8_x || !q8_xs) return 0;
    size_t chunk = (size_t)Q8_GS * 4096;
    int nchunks = (int)((n_floats + chunk - 1) / chunk);
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < nchunks; c++) {
        size_t o = (size_t)c * chunk, len = n_floats - o < chunk ? n_floats - o : chunk;
        q8_quantize(q8_w + o, q8_ws + o / Q8_GS, base + o, len);
    }
    return 1;
}

#if USE_AVX2
// sum of 32 int8 products a[i]*b[i], as 8 int32 lanes
static inline __m256i q8_dot32(__m256i a, __m256i b) {
    __m256i ua = _mm256_sign_epi8(a, a);   // |a|, fits u8 since a >= -127
    __m256i sb = _mm256_sign_epi8(b, a);   // b with a's sign
#if defined __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ua, sb);
#elif defined __AVX512VNNI__ && defined __AVX512VL__
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), ua, sb);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(ua, sb), _mm256_set1_epi16(1));
#endif
}
#endif

// rows [r0, r1) of W (d,n) @ x, using the int8 copy of w and the quantized input q8_x
static void matmul_rows_q8(float* xout, const float* w, int n, int r0, int r1) {
    size_t off = (size_t)(w - q8_fbase);
    const int8_t* wq = q8_w + off;
    const float* wsc = q8_ws + off / Q8_GS;
    int ng = n / Q8_GS;
    int i = r0;
#if USE_AVX2
    for (; i + 4 <= r1; i += 4) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const int8_t* w0 = wq + (size_t)i * n;
        const float* s0 = wsc + (size_t)i * ng;
        for (int g = 0; g < ng; g++) {
            __m256i xv = _mm256_loadu_si256((const __m256i*)(q8_x + g * Q8_GS));
            float xs = q8_xs[g];
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q8_dot32(_mm256_loadu_si256((const __m256i*)(w0 + g * Q8_GS)), xv)), _mm256_set1_ps(s0[g] * xs), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q8_dot32(_mm256_loadu_si256((const __m256i*)(w0 + n + g * Q8_GS)), xv)), _mm256_set1_ps(s0[ng + g] * xs), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q8_dot32(_mm256_loadu_si256((const __m256i*)(w0 + 2 * n + g * Q8_GS)), xv)), _mm256_set1_ps(s0[2 * ng + g] * xs), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q8_dot32(_mm256_loadu_si256((const __m256i*)(w0 + 3 * n + g * Q8_GS)), xv)), _mm256_set1_ps(s0[3 * ng + g] * xs), a3);
        }
        xout[i] = hsum256(a0); xout[i+1] = hsum256(a1); xout[i+2] = hsum256(a2); xout[i+3] = hsum256(a3);
    }
#endif
    for (; i < r1; i++) {
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
            int sum = 0;
            for (int k = 0; k < Q8_GS; k++) sum += wq[(size_t)i * n + g * Q8_GS + k] * q8_x[g * Q8_GS + k];
            val += sum * wsc[(size_t)i * ng + g] * q8_xs[g];
        }
        xout[i] = val;
    }
}

// call before dispatching a matmul job on input x: quantizes x in 8-bit mode
static inline void mm_prep(const float* x, int n) {
    if (g_quant) q8_quantize(q8_x, q8_xs, x, n);
}
// rows [r0, r1) of W @ x, in whichever mode is active (x must have gone through mm_prep)
static inline void mm_rows(float* xout, const float* x, const float* w, int n, int r0, int r1) {
    if (g_quant) matmul_rows_q8(xout, w, n, r0, r1);
    else matmul_rows(xout, x, w, n, r0, r1);
}


typedef struct { float* xout; const float* x; const float* w; int n; int d; } MatmulJob;

static void matmul_task(void* ctx, int task) {
    const MatmulJob* j = (const MatmulJob*)ctx;
    int r0 = task * POOL_CHUNK_ROWS;
    int r1 = r0 + POOL_CHUNK_ROWS < j->d ? r0 + POOL_CHUNK_ROWS : j->d;
    mm_rows(j->xout, j->x, j->w, j->n, r0, r1);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    mm_prep(x, n);
    if (pool_nthreads <= 1 || (size_t)d * n < POOL_MIN_WORK) {
        mm_rows(xout, x, w, n, 0, d);
        return;
    }
    MatmulJob job = { xout, x, w, n, d };
    pool_parallel_for(matmul_task, &job, (d + POOL_CHUNK_ROWS - 1) / POOL_CHUNK_ROWS);
}

// several matmuls sharing the same input x, dispatched as ONE pool job (one sync instead of n)
typedef struct { const float* x; int n; int nmat; float* out[3]; const float* w[3]; int d[3]; int first_task[4]; } MultiMatmulJob;

static void multi_matmul_task(void* ctx, int task) {
    const MultiMatmulJob* j = (const MultiMatmulJob*)ctx;
    int m = 0;
    while (task >= j->first_task[m + 1]) m++;
    int r0 = (task - j->first_task[m]) * POOL_CHUNK_ROWS;
    int r1 = r0 + POOL_CHUNK_ROWS < j->d[m] ? r0 + POOL_CHUNK_ROWS : j->d[m];
    mm_rows(j->out[m], j->x, j->w[m], j->n, r0, r1);
}

// xout_k = W_k (d_k,n) @ x for k < nmat (nmat <= 3)
static void matmul_multi(int nmat, const float* x, int n, float** outs, float** ws, const int* ds) {
    mm_prep(x, n);
    MultiMatmulJob j; j.x = x; j.n = n; j.nmat = nmat; j.first_task[0] = 0;
    for (int m = 0; m < nmat; m++) {
        j.out[m] = outs[m]; j.w[m] = ws[m]; j.d[m] = ds[m];
        j.first_task[m + 1] = j.first_task[m] + (ds[m] + POOL_CHUNK_ROWS - 1) / POOL_CHUNK_ROWS;
    }
    pool_parallel_for(multi_matmul_task, &j, j.first_task[nmat]);
}

// FFN up-projection fused with SwiGLU: hb = silu(W1 x) * (W3 x), rows split across the pool
typedef struct { float* hb; float* hb2; const float* x; const float* w1; const float* w3; int n; int d; } FfnUpJob;

static void ffn_up_task(void* ctx, int task) {
    const FfnUpJob* j = (const FfnUpJob*)ctx;
    int r0 = task * POOL_CHUNK_ROWS;
    int r1 = r0 + POOL_CHUNK_ROWS < j->d ? r0 + POOL_CHUNK_ROWS : j->d;
    mm_rows(j->hb, j->x, j->w1, j->n, r0, r1);
    mm_rows(j->hb2, j->x, j->w3, j->n, r0, r1);
    // SwiGLU non-linearity
    for (int i = r0; i < r1; i++) {
        float val = j->hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= j->hb2[i];
        j->hb[i] = val;
    }
}

// one attention head: scores, softmax, weighted sum of values (same math as the original loop)
typedef struct { RunState* s; int pos, head_size, kv_dim, kv_mul, seq_len, loff; } AttnJob;

static void attention_head(void* ctx, int h) {
    const AttnJob* j = (const AttnJob*)ctx;
    RunState* s = j->s;
    int pos = j->pos, head_size = j->head_size, kv_dim = j->kv_dim, kv_mul = j->kv_mul, loff = j->loff;
    // get the query vector for this head
    float* q = s->q + h * head_size;
    // attention scores for this head
    float* att = s->att + h * j->seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        int i = 0;
        float score = 0.0f;
#if USE_AVX2
        __m256 acc = _mm256_setzero_ps();
        for (; i <= head_size - 8; i += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), acc);
        }
        score = hsum256(acc);
#endif
        for (; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    float* xb = s->xb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
            xb[i] += a * v[i];
        }
    }
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // RoPE rotation factors depend only on (pos, head_dim): compute once per token, not per layer
    float* rope_cos = s->rope_cos;
    float* rope_sin = s->rope_sin;
    for (int hd = 0; hd < head_size; hd += 2) {
        float freq = 1.0f / powf(10000.0f, hd / (float)head_size);
        float val = pos * freq;
        rope_cos[hd >> 1] = cosf(val);
        rope_sin[hd >> 1] = sinf(val);
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        {
            float* outs[3] = { s->q, s->k, s->v };
            float* ws[3] = { w->wq + l*dim*dim, w->wk + l*dim*kv_dim, w->wv + l*dim*kv_dim };
            int ds[3] = { dim, kv_dim, kv_dim };
            matmul_multi(3, s->xb, dim, outs, ws, ds);
        }

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = rope_cos[head_dim >> 1];
            float fci = rope_sin[head_dim >> 1];
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention: one pool task per head
        AttnJob attn = { s, pos, head_size, kv_dim, kv_mul, p->seq_len, loff };
        pool_parallel_for(attention_head, &attn, p->n_heads);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // fused with the SwiGLU non-linearity, as one pool job
        mm_prep(s->xb, dim);
        FfnUpJob ffn = { s->hb, s->hb2, s->xb, w->w1 + l*dim*hidden_dim, w->w3 + l*dim*hidden_dim, dim, hidden_dim };
        pool_parallel_for(ffn_up_task, &ffn, (hidden_dim + POOL_CHUNK_ROWS - 1) / POOL_CHUNK_ROWS);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// softmax over the (large) vocab, split across the thread pool.
// max and the per-element exp/divide are order-independent; only the sum is
// accumulated per chunk and then combined in fixed chunk order (deterministic).
#ifndef SOFTMAX_CHUNK
#define SOFTMAX_CHUNK 2048
#endif
typedef struct { float* x; int n; float max_val; float sum; float partial[64]; } SoftmaxJob;

static void softmax_max_task(void* ctx, int t) {
    SoftmaxJob* j = (SoftmaxJob*)ctx;
    int i0 = t * SOFTMAX_CHUNK, i1 = i0 + SOFTMAX_CHUNK < j->n ? i0 + SOFTMAX_CHUNK : j->n;
    float m = j->x[i0];
    for (int i = i0 + 1; i < i1; i++) { if (j->x[i] > m) m = j->x[i]; }
    j->partial[t] = m;
}
static void softmax_exp_task(void* ctx, int t) {
    SoftmaxJob* j = (SoftmaxJob*)ctx;
    int i0 = t * SOFTMAX_CHUNK, i1 = i0 + SOFTMAX_CHUNK < j->n ? i0 + SOFTMAX_CHUNK : j->n;
    float sum = 0.0f;
    for (int i = i0; i < i1; i++) { j->x[i] = expf(j->x[i] - j->max_val); sum += j->x[i]; }
    j->partial[t] = sum;
}
static void softmax_div_task(void* ctx, int t) {
    SoftmaxJob* j = (SoftmaxJob*)ctx;
    int i0 = t * SOFTMAX_CHUNK, i1 = i0 + SOFTMAX_CHUNK < j->n ? i0 + SOFTMAX_CHUNK : j->n;
    for (int i = i0; i < i1; i++) { j->x[i] /= j->sum; }
}

void softmax_parallel(float* x, int n) {
    int nt = (n + SOFTMAX_CHUNK - 1) / SOFTMAX_CHUNK;
    if (nt > 64 || pool_nthreads <= 1) { softmax(x, n); return; }
    SoftmaxJob j; j.x = x; j.n = n;
    pool_parallel_for(softmax_max_task, &j, nt);
    float m = j.partial[0];
    for (int t = 1; t < nt; t++) { if (j.partial[t] > m) m = j.partial[t]; }
    j.max_val = m;
    pool_parallel_for(softmax_exp_task, &j, nt);
    float sum = 0.0f;
    for (int t = 0; t < nt; t++) { sum += j.partial[t]; }
    j.sum = sum;
    pool_parallel_for(softmax_div_task, &j, nt);
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    // Speedup: the truncated list is a prefix of the descending sort, i.e. it only contains
    // the largest probabilities. So first gather just the tokens above a much higher
    // threshold; if those already sum past topp (with a margin for summation-order rounding),
    // the prefix lies entirely inside that small set and we only need to sort it.
    // Otherwise lower the threshold, ending at the original cutoff.
    static const float thresholds[] = { 1e-2f, 1e-3f, 1e-4f };
    for (int t = 0; t <= 3; t++) {
        float th = t < 3 ? thresholds[t] : cutoff;
        if (th < cutoff) th = cutoff;
        n0 = 0;
        float mass = 0.0f;
        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= th) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                mass += probabilities[i];
                n0++;
            }
        }
        if (th == cutoff || mass > topp + 1e-3f) break;
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        // (dividing by exactly 1.0f is the identity, so skip the pass in that common case)
        if (sampler->temperature != 1.0f) {
            for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax_parallel(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
#if defined _WIN32
    win_clock_gettime(CLOCK_REALTIME, &time);
#else
    clock_gettime(CLOCK_REALTIME, &time);
#endif
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -q <int>    1 = 8-bit weights: faster, output differs slightly. default 0\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int quant = 0;              // 1 = 8-bit weights (faster, output differs slightly)

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'q') { quant = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (quant) {
        const float* wbase = transformer.data + sizeof(Config) / sizeof(float);
        size_t nf = (size_t)transformer.file_size / sizeof(float) - sizeof(Config) / sizeof(float);
        g_quant = q8_init(wbase, nf, transformer.config.dim, transformer.config.hidden_dim);
        if (!g_quant) fprintf(stderr, "8-bit mode not supported for this model, using float\n");
    }
    pool_init();
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
