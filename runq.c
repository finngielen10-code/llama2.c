/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// AVX2 int8 matmul kernel (-march=native on an AVX2/FMA CPU); plain C otherwise.
// Both give bit-identical results: the int8 dot products are exact integers and the
// float math per row is done in the same order.
#if defined(__AVX2__) && defined(__FMA__)
    #include <immintrin.h>
    #define RUNQ_SIMD 1
#endif

// thread pool (pool_init / pool_parallel_for); same design as run.c's inline pool
#include "pool.h"

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

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
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    float *rope_cos, *rope_sin; // RoPE rotation for the current position, (dim/2,), filled once per token
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
    float rope_theta; // RoPE base frequency: 10000 for version 2 files, from the header for version 3
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)) };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope_cos = calloc(p->dim / 2, sizeof(float));
    s->rope_sin = calloc(p->dim / 2, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->rope_cos || !s->rope_sin || !s->key_cache
     || !s->value_cache) {
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
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size, float* rope_theta) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32): 2, or 3 = version 2 plus a rope_theta field
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2 && version != 3) { fprintf(stderr, "Bad version %d, need version 2 or 3\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; // the header size for versions 2 and 3 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // set as global, as it will be used in many places
    if (GS <= 0) { fprintf(stderr, "bad group size %d\n", GS); exit(EXIT_FAILURE); }
    // RoPE base: version 3 files (scripts/hf_to_q80.py, rope_theta != 10000) store it in the
    // header padding right after the group size; version 2 means the Llama 2 default 10000.
    float theta = 10000.0f;
    if (version == 3) {
        if (fread(&theta, sizeof(float), 1, file) != 1 || !(theta > 0.0f)) {
            fprintf(stderr, "bad rope_theta in header\n"); exit(EXIT_FAILURE);
        }
    }
    *rope_theta = theta;
    // figure out the file size
    // get the file size in bytes; 64-bit on Windows, where ftell's long is 32 bits and
    // would break checkpoints of 2 GB or more
#if defined _WIN32
    _fseeki64(file, 0, SEEK_END);
    *file_size = _ftelli64(file);
#else
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
#endif
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size, &t->rope_theta);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
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

// one output row, exactly as the original scalar loop computes it
static inline float matmul_row(QuantizedTensor *x, QuantizedTensor *w, int n, int i) {
    float val = 0.0f;
    int32_t ival = 0;
    int in = i * n;
    // do the matmul in groups of GS
    for (int j = 0; j <= n - GS; j += GS) {
        for (int k = 0; k < GS; k++) {
            ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
        }
        val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        ival = 0;
    }
    return val;
}

#ifdef RUNQ_SIMD
// acc (8 x int32) += dot products of 32 int8 pairs, 4 per lane. Integer math, so exact.
static inline __m256i dot_i8(__m256i acc, __m256i x, __m256i w) {
#ifdef __AVXVNNIINT8__
    return _mm256_dpbssd_epi32(acc, x, w);              // signed x signed (Lunar Lake, Arrow Lake)
#else
    // |x| (unsigned) * (w with x's sign); no int16 saturation since |x|,|w| <= 127
    __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}

// lane r of the result = sum of the 8 lanes of a[r]
static inline __m256i hsum8x8(const __m256i *a) {
    __m256i s01 = _mm256_hadd_epi32(a[0], a[1]);
    __m256i s23 = _mm256_hadd_epi32(a[2], a[3]);
    __m256i s45 = _mm256_hadd_epi32(a[4], a[5]);
    __m256i s67 = _mm256_hadd_epi32(a[6], a[7]);
    __m256i s0123 = _mm256_hadd_epi32(s01, s23);
    __m256i s4567 = _mm256_hadd_epi32(s45, s67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(s0123, s4567, 0x20),
                            _mm256_permute2x128_si256(s0123, s4567, 0x31));
}

// 8 rows i0..i0+7. Per group: exact int32 dot products, then per lane the same
// float ops in the same order as matmul_row: (float)ival * ws, * xs, val + that.
// fp-contract=off keeps GCC from fusing those into an FMA (the scalar code isn't fused).
__attribute__((optimize("fp-contract=off")))
static void matmul_rows8(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int i0) {
    int ng = n / GS;
    const int8_t *wq = w->q + (size_t)i0 * n;
    const float *ws = w->s + (size_t)i0 * ng;
    __m256i sidx = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(ng));
    __m256 val = _mm256_setzero_ps();
    for (int g = 0; g < ng; g++) {
        __m256i acc[8];
        for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_si256();
        for (int k = g * GS; k < (g + 1) * GS; k += 32) {
            __m256i xv = _mm256_loadu_si256((const __m256i*)(x->q + k));
            for (int r = 0; r < 8; r++)
                acc[r] = dot_i8(acc[r], xv, _mm256_loadu_si256((const __m256i*)(wq + (size_t)r * n + k)));
        }
        __m256 t = _mm256_cvtepi32_ps(hsum8x8(acc));
        t = _mm256_mul_ps(t, _mm256_i32gather_ps(ws + g, sidx, 4));
        t = _mm256_mul_ps(t, _mm256_set1_ps(x->s[g]));
        val = _mm256_add_ps(val, t);
    }
    _mm256_storeu_ps(xout + i0, val);
}
#endif

// rows [r0, r1) of W (d,n) @ x (n,) -> xout
static void matmul_range(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int r0, int r1) {
    int i = r0;
#ifdef RUNQ_SIMD
    if (GS % 32 == 0) {
        for (; i + 8 <= r1; i += 8) matmul_rows8(xout, x, w, n, i);
    }
#endif
    for (; i < r1; i++) xout[i] = matmul_row(x, w, n, i);
}

// rows per scheduled task (multiple of 8). Tasks are handed out dynamically, so the
// slower LP E-cores just take fewer of them instead of holding up the P-cores.
#ifndef MM_TASK_ROWS
#define MM_TASK_ROWS 64
#endif

typedef struct { float* xout; QuantizedTensor *x, *w; int n, d; } MMJob;

static void matmul_task(void* ctx, int t) {
    MMJob *j = (MMJob*)ctx;
    int r0 = t * MM_TASK_ROWS;
    int r1 = r0 + MM_TASK_ROWS < j->d ? r0 + MM_TASK_ROWS : j->d;
    matmul_range(j->xout, j->x, j->w, j->n, r0, r1);
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    MMJob job = { xout, x, w, n, d };
    pool_parallel_for(matmul_task, &job, (d + MM_TASK_ROWS - 1) / MM_TASK_ROWS);
}

// one attention head (pool task), same arithmetic as the original per-head loop body
typedef struct { RunState *s; int seq_len, loff, pos, head_size, kv_dim, kv_mul; } AttnJob;

static void attention_head(void* ctx, int h) {
    AttnJob *j = (AttnJob*)ctx;
    RunState *s = j->s;
    int pos = j->pos, head_size = j->head_size, kv_dim = j->kv_dim, loff = j->loff;
    // get the query vector for this head
    float* q = s->q + h * head_size;
    // attention scores for this head
    float* att = s->att + h * j->seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float* k = s->key_cache + loff + t * kv_dim + (h / j->kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
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
        float* v = s->value_cache + loff + t * kv_dim + (h / j->kv_mul) * head_size;
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
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // RoPE rotation for this position: same expressions as the original per-layer loop,
    // but evaluated once per token instead of once per layer (identical values)
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(transformer->rope_theta, head_dim / (float)head_size);
        float val = pos * freq;
        s->rope_cos[i / 2] = cosf(val);
        s->rope_sin[i / 2] = sinf(val);
    }

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            float fcr = s->rope_cos[i / 2];
            float fci = s->rope_sin[i / 2];
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        AttnJob aj = { s, p->seq_len, loff, pos, head_size, kv_dim, kv_mul };
        pool_parallel_for(attention_head, &aj, p->n_heads);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
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
    // ids used by generate/chat: 1/2 for the original llama2.c tokenizer.bin
    int bos_id, eos_id;
    // byte-level BPE ("BPE2" files from scripts/hf_tokenizer.py, e.g. SmolLM2); bpe == 0
    // means the original SentencePiece-style tokenizer.bin and none of these are used
    int bpe;
    int* vocab_len;              // byte length of each piece (pieces may contain any byte)
    unsigned char* is_special;   // special tokens: inserted by id, decode to nothing
    int byte_token[256];         // base token of each byte, -1 = none (byte is dropped)
    int n_merges;
    uint64_t* merge_keys;        // (left << 32 | right), sorted, for binary search
    int* merge_rank;             // aligned with merge_keys: rank (lower merges first)
    int* merge_result;           // aligned with merge_keys: merged token id
} Tokenizer;

#define BPE2_MAGIC 0x32455042 // "BPE2"

// printing: byte-level BPE splits multi-byte UTF-8 characters across tokens, so single
// bytes >= 0x80 must be printed, not filtered out as the SentencePiece path does
static int g_print_high_bytes = 0;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

static void read_or_die(void* dst, size_t size, size_t n, FILE* f) {
    if (fread(dst, size, n, f) != n) { fprintf(stderr, "failed read (truncated tokenizer file?)\n"); exit(EXIT_FAILURE); }
}

typedef struct { uint64_t key; int rank, result; } MergeEntry;
static int compare_merges(const void* a, const void* b) {
    uint64_t x = ((const MergeEntry*)a)->key, y = ((const MergeEntry*)b)->key;
    return (x > y) - (x < y);
}

// byte-level BPE tokenizer file (see scripts/hf_tokenizer.py for the layout)
static void build_bpe_tokenizer(Tokenizer* t, FILE* file, int vocab_size) {
    int hdr[6]; // version, vocab_size, bos, eos, n_merges, max_token_length
    read_or_die(hdr, sizeof(int), 6, file);
    if (hdr[0] != 1) { fprintf(stderr, "unsupported BPE2 tokenizer version %d\n", hdr[0]); exit(EXIT_FAILURE); }
    if (hdr[1] != vocab_size) {
        fprintf(stderr, "tokenizer has %d tokens but the model has %d (wrong -z file?)\n", hdr[1], vocab_size);
        exit(EXIT_FAILURE);
    }
    t->bpe = 1;
    t->vocab_size = vocab_size;
    t->bos_id = hdr[2];
    t->eos_id = hdr[3];
    t->n_merges = hdr[4];
    t->max_token_length = (unsigned int)hdr[5];
    if (t->bos_id < 0 || t->bos_id >= vocab_size || t->eos_id < 0 || t->eos_id >= vocab_size || t->n_merges < 0) {
        fprintf(stderr, "bad BPE2 tokenizer header\n"); exit(EXIT_FAILURE);
    }
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_len = (int*)malloc(vocab_size * sizeof(int));
    t->is_special = (unsigned char*)malloc(vocab_size);
    t->vocab_scores = NULL;
    t->sorted_vocab = NULL;
    for (int b = 0; b < 256; b++) t->byte_token[b] = -1;
    for (int i = 0; i < vocab_size; i++) {
        unsigned char flags; int len;
        read_or_die(&flags, 1, 1, file);
        read_or_die(&len, sizeof(int), 1, file);
        if (len < 0 || len > (int)t->max_token_length) { fprintf(stderr, "bad token length\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char*)malloc(len + 1);
        if (len) read_or_die(t->vocab[i], 1, len, file);
        t->vocab[i][len] = '\0';
        t->vocab_len[i] = len;
        t->is_special[i] = flags & 1;
        if (len == 1 && !t->is_special[i]) t->byte_token[(unsigned char)t->vocab[i][0]] = i;
    }
    MergeEntry* m = (MergeEntry*)malloc((t->n_merges ? t->n_merges : 1) * sizeof(MergeEntry));
    for (int r = 0; r < t->n_merges; r++) {
        int e[3];
        read_or_die(e, sizeof(int), 3, file);
        for (int k = 0; k < 3; k++) {
            if (e[k] < 0 || e[k] >= vocab_size) { fprintf(stderr, "bad merge entry\n"); exit(EXIT_FAILURE); }
        }
        m[r].key = ((uint64_t)(uint32_t)e[0] << 32) | (uint32_t)e[1];
        m[r].rank = r;
        m[r].result = e[2];
    }
    qsort(m, t->n_merges, sizeof(MergeEntry), compare_merges);
    t->merge_keys = (uint64_t*)malloc((t->n_merges ? t->n_merges : 1) * sizeof(uint64_t));
    t->merge_rank = (int*)malloc((t->n_merges ? t->n_merges : 1) * sizeof(int));
    t->merge_result = (int*)malloc((t->n_merges ? t->n_merges : 1) * sizeof(int));
    for (int r = 0; r < t->n_merges; r++) {
        t->merge_keys[r] = m[r].key; t->merge_rank[r] = m[r].rank; t->merge_result[r] = m[r].result;
    }
    free(m);
    g_print_high_bytes = 1;
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->bpe = 0;
    t->bos_id = 1;
    t->eos_id = 2;
    t->vocab_len = NULL; t->is_special = NULL;
    t->merge_keys = NULL; t->merge_rank = NULL; t->merge_result = NULL; t->n_merges = 0;
    {
        FILE *file = fopen(tokenizer_path, "rb");
        if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
        int magic = 0;
        if (fread(&magic, sizeof(int), 1, file) == 1 && magic == BPE2_MAGIC) {
            build_bpe_tokenizer(t, file, vocab_size);
            fclose(file);
            return;
        }
        fclose(file);
    }
    // original llama2.c tokenizer.bin
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
    free(t->vocab_len);
    free(t->is_special);
    free(t->merge_keys);
    free(t->merge_rank);
    free(t->merge_result);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    if (t->bpe) {
        // byte-level BPE: pieces are the raw bytes; special tokens print nothing
        return t->is_special[token] ? "" : t->vocab[token];
    }
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
        int ok = isprint(byte_val) || isspace(byte_val) || (g_print_high_bytes && byte_val >= 0x80);
        if (!ok) {
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

// ----------------------------------------------------------------------------
// byte-level BPE encoding, matching Hugging Face tokenizers with a
// Sequence[Digits(individual_digits), ByteLevel(use_regex, no prefix space)] pre-tokenizer
// (GPT-2 / SmolLM2 style). Text is split into pieces, then each piece's bytes are merged by
// merge rank; merges never cross piece boundaries.

enum { CC_LETTER, CC_DIGIT, CC_SPACE, CC_OTHER };

// decode one UTF-8 code point at s[i] (s has n bytes); returns its byte length (>= 1).
// An invalid byte is returned as a 1-byte "code point" with its own value.
static int utf8_cp(const unsigned char* s, int i, int n, uint32_t* cp) {
    unsigned char c = s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + len > n) len = 1;
    for (int k = 1; k < len; k++) if ((s[i + k] & 0xC0) != 0x80) { len = 1; break; }
    if (len == 1) { *cp = c; return 1; }
    uint32_t v = c & (0x7F >> len);
    for (int k = 1; k < len; k++) v = (v << 6) | (s[i + k] & 0x3F);
    *cp = v;
    return len;
}

// Character class for the pre-tokenizer: CC_DIGIT ~ Rust char::is_numeric (each one becomes
// its own piece), CC_SPACE ~ \s, CC_LETTER ~ \p{L}, CC_OTHER = everything else. Exact for
// ASCII and Latin-1; common punctuation/symbol/emoji/numeral blocks are mapped explicitly and
// any other non-ASCII code point counts as a letter (combining marks of non-Latin scripts
// are the known approximation).
static int bpe_class(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp | 32) >= 'a' && (cp | 32) <= 'z') return CC_LETTER;
        if (cp >= '0' && cp <= '9') return CC_DIGIT;
        if (cp == ' ' || (cp >= 9 && cp <= 13)) return CC_SPACE;
        return CC_OTHER;
    }
    if (cp < 0x100) {
        if (cp == 0x85 || cp == 0xA0) return CC_SPACE;
        if (cp == 0xB2 || cp == 0xB3 || cp == 0xB9 || (cp >= 0xBC && cp <= 0xBE)) return CC_DIGIT;
        if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return CC_LETTER;
        if (cp >= 0xC0 && cp != 0xD7 && cp != 0xF7) return CC_LETTER;
        return CC_OTHER; // C1 controls, Latin-1 punctuation and symbols, x and /
    }
    if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
        || cp == 0x202F || cp == 0x205F || cp == 0x3000) return CC_SPACE;
    if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) || (cp >= 0x0966 && cp <= 0x096F)
        || (cp >= 0x2070 && cp <= 0x2079 && cp != 0x2071 && cp != 0x2072 && cp != 0x2073)
        || (cp >= 0x2080 && cp <= 0x2089) || (cp >= 0x2150 && cp <= 0x2189) || (cp >= 0x2460 && cp <= 0x249B)
        || (cp >= 0x24EA && cp <= 0x24FF) || (cp >= 0x2776 && cp <= 0x2793) || cp == 0x3007
        || (cp >= 0x3021 && cp <= 0x3029) || (cp >= 0xFF10 && cp <= 0xFF19)) return CC_DIGIT;
    if ((cp >= 0x0300 && cp <= 0x036F)                      // combining diacritics (\p{M})
        || cp == 0x060C || cp == 0x061B || cp == 0x061F || cp == 0x06D4 || cp == 0x0964 || cp == 0x0965
        || (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x2030 && cp <= 0x205E) // general punctuation
        || (cp >= 0x20A0 && cp <= 0x20FF)                   // currency, combining marks for symbols
        || (cp >= 0x2100 && cp <= 0x214F && cp != 0x2102 && cp != 0x2107 && !(cp >= 0x210A && cp <= 0x2113)
            && cp != 0x2115 && !(cp >= 0x2119 && cp <= 0x211D) && cp != 0x2124 && cp != 0x2126 && cp != 0x2128
            && !(cp >= 0x212A && cp <= 0x212D) && !(cp >= 0x212F && cp <= 0x2139) && !(cp >= 0x213C && cp <= 0x213F)
            && !(cp >= 0x2145 && cp <= 0x2149) && cp != 0x214E)  // letterlike symbols minus the letters
        || (cp >= 0x2190 && cp <= 0x245F) || (cp >= 0x249C && cp <= 0x24E9) || (cp >= 0x2500 && cp <= 0x2775)
        || (cp >= 0x2794 && cp <= 0x2BFF) || (cp >= 0x2E00 && cp <= 0x2E7F)  // arrows, math, boxes, dingbats
        || (cp >= 0x3001 && cp <= 0x3004) || (cp >= 0x3008 && cp <= 0x3020) || (cp >= 0x3030 && cp <= 0x303F)
        || (cp >= 0xE000 && cp <= 0xF8FF)                   // private use
        || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xFE10 && cp <= 0xFE1F) || (cp >= 0xFE30 && cp <= 0xFE6F)
        || (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40)
        || (cp >= 0xFF5B && cp <= 0xFF65) || (cp >= 0xFFF0 && cp <= 0xFFFF)
        || (cp >= 0x1F000 && cp <= 0x1FAFF) || (cp >= 0xE0000 && cp <= 0xE007F)) return CC_OTHER; // emoji, tags
    return CC_LETTER;
}

static int cp_class_at(const unsigned char* s, int i, int n, int* len) {
    uint32_t cp;
    *len = utf8_cp(s, i, n, &cp);
    return bpe_class(cp);
}

// end of the GPT-2 regex match starting at i, within a digit-free span [i, e):
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static int bpe_regex_match(const unsigned char* s, int i, int e) {
    int len, k;
    if (s[i] == '\'' && i + 1 < e) {
        unsigned char c = s[i + 1];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < e && ((c == 'r' && s[i + 2] == 'e') || (c == 'v' && s[i + 2] == 'e') || (c == 'l' && s[i + 2] == 'l'))) return i + 3;
    }
    int start = i;
    int cls = cp_class_at(s, i, e, &len);
    if (s[i] == ' ' && i + 1 < e) { // optional single space before letters or symbols
        int len2, c2 = cp_class_at(s, i + 1, e, &len2);
        if (c2 == CC_LETTER || c2 == CC_OTHER) { cls = c2; start = i + 1; }
    }
    if (cls == CC_LETTER || cls == CC_OTHER) {
        k = start;
        while (k < e && cp_class_at(s, k, e, &len) == cls) k += len;
        return k;
    }
    // whitespace run; \s+(?!\S) leaves the last whitespace char for the next token when a
    // non-space follows (and fails for a single char, falling through to \s+)
    k = i;
    int last = i;
    while (k < e && cp_class_at(s, k, e, &len) == CC_SPACE) { last = k; k += len; }
    if (k == i) return i + len; // not reached for valid classes; never return an empty match
    if (k == e) return k;
    return last > i ? last : k;
}

static int bpe_merge_lookup(const Tokenizer* t, int a, int b, int* result) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    int lo = 0, hi = t->n_merges - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (t->merge_keys[mid] < key) lo = mid + 1;
        else if (t->merge_keys[mid] > key) hi = mid - 1;
        else { *result = t->merge_result[mid]; return t->merge_rank[mid]; }
    }
    return -1;
}

// apply merges to ids[0..n) in rank order (all occurrences of the best pair, left to right),
// then append the result to out
static void bpe_merge_and_emit(const Tokenizer* t, int* ids, int n, int* out, int* n_out) {
    for (;;) {
        int best_rank = -1, best_result = 0, best_a = 0, best_b = 0;
        for (int k = 0; k + 1 < n; k++) {
            int result, r = bpe_merge_lookup(t, ids[k], ids[k + 1], &result);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r; best_result = result; best_a = ids[k]; best_b = ids[k + 1];
            }
        }
        if (best_rank < 0) break;
        int w = 0;
        for (int k = 0; k < n; k++) {
            if (k + 1 < n && ids[k] == best_a && ids[k + 1] == best_b) { ids[w++] = best_result; k++; }
            else ids[w++] = ids[k];
        }
        n = w;
    }
    for (int k = 0; k < n; k++) out[(*n_out)++] = ids[k];
}

// one pre-tokenized piece: bytes -> base tokens -> merges. A byte without a token is dropped
// and, like in HF, nothing merges across it.
static void bpe_encode_piece(const Tokenizer* t, const unsigned char* s, int n, int* work, int* out, int* n_out) {
    int m = 0;
    for (int k = 0; k < n; k++) {
        int id = t->byte_token[s[k]];
        if (id < 0) { bpe_merge_and_emit(t, work, m, out, n_out); m = 0; continue; }
        work[m++] = id;
    }
    bpe_merge_and_emit(t, work, m, out, n_out);
}

static void bpe_encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    const unsigned char* s = (const unsigned char*)text;
    int n = (int)strlen(text);
    int* work = (int*)malloc((n + 1) * sizeof(int));
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = t->bos_id;
    int i = 0;
    while (i < n) {
        int len, cls = cp_class_at(s, i, n, &len);
        if (cls == CC_DIGIT) { // Digits(individual_digits=true): every numeric char alone
            bpe_encode_piece(t, s + i, len, work, tokens, n_tokens);
            i += len;
            continue;
        }
        int e = i; // the regex runs on the digit-free span up to the next numeric char
        while (e < n && cp_class_at(s, e, n, &len) != CC_DIGIT) e += len;
        while (i < e) {
            int j = bpe_regex_match(s, i, e);
            bpe_encode_piece(t, s + i, j - i, work, tokens, n_tokens);
            i = j;
        }
    }
    if (eos) tokens[(*n_tokens)++] = t->eos_id;
    free(work);
}

// id of a special token by its text (e.g. "<|im_start|>"), or -1
static int find_special_token(const Tokenizer* t, const char* text) {
    if (!t->bpe) return -1;
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->is_special[i] && strcmp(t->vocab[i], text) == 0) return i;
    }
    return -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    if (t->bpe) { bpe_encode(t, text, bos, eos, tokens, n_tokens); return; }

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
    // ties: lower token id first. qsort isn't stable, so without this the order of equal
    // probabilities depends on the input array, and sample_topp's threshold shortcut
    // (which sorts a smaller array) could pick a different token than a full sort
    return (a_->index > b_->index) - (a_->index < b_->index);
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
    // Speedup (same result): the truncated list is a prefix of the descending sort, i.e. it
    // only holds the largest probabilities. So first gather just the tokens above a much
    // higher threshold; if those already sum past topp (with a margin for summation-order
    // rounding), the prefix lies entirely inside that small set and only it needs sorting.
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
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
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

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences;
        // with a byte-level BPE tokenizer any special token (<|endoftext|>, <|im_end|>...) does
        if (tokenizer->bpe ? tokenizer->is_special[next] : next == 1) { break; }

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

int read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n. Returns 0 on EOF or a read error
    // (buffer is then empty, not the previous line), 1 otherwise.
    printf("%s", guide);
    fflush(stdout);
    buffer[0] = '\0';
    if (fgets(buffer, bufsize, stdin) == NULL) return 0;
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
        buffer[len - 1] = '\0'; // strip newline
    } else if (!feof(stdin)) {
        // line longer than the buffer: drop the rest instead of leaving it in stdin,
        // where the next read would send it as a separate message
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {}
        fprintf(stderr, "[input truncated to %d characters]\n", (int)(bufsize - 1));
    }
    return 1;
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

// encode text (no BOS/EOS) and append its tokens to tokens[*n]
static void append_encoded(Tokenizer *tokenizer, char *text, int *tokens, int *n, int *scratch) {
    int k = 0;
    encode(tokenizer, text, 0, 0, scratch, &k);
    for (int i = 0; i < k; i++) tokens[(*n)++] = scratch[i];
}

// chat templates: "llama2" = [INST] ... [/INST] (Llama 2 Chat)
//                 "zephyr" = <|system|>/<|user|>/<|assistant|> turns ended by </s> (TinyLlama-Chat v1.0)
//                 "chatml" = <|im_start|>role\n...<|im_end|> turns (SmolLM2, needs its BPE2 tokenizer)
// With no -c, a byte-level BPE tokenizer uses chatml and the original tokenizer uses llama2.
#define CHATML_DEFAULT_SYSTEM "You are a helpful AI assistant named SmolLM, trained by Hugging Face" // SmolLM2's template default
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps, char *chat_template) {
    if (chat_template == NULL) chat_template = tokenizer->bpe ? "chatml" : "llama2";
    int zephyr = strcmp(chat_template, "zephyr") == 0;
    int chatml = strcmp(chat_template, "chatml") == 0;
    int eos = tokenizer->eos_id; // ends an assistant turn (2 = </s> for the original tokenizer)
    int im_start = -1;
    if (chatml) {
        im_start = find_special_token(tokenizer, "<|im_start|>");
        eos = find_special_token(tokenizer, "<|im_end|>");
        if (im_start < 0 || eos < 0) {
            fprintf(stderr, "-c chatml needs a tokenizer with <|im_start|> and <|im_end|> (the model's BPE2 tokenizer, via -z)\n");
            exit(EXIT_FAILURE);
        }
    } else if (tokenizer->bpe) {
        fprintf(stderr, "-c %s needs the original llama2.c tokenizer; use -c chatml with this tokenizer\n", chat_template);
        exit(EXIT_FAILURE);
    }

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(4096 * sizeof(int));
    int* scratch_tokens = (int*)malloc(2048 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next = 0;    // will store the next token in the sequence (set before first use)
    int token;       // stores the current token to feed into the transformer
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
                    snprintf(system_prompt, sizeof(system_prompt), "%s", cli_system_prompt); // bounded: -y can be long
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                snprintf(user_prompt, sizeof(user_prompt), "%s", cli_user_prompt); // bounded: -i can be long
            } else {
                // otherwise get user prompt from stdin
                // stdin closed or failing: end the chat
                if (!read_stdin("User: ", user_prompt, sizeof(user_prompt))) { printf("\n"); break; }
            }
            if (zephyr) {
                // <s><|system|>\n{sys}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n
                // </s> must be the real EOS token (2), so the prompt is built piecewise.
                // Later turns start with "\n" because the previous reply ended with </s>.
                num_prompt_tokens = 0;
                if (pos == 0) prompt_tokens[num_prompt_tokens++] = tokenizer->bos_id;
                if (pos == 0 && system_prompt[0] != '\0') {
                    sprintf(rendered_prompt, "<|system|>\n%s", system_prompt);
                    append_encoded(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, scratch_tokens);
                    prompt_tokens[num_prompt_tokens++] = eos;
                    sprintf(rendered_prompt, "\n<|user|>\n%s", user_prompt);
                } else if (pos == 0) {
                    sprintf(rendered_prompt, "<|user|>\n%s", user_prompt);
                } else {
                    sprintf(rendered_prompt, "\n<|user|>\n%s", user_prompt);
                }
                append_encoded(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, scratch_tokens);
                prompt_tokens[num_prompt_tokens++] = eos;
                append_encoded(tokenizer, "\n<|assistant|>\n", prompt_tokens, &num_prompt_tokens, scratch_tokens);
            } else if (chatml) {
                // [<|im_start|>]system\n{sys}[<|im_end|>]\n[<|im_start|>]user\n{user}[<|im_end|>]\n[<|im_start|>]assistant\n
                // Special tokens are inserted by id and the text between them is encoded
                // separately, as HF does. No BOS. Later turns start with "\n": the previous
                // reply ended with <|im_end|>. Like SmolLM2's template, a default system
                // prompt is used when none is given.
                num_prompt_tokens = 0;
                if (pos == 0) {
                    prompt_tokens[num_prompt_tokens++] = im_start;
                    snprintf(rendered_prompt, sizeof(rendered_prompt), "system\n%s",
                             system_prompt[0] != '\0' ? system_prompt : CHATML_DEFAULT_SYSTEM);
                    append_encoded(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, scratch_tokens);
                    prompt_tokens[num_prompt_tokens++] = eos;
                }
                append_encoded(tokenizer, "\n", prompt_tokens, &num_prompt_tokens, scratch_tokens);
                prompt_tokens[num_prompt_tokens++] = im_start;
                snprintf(rendered_prompt, sizeof(rendered_prompt), "user\n%s", user_prompt);
                append_encoded(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, scratch_tokens);
                prompt_tokens[num_prompt_tokens++] = eos;
                append_encoded(tokenizer, "\n", prompt_tokens, &num_prompt_tokens, scratch_tokens);
                prompt_tokens[num_prompt_tokens++] = im_start;
                append_encoded(tokenizer, "assistant\n", prompt_tokens, &num_prompt_tokens, scratch_tokens);
            } else {
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
            }
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        int forced = user_idx < num_prompt_tokens;
        if (forced) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS token (</s> = 2, or <|im_end|> for chatml) ends the Assistant turn (a forced
        // prompt token can be EOS too: zephyr/chatml end each user/system turn with it)
        if (token == eos && !forced) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != eos) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == eos) { printf("\n"); }
    }
    printf("\n");
    if (pos >= steps) {
        fprintf(stderr, "[chat ended: the conversation reached the %d-token limit (-n)]\n", steps);
    }
    free(prompt_tokens);
    free(scratch_tokens);
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
    fprintf(stderr, "  -c <string> chat template: llama2|zephyr|chatml, default: chatml with a BPE2 tokenizer\n");
    fprintf(stderr, "              (SmolLM2), else llama2. Use zephyr for TinyLlama-Chat.\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    int steps_given = 0;        // was -n passed? (chat defaults to the full context instead)
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    char *chat_template = NULL; // llama2|zephyr|chatml; NULL = chatml for a BPE2 tokenizer, else llama2

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
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); steps_given = 1; }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'c') { chat_template = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    // in chat mode -n bounds the whole conversation, so default to the full context
    if (strcmp(mode, "chat") == 0 && !steps_given) steps = 0;
    if (chat_template != NULL && strcmp(chat_template, "llama2") != 0 && strcmp(chat_template, "zephyr") != 0
        && strcmp(chat_template, "chatml") != 0) {
        fprintf(stderr, "unknown chat template: %s (use llama2, zephyr or chatml)\n", chat_template);
        error_usage();
    }

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    pool_init(); // start the worker threads once, before the first forward pass
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
#if defined _WIN32
    // byte-level BPE output is raw UTF-8; make the Windows console display it as such
    if (tokenizer.bpe) SetConsoleOutputCP(65001);
#endif

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps, chat_template);
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
