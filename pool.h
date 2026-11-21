/* Spin-then-park thread pool shared by the inference programs (header-only).

   pool_init() starts persistent workers; pool_parallel_for(fn, ctx, ntasks) runs
   fn(ctx, 0..ntasks-1) across the calling thread and the workers (tasks are claimed from an
   atomic cursor, so slow cores simply take fewer) and returns when all are done. Syncing
   costs ~1 us, vs tens of us for an OpenMP parallel region on Windows.

   Idle workers spin (_mm_pause) for POOL_SPIN_BEFORE_YIELD pauses, then yield; after
   POOL_YIELDS_BEFORE_SLEEP yields (~0.3-1 s of idling, e.g. chat waiting for input) they
   sleep 1 ms at a time, so a waiting program doesn't keep cores at 100%.

   Needs C11 atomics and threads (pthreads, or CreateThread on Windows); otherwise
   (e.g. MSVC) everything runs through OpenMP if enabled, else on the calling thread.
   Include after <windows.h> (win.h) on Windows. Thread count: nprocs/2 + 1 (5 on a
   Core Ultra 5 226V: 4P + 4 LP E-cores); POOL_THREADS or OMP_NUM_THREADS override. */
#ifndef LLAMA2C_POOL_H
#define LLAMA2C_POOL_H

#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_POOL
    #if defined _MSC_VER || defined __STDC_NO_ATOMICS__
        #define USE_POOL 0
    #else
        #define USE_POOL 1
    #endif
#endif
#if defined __x86_64__ || defined __i386__ || defined _M_X64 || defined _M_IX86
    #include <immintrin.h>
    #define CPU_RELAX() _mm_pause()
#elif defined __aarch64__ || defined __arm__
    #define CPU_RELAX() __asm__ __volatile__("yield")
#else
    #define CPU_RELAX() ((void)0)
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
#if USE_POOL
    #include <stdatomic.h>
    #if defined _WIN32
        #define POOL_YIELD() SwitchToThread()
        #define POOL_SLEEP() Sleep(1)
    #else
        #include <pthread.h>
        #include <sched.h>
        #include <unistd.h>
        #define POOL_YIELD() sched_yield()
        #define POOL_SLEEP() do { struct timespec ts_ = { 0, 1000000 }; nanosleep(&ts_, NULL); } while (0)
    #endif
#endif
#ifndef POOL_SPIN_BEFORE_YIELD
#define POOL_SPIN_BEFORE_YIELD 200000
#endif
#ifndef POOL_YIELDS_BEFORE_SLEEP
#define POOL_YIELDS_BEFORE_SLEEP 64
#endif
#ifndef POOL_THREADS
#define POOL_THREADS 0 // 0 = nprocs/2 + 1
#endif

typedef void (*pool_task_fn)(void* ctx, int task);
static int pool_nthreads = 1; // including the calling thread

#if USE_POOL
static pool_task_fn pool_fn;
static void* pool_ctx;
static int pool_ntasks;
static atomic_int pool_gen;        // bumped to publish a new job
static atomic_int pool_next_task;  // work-stealing cursor
static atomic_int pool_active;     // workers still running the current job
static atomic_int pool_started;    // workers that have entered pool_worker

static void pool_run_job(void) {
    for (;;) {
        int t = atomic_fetch_add_explicit(&pool_next_task, 1, memory_order_relaxed);
        if (t >= pool_ntasks) break;
        pool_fn(pool_ctx, t);
    }
}

// Windows 11 on hybrid CPUs: opt the calling thread out of power throttling (EcoQoS) so
// Thread Director prefers performance cores for it. No-op elsewhere.
static void set_high_qos(void) {
#if defined _WIN32 && defined THREAD_POWER_THROTTLING_CURRENT_VERSION
    THREAD_POWER_THROTTLING_STATE st;
    memset(&st, 0, sizeof(st));
    st.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    st.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    st.StateMask = 0; // throttling off
    // resolved at runtime: the API is Windows 8+ and not declared under win.h's _WIN32_WINNT
    typedef BOOL (WINAPI *SetThreadInformationFn)(HANDLE, int, LPVOID, DWORD);
    SetThreadInformationFn fn = (SetThreadInformationFn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "SetThreadInformation");
    if (fn) fn(GetCurrentThread(), ThreadPowerThrottling, &st, sizeof(st));
#endif
}

static void* pool_worker(void* arg) {
    (void)arg;
    set_high_qos();
    // Start from generation 0 (the value before any job), NOT the current pool_gen: a
    // worker that the OS starts late, after the main thread already published job 1,
    // would otherwise treat that job as seen, never run it or decrement pool_active, and
    // the main thread would wait forever (seen as a process hanging at ~100% CPU).
    // pool_init also waits for all workers to start, so this is belt and braces.
    int seen = 0;
    atomic_fetch_add_explicit(&pool_started, 1, memory_order_release);
    for (;;) {
        int g, spins = 0, yields = 0;
        while ((g = atomic_load_explicit(&pool_gen, memory_order_acquire)) == seen) {
            if (yields >= POOL_YIELDS_BEFORE_SLEEP) { POOL_SLEEP(); continue; } // long idle: park
            CPU_RELAX();
            if (++spins >= POOL_SPIN_BEFORE_YIELD) { POOL_YIELD(); spins = 0; yields++; }
        }
        seen = g;
        pool_run_job();
        atomic_fetch_sub_explicit(&pool_active, 1, memory_order_release);
    }
    return NULL;
}
#if defined _WIN32
static DWORD WINAPI pool_worker_win(LPVOID arg) { pool_worker(arg); return 0; }
#endif

static int num_procs(void) {
#if defined _OPENMP
    return omp_get_num_procs();
#elif defined _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1;
#endif
}

static void pool_init(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;
    set_high_qos();
    int nt = POOL_THREADS > 0 ? POOL_THREADS : num_procs() / 2 + 1;
    if (getenv("OMP_NUM_THREADS")) nt = atoi(getenv("OMP_NUM_THREADS"));
    if (nt < 1) nt = 1;
    pool_nthreads = nt;
    for (int t = 1; t < nt; t++) {
#if defined _WIN32
        HANDLE th = CreateThread(NULL, 0, pool_worker_win, NULL, 0, NULL);
        if (!th) { pool_nthreads = t; break; }
        CloseHandle(th);
#else
        pthread_t th;
        if (pthread_create(&th, NULL, pool_worker, NULL) != 0) { pool_nthreads = t; break; }
        pthread_detach(th);
#endif
    }
    // don't publish a job before every worker is running (see pool_worker)
    while (atomic_load_explicit(&pool_started, memory_order_acquire) < pool_nthreads - 1) { POOL_YIELD(); }
}

// runs fn(ctx, 0..ntasks-1) across the pool; returns when all tasks are done.
// Not reentrant: call it from the main thread only, never from inside a task.
static void pool_parallel_for(pool_task_fn fn, void* ctx, int ntasks) {
    if (pool_nthreads <= 1 || ntasks <= 1) {
        for (int t = 0; t < ntasks; t++) fn(ctx, t);
        return;
    }
    pool_fn = fn; pool_ctx = ctx; pool_ntasks = ntasks;
    atomic_store_explicit(&pool_next_task, 0, memory_order_relaxed);
    atomic_store_explicit(&pool_active, pool_nthreads - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool_gen, 1, memory_order_release); // publish
    pool_run_job();
    while (atomic_load_explicit(&pool_active, memory_order_acquire) != 0) { CPU_RELAX(); }
}
#else
// no C11 atomics (e.g. MSVC): OpenMP if enabled, else the calling thread
static void pool_init(void) {}
static void pool_parallel_for(pool_task_fn fn, void* ctx, int ntasks) {
    int t;
    #pragma omp parallel for schedule(dynamic, 1) private(t)
    for (t = 0; t < ntasks; t++) fn(ctx, t);
}
#endif

#endif // LLAMA2C_POOL_H
