// Regression test for pool.h: every task of every job must run exactly once, even when
// the OS starts worker threads late. The old pool (workers starting from the current job
// counter, pool_init not waiting) deadlocked under exactly this condition.
//
// Build and run (exit 0 = pass; a hang means the deadlock is back):
//   gcc -O2 -I. -DPOOL_TEST_WORKER_START_DELAY_MS=50 tests/pool_test.c -o build/pool_test.exe
//   timeout 60 build/pool_test.exe
#include <stdio.h>
#include <stdatomic.h>
#if defined _WIN32
    #include <windows.h>
#endif
#include "pool.h"

#define JOBS 2000
#define TASKS 64

static atomic_int hits[TASKS];

static void task(void* ctx, int t) { (void)ctx; atomic_fetch_add(&hits[t], 1); }

int main(void) {
    pool_init();
    for (int job = 0; job < JOBS; job++) pool_parallel_for(task, NULL, TASKS);
    int bad = 0;
    for (int t = 0; t < TASKS; t++) {
        if (atomic_load(&hits[t]) != JOBS) { printf("task %d ran %d times, expected %d\n", t, atomic_load(&hits[t]), JOBS); bad = 1; }
    }
    printf("%s: %d threads, %d jobs x %d tasks\n", bad ? "FAIL" : "PASS", pool_nthreads, JOBS, TASKS);
    return bad;
}
