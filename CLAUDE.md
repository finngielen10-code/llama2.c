# llama2.c — CPU inference optimization

Fork of karpathy/llama2.c used to speed up `run.c` inference on Windows.

## Environment
- Windows 11, MinGW-w64 GCC 16 (WinLibs UCRT), Git Bash available.
- CPU: Intel Core Ultra 5 226V (Lunar Lake: 4 P-cores + 4 LP E-cores, AVX2/FMA, no AVX-512).
- Model: `stories15M.bin` (dim 288, 6 layers, 6 heads, vocab 32000) + `tokenizer.bin`. Not in git.

## Build & run
```
gcc -O3 -fopenmp -march=native -fpermissive run.c win.c -lm -o run.exe
./run.exe stories15M.bin -s 42
```
`win.c`/`win.h` provide mmap/clock shims for Windows (`clock_gettime` was renamed to `win_clock_gettime` to avoid clashing with the MinGW one; `run.c`/`runq.c` call `win_clock_gettime` on `_WIN32`).

### Portability (upstream CI builds)
The fast paths are compile-time optional, so every Makefile target and `build_msvc.bat` still build:
- `USE_AVX2` (needs `__AVX2__` and `__FMA__`, i.e. `-march=native`): SIMD kernels; otherwise plain-C kernels that match the original output exactly.
- `USE_POOL` (C11 atomics, i.e. not MSVC): spin-wait thread pool; otherwise everything runs on the calling thread. The pool uses pthreads, or `CreateThread` on Windows (MinGW then needs no `-lpthread`).
- OpenMP is optional: without `-fopenmp` the core count comes from `sysconf`/`GetSystemInfo`.
- No VLAs (MSVC): the RoPE tables live in `RunState`.

## Optimization rules
- The compile command above is fixed: don't rely on extra flags (no `-ffast-math`); put tunables in `run.c` as `#define`s with `#ifndef` guards.
- Output for `-s 42` must stay **byte-identical** to `tests/ref_s42.txt` (the original, unoptimized output). Reordering float sums can change sampled tokens, so always check.
- Make one change at a time and keep it only if it measurably wins.
- Timing is very noisy (±25% run to run). Never judge from a single run:
  - `scripts/bench.sh [N]` — build, verify output, median tok/s over N runs.
  - `scripts/ab.sh <old_run.c|git-rev> [N]` — interleaved A/B against a previous version (preferred for small gains).
- Build/bench artifacts go in `build/` (gitignored).

## Current design of `run.c`
Result: ~131 → **~640 tok/s** median on stories15M (~4.9×), `-s 42` output byte-identical.

- **Matmul kernel** (`matmul_rows`): AVX2/FMA, 4 rows × 8 lanes at a time, `hsum256` at the end. Each output row is computed the same way no matter which thread runs it, so results don't depend on thread count.
- **Thread pool instead of OpenMP**: OpenMP fork/join on Windows costs tens of µs per parallel region (more than a whole small matmul). `pool_init` starts persistent worker threads that busy-wait (`_mm_pause`, then yield after `POOL_SPIN_BEFORE_YIELD` spins) on a generation counter. `pool_parallel_for(fn, ctx, ntasks)` publishes a job; threads grab tasks from an atomic cursor (work stealing). Sync costs ~1 µs.
  - `POOL_MIN_WORK` (32768): matmuls with `d*n` below this run on the calling thread only.
  - `POOL_CHUNK_ROWS` (64, multiple of 4): rows per stolen task.
  - Thread count: `nprocs/2 + 1` (5 on the 226V). `OMP_NUM_THREADS` overrides. 4 → ~530, 5 → ~575, 7–8 → fast but erratic (OS work preempts a spinner and everyone waits).
- **Fused jobs** (fewer syncs per layer): `matmul_multi` does Q, K, V as one pool job; `ffn_up_task` does W1 and W3 plus SwiGLU as one job.
- **Attention**: one pool task per head (`attention_head`); the q·k dot product uses AVX2/FMA.
- **RoPE**: cos/sin computed once per token (`rope_cos`/`rope_sin`), not once per layer.
- **Sampler**:
  - `softmax_parallel` splits the 32000-entry vocab softmax across the pool in `SOFTMAX_CHUNK` (2048) pieces: max, then exp, then divide. Each pass is a separate job.
  - Temperature 1.0 skips the divide-by-temperature pass (dividing by 1.0f changes nothing).
  - `sample_topp` first tries higher probability cutoffs, so it sorts only a few hundred tokens instead of thousands. It falls back to the original cutoff, so the result is the same.
- **Windows only** (`set_high_qos`): turns off power throttling for the pool's threads so Windows 11 keeps them on P-cores. This gave ~624 → ~650 tok/s. It is compiled out on other platforms.

## Optional 8-bit mode (`-q 1`)
- Off by default; the default path calls the same float kernel as before, so `-s 42` output is unchanged.
- At load, all weights are quantized "Q8_0" style: groups of 32 (`Q8_GS`, must be 32) as int8 in [-127,127] plus a float scale. The int8 copy mirrors the float region element by element, so `matmul_rows_q8` finds a matrix's int8 copy by its float pointer's offset.
- Each matmul input is quantized once (`mm_prep`) before its pool job; `mm_rows` picks the float or int8 kernel. The dot product uses AVX-VNNI `dpbusd` when available, else `maddubs`+`madd`.
- Only matmuls change; attention, norms and the embedding lookup stay float.
- Real stories15M on a Linux cloud Xeon (4 cores), `-s 42`, 11 interleaved runs: ~910 → ~1900 tok/s median (~2.1×).
- Quality: when both modes are fed the same 256-token story, 8-bit picks the same top next token 98.8% of the time (252/255). Average log-prob of the story is -0.367 vs -0.364 for float. Stories read just as well, but they branch off from the float story after a few dozen tokens, so the output can't be checked against `tests/ref_s42.txt`.

## Findings
- Original was ~131 tok/s with 8 threads but 83 tok/s single-threaded: the scalar matmul wasn't vectorized, and OpenMP overhead was large.
- The classifier matmul (32000 × 288) is ~45% of the time and runs close to DRAM bandwidth (~74 GB/s). Further big gains need smaller weights (e.g. int8 quantization, see `runq.c`), which changes the output.
- Tried and reverted (no measurable win): software prefetch, a 64-byte-aligned weight buffer, adaptive chunk sizes, 8 accumulators in the matmul.
- SIMD sums add numbers in a different order than the original. Output for `-s 42` is identical, but other seeds/prompts could in theory pick a different token.
- On Linux the throttling code is off and timings are not comparable with the Windows laptop numbers.
- Windows laptop (2026-11-20, float path): ~680 tok/s median when cool, ~450-520 when hot/throttled, so always A/B.
- Per-token profile (float path): classifier ~40%, FFN up ~15%, QKV ~12%, W2 ~10%, softmax+top-p ~8%, attention ~7%, Wo ~6%. ~60 MB of fp32 weights stream from DRAM per token.
- Also rejected on the laptop (2026-11-20): software prefetch of the next 4-row block (-7% to -22% at distances 1/2/8); `POOL_CHUNK_ROWS` 16/32/128 (all slower than 64); 4 pool threads (clearly slower than 5); 6 threads (inconclusive: +8.6% in one 31-run A/B, tie in the next, so keep 5).
- Vectorized exp: UCRT `expf` is not correctly rounded (~23k mismatches vs double exp). An AVX2 double-precision exp that falls back to scalar `expf` within 1/256 ulp of a rounding midpoint was verified bit-identical over all 2^32 inputs, but isn't worth it: `expf` is only ~2.8 ns/call (32000 calls = 91 us serial, ~18 us on the pool) and the unpredictable fallback branch made the vector version slower (114 us). Call `_mm256_zeroupper()` before calling libm from AVX code.
