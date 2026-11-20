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
`win.c`/`win.h` provide mmap/clock shims for Windows (`clock_gettime` was renamed to `win_clock_gettime` locally to avoid clashing with the MinGW one).

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
- Linux Xeon, random weights shaped like stories15M: ~920 → ~1700 tok/s (~1.8×). Logits differ from float by ~2% (relative L2), and the top token matched in the tested positions. Output text differs from the float run, so it can't be checked against `tests/ref_s42.txt`.

## Findings
- Original was ~131 tok/s with 8 threads but 83 tok/s single-threaded: the scalar matmul wasn't vectorized, and OpenMP overhead was large.
- The classifier matmul (32000 × 288) is ~45% of the time and runs close to DRAM bandwidth (~74 GB/s). Further big gains need smaller weights (e.g. int8 quantization, see `runq.c`), which changes the output.
- Tried and reverted (no measurable win): software prefetch, a 64-byte-aligned weight buffer, adaptive chunk sizes, 8 accumulators in the matmul.
- SIMD sums add numbers in a different order than the original. Output for `-s 42` is identical, but other seeds/prompts could in theory pick a different token.
- On Linux the throttling code is off and timings are not comparable with the Windows laptop numbers.
