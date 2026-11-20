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

## Findings so far (stories15M)
- Original: ~131 tok/s median (8 threads); 83 tok/s single-threaded → the scalar matmul was not vectorized.
- AVX2/FMA matmul (4 rows × 8 lanes, `hsum256`) kept output identical.
- OpenMP fork/join dominates for this tiny model: parallelize only matmuls with `d*n >= MATMUL_PAR_THRESHOLD` (1M, i.e. just the classifier) and keep attention serial (`ATTN_PAR_MIN_POS`). ~280 tok/s.
- Precomputing RoPE cos/sin once per token (not per layer): small, exact win (~+3%).
- Fewer threads (2 or 4) for the classifier was slower than the default 8.
- Open idea: sampler `softmax` runs 32000 serial `expf` per token; a vectorized exp may change output bits.
