#!/bin/bash
# Build run.c with the project's standard flags, check that the -s 42 output
# matches tests/ref_s42.txt, then report median tok/s over N runs.
# usage: scripts/bench.sh [N=11]
set -u
cd "$(dirname "$0")/.."
OUT=build/bench; N=${1:-11}
mkdir -p "$OUT"
gcc -O3 -fopenmp -march=native -fpermissive run.c win.c -lm -o "$OUT/bench.exe" 2>"$OUT/build.log" \
  || { echo "BUILD FAIL"; grep -i error "$OUT/build.log" | head -20; exit 1; }
"$OUT/bench.exe" stories15M.bin -s 42 > "$OUT/out.txt" 2>/dev/null
cmp -s "$OUT/out.txt" tests/ref_s42.txt \
  || { echo "OUTPUT MISMATCH vs tests/ref_s42.txt"; diff "$OUT/out.txt" tests/ref_s42.txt | head; exit 2; }
for i in $(seq "$N"); do
  "$OUT/bench.exe" stories15M.bin -s 42 2>&1 >/dev/null | grep -o '[0-9.]*$'
done | sort -n > "$OUT/runs.txt"
awk '{a[NR]=$1; s+=$1} END{printf "output OK | median %.1f  mean %.1f  min %.1f  max %.1f tok/s (n=%d)\n", a[int((NR+1)/2)], s/NR, a[1], a[NR], NR}' "$OUT/runs.txt"
