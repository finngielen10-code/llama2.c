#!/bin/bash
# Build runq.c with the project's standard flags, check that the -s 42 output on the
# TinyLlama chat model matches tests/ref_q_s42.txt, then report median tok/s over N runs.
# usage: scripts/benchq.sh [N=5] [src=runq.c]
set -u
cd "$(dirname "$0")/.."
OUT=build/benchq; N=${1:-5}; SRC=${2:-runq.c}
MODEL=${MODEL:-tinyllama_chat_q80.bin}; STEPS=${STEPS:-128}
mkdir -p "$OUT"
gcc -O3 -fopenmp -march=native -fpermissive -I. "$SRC" win.c -lm -o "$OUT/benchq.exe" 2>"$OUT/build.log" \
  || { echo "BUILD FAIL"; grep -i error "$OUT/build.log" | head -20; exit 1; }
"$OUT/benchq.exe" "$MODEL" -s 42 -n "$STEPS" > "$OUT/out.txt" 2>/dev/null
cmp -s "$OUT/out.txt" tests/ref_q_s42.txt \
  || { echo "OUTPUT MISMATCH vs tests/ref_q_s42.txt"; diff "$OUT/out.txt" tests/ref_q_s42.txt | head; exit 2; }
for i in $(seq "$N"); do
  "$OUT/benchq.exe" "$MODEL" -s 42 -n "$STEPS" 2>&1 >/dev/null | grep -o '[0-9.]*$'
done | sort -n > "$OUT/runs.txt"
awk '{a[NR]=$1; s+=$1} END{printf "output OK | median %.2f  mean %.2f  min %.2f  max %.2f tok/s (n=%d)\n", a[int((NR+1)/2)], s/NR, a[1], a[NR], NR}' "$OUT/runs.txt"
