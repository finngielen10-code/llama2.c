#!/bin/bash
# Interleaved A/B of two runq.c versions on the TinyLlama chat model (alternating runs,
# so background load hits both equally). Both must reproduce tests/ref_q_s42.txt first.
# usage: scripts/abq.sh <old_runq.c|git-rev> [N=7]     env: STEPS (48), MODEL, OMP_NUM_THREADS
set -u
cd "$(dirname "$0")/.."
OUT=build/abq; N=${2:-7}; STEPS=${STEPS:-48}; MODEL=${MODEL:-tinyllama_chat_q80.bin}
mkdir -p "$OUT"
if [ -f "$1" ]; then cp "$1" "$OUT/A.c"; else git show "$1:runq.c" > "$OUT/A.c" || exit 1; fi
cp runq.c "$OUT/B.c"
for v in A B; do
  gcc -O3 -fopenmp -march=native -fpermissive -I. "$OUT/$v.c" win.c -lm -o "$OUT/$v.exe" 2>"$OUT/$v.log" \
    || { echo "BUILD FAIL $v"; grep -i error "$OUT/$v.log" | head; exit 1; }
  "$OUT/$v.exe" "$MODEL" -s 42 -n 128 2>/dev/null | cmp -s - tests/ref_q_s42.txt \
    || { echo "OUTPUT MISMATCH in $v"; exit 2; }
done
: > "$OUT/A.txt"; : > "$OUT/B.txt"
for i in $(seq "$N"); do
  for v in A B; do
    "$OUT/$v.exe" "$MODEL" -s 42 -n "$STEPS" 2>&1 >/dev/null | grep -o '[0-9.]*$' >> "$OUT/$v.txt"
  done
done
med() { sort -n "$1" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
a=$(med "$OUT/A.txt"); b=$(med "$OUT/B.txt")
awk -v a="$a" -v b="$b" -v n="$N" 'BEGIN{printf "outputs OK | A median %.2f  B median %.2f tok/s  B/A = %.3f (n=%d each)\n", a, b, b/a, n}'
