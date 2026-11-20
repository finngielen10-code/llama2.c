#!/bin/bash
# Interleaved A/B benchmark: A = a previous version of run.c, B = current run.c.
# Interleaving cancels out thermal/background drift, which is large on this laptop.
# usage: scripts/ab.sh <old_run.c | git-rev> [N=21]
#   e.g. scripts/ab.sh HEAD        (compare working tree against last commit)
set -u
cd "$(dirname "$0")/.."
OUT=build/ab; N=${2:-21}
mkdir -p "$OUT"
if [ -f "$1" ]; then cp "$1" "$OUT/a_run.c"; else git show "$1:run.c" > "$OUT/a_run.c" || exit 1; fi
gcc -O3 -fopenmp -march=native -fpermissive "$OUT/a_run.c" win.c -I. -lm -o "$OUT/A.exe" 2>/dev/null || { echo "A BUILD FAIL"; exit 1; }
gcc -O3 -fopenmp -march=native -fpermissive run.c win.c -lm -o "$OUT/B.exe" 2>"$OUT/build.log" || { echo "B BUILD FAIL"; grep -i error "$OUT/build.log"; exit 1; }
"$OUT/B.exe" stories15M.bin -s 42 2>/dev/null | cmp -s - tests/ref_s42.txt || { echo "OUTPUT MISMATCH (B)"; exit 2; }
: > "$OUT/a.txt"; : > "$OUT/b.txt"
for i in $(seq "$N"); do
  "$OUT/A.exe" stories15M.bin -s 42 2>&1 >/dev/null | grep -o '[0-9.]*$' >> "$OUT/a.txt"
  "$OUT/B.exe" stories15M.bin -s 42 2>&1 >/dev/null | grep -o '[0-9.]*$' >> "$OUT/b.txt"
done
for f in a b; do
  printf "%s: " "$(echo $f | tr ab AB)"
  sort -n "$OUT/$f.txt" | awk '{a[NR]=$1; s+=$1} END{printf "median %.1f  mean %.1f tok/s\n", a[int((NR+1)/2)], s/NR}'
done
