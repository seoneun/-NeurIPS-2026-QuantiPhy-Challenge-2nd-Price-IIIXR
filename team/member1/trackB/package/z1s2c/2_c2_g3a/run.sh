#!/bin/bash
set -e
RR=$(pwd); G=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$G"); O=$1
C2CSV=$PKG/1_s1_c2/inputs/orig/pred_t015_id1.csv
F=$O/s2root; rm -rf "$F"; mkdir -p "$F/cache"; cp -as "$G/root/analysis" "$F/analysis"; cp -as "$PKG/1_s1_c2/root" "$F/analysis/v3/ag/r10/cl"
ln -s "$RR/preds" "$F/preds"; ln -s "$RR/patches" "$F/cache/tmp"
C=$F/analysis/v3/ag/r10/cl; W=$F/analysis/v3/ag/r10/work
for s in chain c9u r36audit g2 g3; do
  (cd "$C" && env QP_ROOT="$C" QP_WORK="$W" QP_R="$F" C2CSV="$C2CSV" PYTHONDONTWRITEBYTECODE=1 python3 "$G/py/$s.py" "$O") > "$O/log_$s.txt" 2>&1
done
