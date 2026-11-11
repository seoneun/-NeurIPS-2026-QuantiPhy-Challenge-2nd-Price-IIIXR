#!/bin/bash
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
OUT=$1; mkdir -p "$OUT/rows"; OUT=$(cd "$OUT" && pwd)
export QP_SUB="$OUT" QP_ROWS="$OUT/rows"
P="python3 -B"; RB=analysis/v3/calib/rebuild; BP=analysis/v3/calib/build_probe.py
cp "${PRED_T002:?}" "$OUT/pred_t002_id1.csv"
$P $RB/merge_rows.py "$OUT/pred_t002_id1.csv" $RB/pred_t046_rows.csv "$OUT/pred_t046_id1.csv"
for n in pred_t047 pred_t048 pred_t049 pred_t050 pred_t051 pred_t052 pred_t034 pred_t035 pred_t036 pred_t037 pred_t038 pred_t039 pred_t040 pred_t013 \
         pred_t041 pred_t042 pred_t043 pred_t044 pred_t006 pred_t007 pred_t045 pred_t008; do
  $P $BP $RB/$n.tsv
done
cp "$OUT/pred_t007_id1.csv" submissions/
$P analysis/v3/ag/rul351/build_r4.py
$P $BP cache/tmp/r4_fromcode/pred_t014.tsv
cp "$OUT/pred_t014_id1.csv" submissions/
$P analysis/v3/ag/r5_rebuild/rebuild_r5_blocks.py "$OUT"
$P $BP "$OUT/r5/pred_t009.tsv"
$P analysis/v3/ag/r5_rebuild3/rebuild_r5c_blocks.py "$OUT"
$P $BP "$OUT/r5c/pred_t010.tsv"
$P analysis/v3/ag/r5_rebuild4/rebuild_r5d_blocks.py "$OUT"
$P $BP "$OUT/r5d/pred_t011.tsv"
$P analysis/v3/ag/r6_rebuild/rebuild_r6_blocks.py "$OUT"
$P $BP "$OUT/r6/pred_t003.tsv"
