#!/bin/bash
set -euo pipefail
R=${QP_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}; cd "$R"
SE=$R/team/member1/trackB; W=${QP_OUT:?}; D=$W/d23; L=analysis/v3/calib/r9/levers; M=analysis/v3/ag/r9_main
C="python3 -B analysis/v3/ag/r8_main/compose.py"
rm -rf "$D"; mkdir -p "$D/hbuild"; cp -r "$SE/build/." "$D/hbuild/"
echo "id0,value,rule" > "$D/hbuild/inputs/cands/23_rul218.csv"
(cd "$D/hbuild" && python3 build_h8.py inputs/layers_pred_t017.txt "$D/pred_t079_id1.csv" "$D/pred_t079_id1_src.csv" > /dev/null)
QP_ROOT="$R" QP_OUT="$D" QP_HBASE="$D/pred_t079_id1.csv" QP_HSRC="$D/pred_t079_id1_src.csv" bash $M/build_r9.sh > "$D/build_r9.log"
cp "$D/pred_t019_id1.csv" "$W/pred_t020_id1.csv"
python3 -B $M/imports_r9i.py "$W"
$C "$W/pred_t165_id1.csv" "$W/pred_t020_id1.csv" "$W/imp_r9i.csv"
$C "$W/pred_t166_id1.csv" "$W/pred_t165_id1.csv" $L/q1e_test.csv
python3 -B $M/imports_r9k.py "$W" "${QP_PRED_T018:?}"
$C "$W/pred_t167_id1.csv" "$W/pred_t166_id1.csv" "$W/imp_r9k.csv"
$C "$W/pred_t168_id1.csv" "$W/pred_t167_id1.csv" $L/q1g_fix.csv
$C "$W/pred_t169_id1.csv" "$W/pred_t168_id1.csv" $L/q1h_fix.csv
python3 -B $M/imports_r9n.py "$W"
$C "$W/pred_t028_id1.csv" "$W/pred_t169_id1.csv" "$W/imp_r9n.csv"
