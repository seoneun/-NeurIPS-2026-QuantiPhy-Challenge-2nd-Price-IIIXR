#!/bin/bash
set -euo pipefail
O=$(cd "$1" && pwd); FDET=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
H=$(cd "$(dirname "$0")" && pwd); L=$PWD/team/member1/trackB/package/layers; Q=$L/61_62_common/strict_inputs/analysis/v3/compliance
WK=$O/pred_t018/wC; OV=$WK/ov; B=pred_t064
rm -rf "$WK"; mkdir -p "$OV/cache/tmp" "$OV/submissions" "$OV/preds" "$OV/geo/out" "$OV/analysis/v3/compliance"
ln -s "$H/analysis/v3/ag" "$OV/analysis/v3/ag"
ln -s "$Q/qtrkredir" "$Q/qtrack" "$OV/analysis/v3/compliance/"
ln -s "$PWD"/geo/*.py "$OV/geo/"
ln -s "$PWD/geo/out/parse_test_v2.jsonl" "$H"/geo/out/tracks_test_* "$OV/geo/out/"
ln -s "$H/../preds/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" "$OV/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv"
ln -s "$FDET" "$OV/submissions/${B}_detail.csv"
cd "$OV"
export QP_ROOT=$OV
python3 -B analysis/v3/ag/rul272/gen_e1x.py $B
python3 -B analysis/v3/ag/rul058/rul058.py $B
python3 -B analysis/v3/ag/rul197/rul197.py $B
QT="env QP_QTRK_ROOT=$OV QP_QTRK_CSUF=_mq3 PYTHONPATH=$OV/analysis/v3/compliance/qtrkredir"
$QT QP_QTRK_LAYER=0 python3 -B analysis/v3/ag/rul221/rule.py $B
mv cache/tmp/ag_rul221_patch_test.json cache/tmp/ag_d02d3edgekey0_patch_test.json
$QT QP_QTRK_LAYER=2 python3 -B analysis/v3/ag/rul221/rule.py $B
mv cache/tmp/ag_rul221_patch_test.json cache/tmp/ag_d02d3edgekey2_patch_test.json
$QT QP_QTRK_LAYER=0 python3 -B analysis/v3/ag/rul045/rul045.py $B
$QT QP_QTRK_LAYER=7 python3 -B analysis/v3/ag/rul056/rul056.py $B
python3 -B analysis/v3/ag/derive_late/rul057.py $B
python3 -B "$H/collect.py" "$H/rows.tsv" "$OV/cache/tmp" "$O/pred_t018/C.csv"
rm -rf "$WK"
