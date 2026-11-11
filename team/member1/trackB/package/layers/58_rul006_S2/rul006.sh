#!/bin/bash
set -euo pipefail
mkdir -p "$1"; O=$(cd "$1" && pwd); H1=$2; OUT=$3
H=$(cd "$(dirname "$0")" && pwd); Q=$H/strict_inputs; C=$H/../61_62_common/strict_inputs; OV=$O/ov
mkdir -p "$OV/cache/tmp" "$OV/submissions" "$OV/preds" "$OV/geo/out" "$OV/analysis/v3/compliance" "$OV/analysis/v3/ag" "$O/L6/patches"
ln -s "$Q/analysis/v3/compliance/rules" "$C/analysis/v3/compliance/qtrkredir" "$C/analysis/v3/compliance/qtrack" "$OV/analysis/v3/compliance/"
ln -s "$Q/analysis/v3/ag/rul006" "$OV/analysis/v3/ag/"
ln -s "$Q"/geo/out/tracks_* "$PWD/geo/out/parse_test_v2.jsonl" "$OV/geo/out/"
ln -s "$H/../65_rul302/inputs/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" "$OV/preds/"
for v in m v; do ln -s "$Q/cache/tmp/ag_qtrk_meas_test_${v}_qdir3.csv" "$OV/cache/tmp/ag_ownt_meas_test_${v}_ownt.csv"; done
python3 -B "$H/mkbase.py" "$H/inputs" "$H1" "$OV/submissions"
(cd "$OV" && QP_QTRK_ROOT=$OV QP_QTRK_LAYER=6 QP_QTRK_CSUF=_mq3 PYTHONPATH=$OV/analysis/v3/compliance/qtrkredir \
  python3 -B analysis/v3/compliance/rules/bindmoverS3.py pred_t031 qdir3) > "$O/bindmoverS3.L6.log"
mv "$OV/cache/tmp/ag_bindmoverS3_patch_test.json" "$O/L6/patches/"
(cd "$OV" && python3 -B analysis/v3/compliance/qtrkredir/layer_merge.py "$O" 7 bindmoverS3) > "$O/merge.log"
python3 -B "$H/mkcand.py" "$OV/cache/tmp/ag_bindmoverS3_patch_test.json" "$H/inputs" "$OUT"
