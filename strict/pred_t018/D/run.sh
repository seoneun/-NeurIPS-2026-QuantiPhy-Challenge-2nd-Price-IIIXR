#!/bin/bash
set -euo pipefail
O=$(cd "$1" && pwd); FS=$(cd "$2" && pwd); RR=$PWD; H=$(cd "$(dirname "$0")" && pwd)
P=$RR/team/member1/trackB/package/layers; Q58=$P/58_rul006_S2/strict_inputs; C=$P/61_62_common/strict_inputs
WD=$O/pred_t018/wD; OV=$WD/ov; GEO=$H/../preds/test_geo_v4ntqdir3srngsr_d_vw_raw.csv; rm -rf "$WD"
mkdir -p "$OV/cache/tmp" "$OV/submissions" "$OV/preds" "$OV/geo/out" "$OV/analysis/v3/compliance/rules" "$OV/analysis/v3/ag"
ln -s "$H"/rules/*.py "$Q58/analysis/v3/compliance/rules/bindmoverS3.py" "$OV/analysis/v3/compliance/rules/"
ln -s "$C/analysis/v3/compliance/qtrkredir" "$C/analysis/v3/compliance/qtrack" "$OV/analysis/v3/compliance/"
ln -s "$Q58/analysis/v3/ag/rul006" "$OV/analysis/v3/ag/"
ln -s "$C"/geo/*.py "$OV/geo/"
ln -s "$H"/geo/out/tracks_* "$RR/geo/out/parse_test_v2.jsonl" "$OV/geo/out/"
ln -s "$RR/preds/test_caw9b_orig.csv" "$RR/preds/test_caw9b_orig_crop.csv" "$OV/preds/"
for u in "" u; do ln -s "$GEO" "$OV/preds/test_geo_v4ntqdir3${u}srngsr_d_vw_raw.csv"; done
ln -s "$FS"/pred_t056_detail.csv "$FS"/pred_t056_id1.csv "$OV/submissions/"
for v in m v; do ln -s "$Q58/cache/tmp/ag_qtrk_meas_test_${v}_qdir3.csv" "$OV/cache/tmp/ag_ownt_meas_test_${v}_ownt.csv"
  ln -s "$H/cache/tmp/ag_lab3d_meas_test_${v}_refP_q3L3.csv" "$OV/cache/tmp/ag_lab3d_meas_test_${v}_refP.csv"
  for t in ref flab; do ln -s "$H/cache/tmp/ag_lab3d_dump_test_${v}_${t}P_q3L3.jsonl" "$OV/cache/tmp/ag_lab3d_dump_test_${v}_${t}P.jsonl"; done; done
L=(env QP_QTRK_ROOT=$OV QP_QTRK_CSUF=_mq3 PYTHONPATH=$OV/analysis/v3/compliance/qtrkredir)
cd "$OV"
python3 -B analysis/v3/compliance/rules/strict_final.py submissions/pred_t056_id1.csv submissions/pred_t056_detail.csv pred_t058 > "$WD/rules.log"
"${L[@]}" QP_QTRK_LAYER=0 python3 -B analysis/v3/compliance/rules/rul440.py test >> "$WD/rules.log"
python3 -B analysis/v3/compliance/rules/rul442.py pred_t056 qdir3 >> "$WD/rules.log"
"${L[@]}" QP_QTRK_LAYER=1 python3 -B analysis/v3/compliance/rules/bindmoverS3.py pred_t056 qdir3 >> "$WD/rules.log"
python3 -B analysis/v3/compliance/rules/rul259.py pred_t056 qdir3 >> "$WD/rules.log"
python3 -B analysis/v3/compliance/rules/rul340.py pred_t056 pred_t056 qdir3 >> "$WD/rules.log"
python3 -B analysis/v3/compliance/rules/rul260.py pred_t056 pred_t056 >> "$WD/rules.log"
python3 -B "$H/mkcur.py" submissions/pred_t056_id1.csv cache/tmp/ag_rul260_patch_test.json submissions/pred_t057_id1.csv
python3 -B analysis/v3/compliance/rules/rul336.py pred_t056 pred_t057 qdir3 >> "$WD/rules.log"
python3 -B analysis/v3/compliance/rules/rul335.py pred_t056 pred_t057 >> "$WD/rules.log"
cd "$RR"
python3 -B "$H/pred_t018_D.py" "$OV" "$O/pred_t018/D.csv"
rm -rf "$WD"
