#!/bin/bash
set -euo pipefail
O=$(cd "$1" && pwd); RR=$PWD; H=$(cd "$(dirname "$0")" && pwd)
WD=$O/pred_t018/w; F=$WD/f; rm -rf "$WD"; mkdir -p "$F/preds" "$F/submissions" "$F/logs"
for e in "$RR"/*; do case "$(basename "$e")" in preds|submissions|logs|output|cache) ;; *) ln -s "$e" "$F/";; esac; done
ln -s "$RR"/preds/* "$F/preds/"
for u in "" u; do ln -s "$H/preds/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" "$F/preds/test_geo_v4ntqdir3${u}srngsr_d_vw_raw.csv"; done
QP_ROOT=$F bash "$H/fusion.sh" > "$WD/fusion.log" 2>&1
FS=$F/submissions
bash "$H/B/run.sh" "$O" "$FS/pred_t056_detail.csv"
bash "$H/C/run.sh" "$O" "$FS/pred_t056_detail.csv" > "$WD/C.log"
bash "$H/D/run.sh" "$O" "$FS"
python3 -B "$H/merge.py" "$FS/pred_t056_id1.csv" "$O/pred_t018/pred_t018_values.csv" "$O/pred_t018/B.csv" "$O/pred_t018/C.csv" "$O/pred_t018/D.csv"
rm -rf "$WD"
