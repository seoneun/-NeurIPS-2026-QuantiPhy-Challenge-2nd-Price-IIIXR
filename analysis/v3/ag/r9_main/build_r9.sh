#!/bin/bash
set -euo pipefail
R=${QP_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}; cd "$R"
W=${QP_OUT:?}; HB=${QP_HBASE:?}; HS=${QP_HSRC:?}; M=analysis/v3/ag/r9_main; L=analysis/v3/calib/r9/levers
export QP_ROOT="$R" QP_OUT="$W" QP_HBASE="$HB" QP_HSRC="$HS"
C="python3 -B analysis/v3/ag/r8_main/compose.py"
mkdir -p "$W"
python3 -B $M/restore_ours.py "$HB" "$HS" "$W/pred_t163_id1.csv"
python3 -B $M/imports_r9.py "$W/pred_t163_id1.csv" "$HS" "$W" "${QP_PRED_T018:?}"
$C "$W/pred_t158_id1.csv" "$W/pred_t163_id1.csv" "$W/imp_blocks2.csv" "$W/imp_pred_t018_simsize.csv"
$C "$W/pred_t159_id1.csv" "$W/pred_t158_id1.csv" $L/r2sim_test.csv
python3 -B $M/imports_r9c.py "$W"
$C "$W/pred_t160_id1.csv" "$W/pred_t159_id1.csv" "$W/imp_q1.csv" "$W/imp_atrium.csv" "$W/imp_survey2.csv" \
  $L/r2seg_test.csv
python3 -B $M/imports_r9d.py "$W"
$C "$W/pred_t161_id1.csv" "$W/pred_t160_id1.csv" "$W/imp_q1c.csv"
python3 -B $M/imports_r9e.py "$W/pred_t161_id1.csv" "$W"
$C "$W/pred_t162_id1.csv" "$W/pred_t161_id1.csv" "$W/imp_r9e.csv"
python3 -B $M/imports_pred_t019.py "$W"
$C "$W/pred_t019_id1.csv" "$W/pred_t162_id1.csv" "$W/imp_q1d.csv"
