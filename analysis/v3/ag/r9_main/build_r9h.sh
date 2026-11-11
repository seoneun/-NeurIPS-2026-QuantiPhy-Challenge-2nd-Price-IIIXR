#!/bin/bash
set -euo pipefail
R=${QP_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}; cd "$R"
W=${QP_OUT:?}; M=analysis/v3/ag/r9_main; C="python3 -B analysis/v3/ag/r8_main/compose.py"
cp "${QP_PRED_T026:?}" "$W/pred_t026_id1.csv"
python3 -B $M/imports_r9h45b.py "$W" "$W/pred_t026_id1.csv" pred_t164 pred_t027
$C "$W/pred_t164_id1.csv" "$W/pred_t026_id1.csv" "$W/imp_pred_t164.csv"
$C "$W/pred_t027_id1.csv" "$W/pred_t164_id1.csv" "$W/imp_pred_t027.csv"
