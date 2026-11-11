#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; O="$1"; DET="$2"; GEO="$D/../preds/test_geo_v4ntqdir3srngsr_d_vw_raw.csv"
{
  echo "id0,value"
  python3 "$D/rul260.py" "$DET" "$D/rows.csv"
  python3 "$D/macroscaleP.py" "$DET" "$GEO" "$D/rows.csv"
  python3 "$D/sim3dfarh.py" "$DET" "$D/cache/tmp" "$D/rows.csv"
} > "$O/pred_t018/B.csv"
