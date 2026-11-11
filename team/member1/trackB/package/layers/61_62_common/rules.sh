#!/bin/bash
set -euo pipefail
mkdir -p "$1"; O=$(cd "$1" && pwd); B=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
H=$(cd "$(dirname "$0")" && pwd); Q=$H/strict_inputs; OV=$O/ov; BASE=$(basename "$B" _detail.csv)
mkdir -p "$OV/cache/tmp" "$OV/submissions" "$OV/preds" "$OV/geo/out"
ln -s "$Q/analysis" "$OV/analysis"
ln -s "$Q"/geo/*.py "$OV/geo/"
ln -s "$Q"/geo/out/tracks_* "$PWD/geo/out/parse_test_v2.jsonl" "$OV/geo/out/"
ln -s "$H/../65_rul302/inputs/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" "$OV/preds/"
ln -s "$B" "$OV/submissions/"
TAGS=""; NL=0
for spec in "$@"; do
  t=${spec%%:*}; TAGS="$TAGS $t"
  ks=${spec#*:}
  for K in ${ks//,/ }; do
    mkdir -p "$O/L$K/patches"; NL=$(( K + 1 > NL ? K + 1 : NL ))
    (cd "$OV" && QP_QTRK_ROOT=$OV QP_QTRK_LAYER=$K QP_QTRK_CSUF=_mq3 PYTHONPATH=$OV/analysis/v3/compliance/qtrkredir \
      python3 -B "analysis/v3/compliance/rules/$t.py" "$BASE" qdir3) > "$O/$t.L$K.log"
    mv "$OV/cache/tmp/ag_${t}_patch_test.json" "$O/L$K/patches/"
  done
done
(cd "$OV" && python3 -B analysis/v3/compliance/qtrkredir/layer_merge.py "$O" $NL $TAGS) > "$O/merge.log"
