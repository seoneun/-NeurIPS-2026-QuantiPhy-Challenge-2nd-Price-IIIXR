#!/bin/bash
set -euo pipefail
source "$QP_ROOT/scripts/common.sh"
source scripts/fusion_recipe.sh
unset QP_R QP_RECIPE QP_DIMGM_BOUND POOL_KINDS QP_TIGHT_CAW QP_STRONG_EXTRA QP_TEST_VID QP_GMFIX QP_SANE
DROP_V=" --var_caw --sib_caw --event_geo --c2_anchor --var_geo_plain --var_ans_plain --lab_pool --scene_prior --d3lab_focal --scene_prior_ext --disp_bound "
DROP_F=" --xrow_prior --cqc --var_norm "
strip() { local out=() skip=0 a; for a in "$@"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    if [[ "$DROP_V" == *" $a "* ]]; then skip=1; continue; fi
    if [[ "$DROP_F" == *" $a "* ]]; then continue; fi
    out+=("$a"); done; printf '%s\0' "${out[@]}"; }
mapfile -d '' X30S < <(strip "${X30[@]}")
mapfile -d '' FLBS < <(strip "${FLB[@]}")
for u in "" u; do
  QP_SANE=h1only python3 analysis/v3/submit_v3.py geo_v4ntqdir3${u}srngsr_d_vw pred_t056_x$u "${X30S[@]}" "${FLBS[@]}" --no_c1 --patch_json "" "${X48A[@]}" \
    --d3lab_focal "854x480=0.5959,853x480=0.5959:1.3:gm:gm"
done
python3 tools/mo_guard.py pred_t056 pred_t056_x pred_t056_xu
