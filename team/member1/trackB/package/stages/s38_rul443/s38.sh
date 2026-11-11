#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; R="$(pwd)"
WK="$1"; H37="$2"; OUT="$3"; shift 3
rm -rf "$WK"; mkdir -p "$WK/root/submissions" "$WK/patches"; WK="$(cd "$WK" && pwd)"
for e in analysis caw data geo preds scripts tools; do ln -s "$R/$e" "$WK/root/$e"; done
ln -s "$R"/patches/*.json "$D"/patches/*.json "$WK/patches/"
for t in rul456:rul183 rul457:rul192 rul458:rul229 rul459:rul230 rul460:rul231 rul461:rul234; do
  ln -s "$R/patches/ag_${t#*:}_patch_test.json" "$WK/patches/ag_${t%%:*}_patch_test.json"
done
(
  export QP_ROOT="$WK/root" QP_PATCH_DIR="$WK/patches"
  source "$QP_ROOT/scripts/common.sh"
  unset QP_R QP_RECIPE QP_DIMGM_BOUND POOL_KINDS QP_TIGHT_CAW QP_STRONG_EXTRA QP_TEST_VID QP_GMFIX QP_SANE
  source scripts/fusion_recipe.sh
  declare -A DROP=([--var_caw]=1 [--sib_caw]=1 [--event_geo]=1 [--cqc]=0 [--xrow_prior]=0 [--c2_anchor]=1 [--var_geo_plain]=1
                   [--var_ans_plain]=1 [--lab_pool]=1 [--scene_prior]=1 [--scene_prior_ext]=1 [--disp_bound]=1)
  filt() { local -n arr=$1; local skip=0 x
    for x in "${arr[@]}"; do
      if [ $skip -gt 0 ]; then skip=$((skip - 1)); continue; fi
      if [ -n "${DROP[$x]+y}" ]; then skip=${DROP[$x]}; continue; fi
      printf '%s\0' "$x"; done; }
  mapfile -d '' X30 < <(filt X30); mapfile -d '' FLB < <(filt FLB)
  P47=$(grep -vxF -f "$D/inputs/cross_tags.txt" "$D/inputs/tags_pred_t114.txt" | paste -sd,)
  P48=$(tr ',' '\n' <<< "$P_PRED_T002" | grep -vxF -f "$D/inputs/cross_tags.txt" | paste -sd,)
  fusion_pair pred_t193 rj5sQ2 "$P47" --probe "S2:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;S3:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;D2:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D3:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D2:agree_wgm:size:w0.7;S2:agree_wgm,agree_gm:size:geo:simulation;S3:agree_gm:disp:geo:captured" \
    --rul263_scale simulation:1.1:-height,simulation:1.05:+height,captured:0.8
  fusion_pair pred_t194 rj5sQ2 "$P48" "${X48A[@]}" --calib_fuse analysis/v3/calib/fusion_params.json
)
python3 -B "$D/table.py" "$WK/root/submissions" "$D/inputs/rows.csv" "$H37" "$OUT" "$@"
rm -rf "$WK"
