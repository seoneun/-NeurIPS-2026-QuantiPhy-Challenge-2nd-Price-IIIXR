#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export QP_PATCH_DIR="${QP_PATCH_DIR:-$QP_ROOT/patches}"
unset QP_R QP_RECIPE QP_DIMGM_BOUND POOL_KINDS QP_TIGHT_CAW QP_STRONG_EXTRA QP_TEST_VID QP_GMFIX QP_SANE
mkdir -p submissions
source scripts/fusion_recipe.sh
fusion_pair pred_t002 rj5sQ2 "$P_PRED_T002" "${X48A[@]}"
