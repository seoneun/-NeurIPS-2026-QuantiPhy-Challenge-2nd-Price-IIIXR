#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail
need_tracks tracks_val tracks_test tracks_val_v2sc tracks_test_v2sc tracks_val_v2scu tracks_test_v2scu \
  tracks_val_v3ntmv tracks_test_v3ntmv tracks_val_v3ntmvu tracks_test_v3ntmvu moge_val moge_test
stage_upstream
GPU_ENV=(); [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && GPU_ENV=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
export CUDA_VISIBLE_DEVICES=""

for S in val test; do
  logged ruler_p_$S.log $PY analysis/v3/ruler_build.py $S p
  for k in 0 1 2; do
    logged meas_${S}_rl$k.log $PY geo/measure_v3.py --split $S --fix abcdef --tag _rl$k --parse_tag _rl$k --moge --moge_opts fbrej
    $PY geo/arbitrate_v2.py --split $S --meas_tag _rl$k --tag geo_rl$k --quiet
  done
done

export QP_REJ_LO=0.2 QP_REJ_HI=5 QP_PRIOR_DRIFT=1.5 QP_PRIOR_ACCGUARD=3 QP_SCENE_HEAD=1 QP_TSHIFT=1
export QP_DEPTH_DEDUP=1 QP_SCENE_SA2VA=1 QP_SECOND_FIX=1 QP_SAMECLASS=1 QP_SCENE_RULER=1 QP_FF_ROT=1 QP_SIDE_FIX=1 QP_FLOOR=1 \
       QP_STATIC_CAP=25 QP_FF_SUB=1 QP_FF_SUB_SRC=simulation QP_PARSE2=1 QP_NEAREST=1 QP_BELT=1
SELF="RL_SELF=1 RL_SELF_NOGRAV=1 RL_SELF_MODE=2d_selfruler RL_OVER_VLM=1"

chain() {
  local T=$1 TM=$2 TV=$3; shift 3
  for S in val test; do
    logged meas_${S}_v3m_r$T.log env "$@" $PY geo/measure_v3.py --split $S --fix abcdef --tag _v3m_r$T --moge --moge_opts fbrej --tracks $TM
    logged meas_${S}_v3v_rnt$T.log env "$@" $PY geo/measure_v3.py --split $S --fix abcdef --tag _v3v_rnt$T --moge --moge_opts fbrej --tracks $TV
    env "$@" $PY geo/arbitrate_v2.py --split $S --meas_tag _v3m_r$T --tag geo_v3m_r$T --quiet
    env "$@" $PY geo/arbitrate_v2.py --split $S --meas_tag _v3v_rnt$T --tag geo_v3v_rnt$T --quiet
    logged fuse_${S}_$T.log env "$@" $PY analysis/v3/fuse_v3.py $S _v3m_r$T _v3v_rnt$T geo_v4nt$T
    logged xground_${S}_$T.log env "$@" XG_VG=_notext XG_RV=geo_v3v_rnt$T XG_BASE=geo_v4nt${T}_d_vw XG_OUT=geo_v4nt${T}g $PY analysis/v3/xground.py $S
  done
  for S in val test; do
    logged ruler_m2_${S}_$T.log env "$@" $SELF RL_BASE_M2=geo_v4nt${T}g_d_vw RL_OUT=geo_v4nt${T}srng $PY analysis/v3/ruler_build.py $S m2
  done
  for S in val test; do logged sruler_${S}_$T.log env "$@" $PY analysis/v3/sruler_apply.py $S $T; done
}

chain rj5sd  v2sc  v3ntmv
chain rj5sdu v2scu v3ntmvu
for P in "rj5sd rj5sdG" "rj5sdu rj5sdGu"; do
  set -- $P; A=$1; B=$2
  for S in val test; do
    for X in "" _raw; do cp preds/${S}_geo_v4nt${A}g_d_vw$X.csv preds/${S}_geo_v4nt${B}g_d_vw$X.csv; done
    cp geo/out/meas_${S}_v2_v3m_r${A}.csv geo/out/meas_${S}_v2_v3m_r${B}.csv
    cp geo/out/meas_${S}_v2_v3v_rnt${A}.csv geo/out/meas_${S}_v2_v3v_rnt${B}.csv
    logged ruler_m2_${S}_$B.log env $SELF RL_SELF_GRAV=1 RL_SELF_GRAV_MODE=2d_gself RL_BASE_M2=geo_v4nt${B}g_d_vw RL_OUT=geo_v4nt${B}srng \
      $PY analysis/v3/ruler_build.py $S m2
    logged sruler_${S}_$B.log $PY analysis/v3/sruler_apply.py $S $B
  done
done

for S in val test; do gpu logged person_focal_$S.log env -u CUDA_VISIBLE_DEVICES ${GPU_ENV[@]+"${GPU_ENV[@]}"} $PY geo/person_focal.py $S; done

G="RL_SELF_GRAV=1 RL_SELF_GRAV_MODE=2d_gself"
chain rj5sQ  v2sc  v3ntmv  $G QP_FPERSON=1 QP_FPERSON_MINN=2
chain rj5sQu v2scu v3ntmvu $G QP_FPERSON=1 QP_FPERSON_MINN=2

for S in val test; do
  logged personf_merge_$S.log $PY geo/personf_merge.py $S geo_v4ntrj5sQsrngsr_d_vw  geo_v4ntrj5sdGsrngsr_d_vw  geo_v4ntrj5sQ2srngsr_d_vw
  logged personf_merge_u_$S.log $PY geo/personf_merge.py $S geo_v4ntrj5sQusrngsr_d_vw geo_v4ntrj5sdGusrngsr_d_vw geo_v4ntrj5sQ2usrngsr_d_vw
done
echo "04_geo_chain done: preds/{val,test}_geo_v4ntrj5sQ2{,u}srngsr_d_vw_raw.csv"
