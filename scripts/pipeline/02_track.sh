#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail
stage_upstream

(
  export QP_TEST_VID=$QP_ROOT/data/test
  for S in val test; do gpu logged track_v1_$S.log $PY geo/track.py --split $S --shard 0 --nshards 1; done
  logged measure_v1_test.log $PY geo/measure.py --split test
  logged measure_v1_val.log $PY geo/measure.py --split val
  for S in val test; do logged affected_v2_$S.log $PY geo/affected_v2.py $S; done
  for S in val test; do gpu logged track_v2_$S.log $PY geo/track_v2.py --split $S --video_list geo/out/retrack_${S}_v2.txt; done
  logged calib_owl_v2.log $PY geo/calib_owl_v2.py
)

mkdir -p third_party
[ -d third_party/MoGe ] || { git clone https://github.com/microsoft/MoGe.git third_party/MoGe && \
  git -C third_party/MoGe checkout 74fbce054ebed49800de42d0ad0e83495065719a; }
[ -d third_party/utils3d-moge ] || { git clone https://github.com/EasternJournalist/utils3d-moge.git third_party/utils3d-moge && \
  git -C third_party/utils3d-moge checkout 62f09d58509485564e24d5d9f6aac9ee9ebc0c37; }
for S in val test; do gpu logged moge_$S.log $PY geo/moge_run.py --split $S; done
VAL_VIDS=$($PY -c "import csv; print(','.join(sorted({r['video_id'].strip() for r in csv.DictReader(open('data/val/validation_dataset.csv'))})))")
TEST_VIDS=$($PY -c "import csv; print(','.join(sorted(set(r['video_id'].strip() for r in csv.DictReader(open('caw/inputs/test_input.csv'))))))")
gpu logged vlmground_val_notext.log $PY geo/vlm_ground.py --split val --notext --out_tag _notext
gpu logged track_v3nt_val.log $PY geo/track_v3.py --split val --vg_tag _notext --out_tag v3nt --videos "$VAL_VIDS"
gpu logged vlmground_test_notext.log $PY geo/vlm_ground.py --split test --notext --out_tag _notext
gpu logged track_v3nt_test.log $PY geo/track_v3.py --split test --vg_tag _notext --out_tag v3nt --videos "$TEST_VIDS"

for S in val test; do
  gpu logged molmoground_$S.log env VLLM_ENABLE_V1_MULTIPROCESSING=0 QP_ALLOW_BUSY_GPU=1 $PY geo/molmo_ground.py --split $S
  gpu logged sam2pts_${S}_v2.log $PY geo/sam2_points.py --split $S --src v2
  gpu logged sam2pts_${S}_v3nt.log $PY geo/sam2_points.py --split $S --src v3nt
done

for S in val test; do
  logged mover_scan_$S.log env MV_EXTRA_WHYS=target_low_det,target_no_track,prior_low_det,prior_no_track $PY geo/mover_scan.py --split $S
  gpu logged sam2_movers_$S.log $PY geo/sam2_movers.py --split $S
  logged patch_movers_${S}_v2mv.log    env PM_SAMECLASS_SKIP=0 $PY geo/patch_movers.py --split $S --src v2mo   --out v2mv
  logged patch_movers_${S}_v3ntmv.log  env PM_SAMECLASS_SKIP=0 $PY geo/patch_movers.py --split $S --src v3ntmo --out v3ntmv
  logged patch_movers_${S}_v2mvu.log   env PM_SAMECLASS_SKIP=0 $PY geo/patch_movers.py --split $S --src v2     --out v2mvu
  logged patch_movers_${S}_v3ntmvu.log env PM_SAMECLASS_SKIP=0 $PY geo/patch_movers.py --split $S --src v3nt   --out v3ntmvu
done

gpu logged sa2va_val_first.log $PY geo/sa2va_track.py --split val --keys first --out _first
gpu logged sa2va_test_first.log $PY geo/sa2va_track.py --split test --filter scene_weak --keys first --out _first
for S in val test; do
  rm -rf geo/out/tracks_${S}_v2sc geo/out/tracks_${S}_v2scu
  logged patch_sa2va_${S}_v2sc.log  $PY geo/patch_sa2va.py --split $S --mode scene --src v2mv  --out v2sc  --in _first
  logged patch_sa2va_${S}_v2scu.log $PY geo/patch_sa2va.py --split $S --mode scene --src v2mvu --out v2scu --in _first
done

for S in val test; do logged scene_ruler_$S.log $PY geo/scene_ruler.py --split $S; done
echo "02_track done"
