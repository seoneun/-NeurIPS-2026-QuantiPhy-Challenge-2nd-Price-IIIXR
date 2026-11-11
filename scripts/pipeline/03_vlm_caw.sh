#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail
stage_upstream
export TORCHCODEC_NUM_THREADS=4
G16="g16:0.01:0.001:-1:1:0:16"
[ -d caw/code-as-world ] || { git clone https://github.com/mirros-lab/code-as-world caw/code-as-world && \
  git -C caw/code-as-world checkout 1353bf07d24e5463caff92ce23ffd34d03984831; }

gpu logged caw_val_env030.log bash caw/run_val_official.sh env030
logged caw_analyze_val.log $PY caw/analyze_caw.py val caw/out/val_env030 env030

gpu logged caw_test_orig.log bash caw/sc/run_test_orig.sh

gpu logged crops_val.log env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY caw/sc/make_crops.py val
gpu logged caw_val_crop.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/val_crop.csv --video-dir cache/tmp/crops_val \
    --out-dir caw/sc/out/val_crop --chunk-videos 100 --configs $G16
gpu logged crops_test.log env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY caw/sc/make_crops.py test 2.0 0.4
gpu logged caw_test_crop.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/test_crop.csv --video-dir cache/tmp/crops_test \
    --out-dir caw/sc/out/test_crop --chunk-videos 60 --configs $G16
logged build_caw.log bash -c "$PY caw/sc/build_caw_crop.py val && $PY caw/sc/build_caw_orig.py && $PY caw/sc/build_caw_crop.py test single caw9b_orig"
cp preds/val_caw9b_crop.csv preds/val_caw9b_orig_crop.csv

logged sib_inputs.log env CUDA_VISIBLE_DEVICES= $PY caw/sc/build_sibling_inputs.py
gpu logged caw_test_sib.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/test_sib.csv --video-dir data/test \
    --out-dir caw/sc/out/test_sib --chunk-videos 60 --configs $G16

logged tshift_input.log env CUDA_VISIBLE_DEVICES= $PY analysis/v3/mk_tshift_input.py
gpu logged caw_test_tshift.log $PY caw/sc/run_caw_sc.py --input-csv caw/inputs/test_tshift.csv --video-dir data/test \
    --out-dir caw/sc/out/test_tshift --configs $G16
$PY caw/sc/filter_tshift30.py

for S in val test; do logged ruler_q_$S.log env CUDA_VISIBLE_DEVICES= $PY analysis/v3/ruler_build.py $S q; done
gpu logged caw_val_ruler.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/val_ruler.csv --video-dir caw/val_videos \
    --out-dir caw/sc/out/val_ruler --chunk-videos 30 --configs $G16
gpu logged caw_test_ruler.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/test_ruler.csv --video-dir data/test \
    --out-dir caw/sc/out/test_ruler --chunk-videos 30 --configs $G16

gpu logged caw_test_t1n16.log $PY caw/sc/run_caw_sc.py --input-csv caw/inputs/test_input.csv --video-dir data/test \
    --out-dir caw/sc/out/test_t1n16 --chunk-videos 30 --configs t1n16:1.0:1.0:-1:16:0
gpu logged caw_test_crop_t1n16.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/test_crop.csv --video-dir cache/tmp/crops_test \
    --out-dir caw/sc/out/test_crop_t1n16 --chunk-videos 60 --configs t1n16:1.0:1.0:-1:16:0
gpu logged caw_val_crop_t1n16.log $PY caw/sc/run_caw_sc.py --input-csv caw/sc/inputs/val_crop.csv --video-dir cache/tmp/crops_val \
    --out-dir caw/sc/out/val_crop_t1n16 --chunk-videos 60 --configs t1n16:1.0:1.0:-1:16:0

gpu logged qwen_val.log $PY tools/run_vlm.py --split val --mode video --out preds/val_qwen3vl8b_video.csv
mkdir -p preds/shards
for G in 0 1 2 3; do
  gpu logged qwen_test_s$G.log $PY tools/run_vlm.py --split test --mode video --shard $G --nshards 4 --out preds/shards/test_qwen3vl8b_video_s$G.csv
done
gpu $PY tools/merge_shards.py "preds/shards/test_qwen3vl8b_video_s?.csv" preds/test_qwen3vl8b_video.csv 3289
for S in val test; do logged heuristic_$S.log $PY tools/heuristic.py --split $S; done
echo "03_vlm_caw done"
