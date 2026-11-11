#!/bin/bash
R="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"; export QP_ROOT="$R"
TAG=${1:?tag}; PY=${2:-${PY:-${PYTHON:-python}}}
O=$R/caw/out/val_$TAG
mkdir -p $O
export TORCHCODEC_NUM_THREADS=4
echo "[caw] $(date '+%F %T') tag=$TAG python=$(command -v $PY)"
$PY -c "import vllm, transformers, torch, qwen_vl_utils; print('[caw] vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
cd $R/caw/code-as-world
$PY -m code_as_world.evaluation 9b --model-path $R/models/Code-as-World-VL-9B \
  --input-csv $R/data/val/validation_dataset.csv --video-dir $R/caw/val_videos \
  --output-csv $O/9b.csv --out-json $O/9b.json --generations-log $O/9b.generations.jsonl
echo "[caw] $(date '+%F %T') rc=$?"
