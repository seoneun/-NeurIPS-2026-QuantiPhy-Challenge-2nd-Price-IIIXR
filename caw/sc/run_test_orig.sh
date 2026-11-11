#!/bin/bash
R="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"; export QP_ROOT="$R"
cd $R
export TORCHCODEC_NUM_THREADS=4
"${PY:-${PYTHON:-python}}" caw/sc/run_caw_sc.py --input-csv caw/inputs/test_input.csv --video-dir data/test --out-dir caw/sc/out/test_orig --chunk-videos 30 --configs "g16:0.01:0.001:-1:1:0:16"
echo TESTORIG_DONE
