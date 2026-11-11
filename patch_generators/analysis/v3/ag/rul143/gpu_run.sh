#!/bin/bash
# reproduce the teammate's scene3d / kin2d GPU + CPU stages in the rul143 work tree (copies of origin/ours-scene3d-kin2d)
set -x
W=${QP_ROOT:-.}/analysis/v3/ag/rul143/wt; cd $W
${PYTHON:-python}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
RT=$(cat logs/test_rows_scene3d.txt)
$PY scripts/ours/scene3d.py ground  --split test --rows $RT
$PY scripts/ours/scene3d.py geom    --split test --rows $RT --tag _detS
$PY scripts/ours/scene3d.py measure --split test --rows $RT --tag _detS --scale v2
$PY scripts/ours/kin2d.py --split test --stage locate  --tag _testS2 --rows logs/test_rows_kin2d.txt
$PY scripts/ours/kin2d.py --split test --stage measure --tag _testS2 --rows logs/test_rows_kin2d.txt
RV=$(cat logs/val_rows_scene3d.txt)
$PY scripts/ours/scene3d.py ground  --split val --rows $RV
$PY scripts/ours/scene3d.py geom    --split val --rows $RV --tag _detS1
$PY scripts/ours/scene3d.py measure --split val --rows $RV --tag _detS1 --scale v2
$PY scripts/ours/kin2d.py --split val --stage locate  --tag _v5dS1 --rows logs/val_rows_kin2d.txt
$PY scripts/ours/kin2d.py --split val --stage measure --tag _v5dS1 --rows logs/val_rows_kin2d.txt
echo ALLDONE
