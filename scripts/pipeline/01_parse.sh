#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail
stage_upstream
for S in val test; do gpu logged parse_$S.log $PY geo/parse.py --split $S; done
for S in val test; do logged parse_v2_$S.log $PY geo/parse_v2.py --split $S; done
echo "01_parse done"
