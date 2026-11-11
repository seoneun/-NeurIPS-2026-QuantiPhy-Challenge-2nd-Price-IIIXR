#!/bin/bash
# rul116: rigid-length centre rebuild for border-cut dist2 targets (A) + rul439 positional-qualifier exception (B).
# needs cache/tmp/ag_c66freeedge_{diag,patch}_{split} (run rul117 first). Patch order: ... rul117, rul116, c63plainkey
R="${QP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"; cd "$R"
for S in val test; do OMP_NUM_THREADS=2 python analysis/v3/ag/rul116/diagA.py $S > cache/tmp/ag_c66d2edge_diagA_$S.log 2>&1 || exit 1; done
OMP_NUM_THREADS=2 python analysis/v3/ag/rul116/rule.py > cache/tmp/ag_c66d2edge_rule.log 2>&1
tail -40 cache/tmp/ag_c66d2edge_rule.log
