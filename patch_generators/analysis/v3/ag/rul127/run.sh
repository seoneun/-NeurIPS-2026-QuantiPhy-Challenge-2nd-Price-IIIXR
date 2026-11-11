#!/bin/bash
# rul127: tedge's uncut-run cross-check from the box centre when the mask-centroid slope drifts (see rule.py)
# usage: bash analysis/v3/ag/rul127/run.sh [base submission, default sub_xt_Nzz18]
R="${QP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"; cd "$R"
RL() { for i in $(seq 1 180); do OMP_NUM_THREADS=2 "$@" </dev/null; rc=$?; [ $rc -ne 75 ] && return $rc; sleep 10; done; }
for S in val test; do RL python analysis/v3/ag/rul127/diag.py $S || exit 1; done
RL python analysis/v3/ag/rul127/rule.py ${1:-sub_xt_Nzz18} 0.03 rul127
