#!/bin/bash
# runs rul072 against the sr86 (sub_xt_Nzz2) patch list
R="${QP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"; cd "$R"
PL=$(grep -o '"Nzz2||--patch_json [^"]*"' cache/tmp/sr86.sh | sed 's/.*--patch_json //; s/"$//')
C24TAT_PATCHES="$PL" OMP_NUM_THREADS=2 python analysis/v3/ag/rul072/rul072.py "$@"
