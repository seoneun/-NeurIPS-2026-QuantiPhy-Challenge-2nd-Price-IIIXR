#!/bin/bash
set -euo pipefail
B=$1; O=$2; D=analysis/v3/ag/r9_main/levers/revisit; W=output/work/revisit
mkdir -p "$W"
for v in captured_0008 captured_0008s captured_0008x; do python3 -B analysis/v3/ag/r9_main/levers/labtime/otrack.py $v "$W/trk_$v.csv"; done
python3 -B $D/build.py "$B" "$W" "$O"
