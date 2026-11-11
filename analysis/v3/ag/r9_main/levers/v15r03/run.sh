#!/bin/bash
set -euo pipefail
B=$1; O=$2; D=analysis/v3/ag/r9_main/levers/v15r03
p=(); for t in v2sc v3ntmv; do for f in abcdef abdef; do python3 -B $D/meas_v15.py test $t $f > /dev/null & p+=($!); done; done
for i in "${p[@]}"; do wait "$i"; done
python3 -B $D/build.py "$B" "$O"
