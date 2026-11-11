#!/bin/bash
set -e
G=$(cd "$(dirname "$0")" && pwd); O=$1
python3 "$G/build_g4.py" "$G" "$O/pred_t132_id1.csv" "$O/pred_t128_id1.csv" "$O/pred_t127_id1.csv" "$O" > "$O/log_build_g4.txt"
