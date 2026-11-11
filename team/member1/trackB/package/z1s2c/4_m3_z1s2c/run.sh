#!/bin/bash
set -e
G=$(cd "$(dirname "$0")" && pwd); O=$1; M3=$O/pred_t139_id1.csv
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR AVX512VL AVX512BW AVX512DQ AVX512_FP16 AVX512VNNI AVX512IFMA AVX512VBMI AVX512VBMI2 AVX512BITALG AVX512VPOPCNTDQ AVX512_BF16"
python3 "$G/py/build_y1.py" "$O" "$M3"
python3 "$G/py/build_z1.py" "$O" "$M3"
python3 "$G/py/build_z1s2.py" "$O"
python3 "$G/py/build_z1s2c.py" "$O" "$M3"
