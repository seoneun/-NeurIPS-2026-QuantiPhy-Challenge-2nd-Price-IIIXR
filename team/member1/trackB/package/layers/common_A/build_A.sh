#!/bin/bash
set -euo pipefail
O=$1; L=$2; D=$(cd "$(dirname "$0")" && pwd); B=$D/../../../build
H1=$O/h1/a/b; Z=$O/h1/pred_t004_id1.csv; W=$L/wA; I=$W/inputs
rm -rf "$W"; mkdir -p "$I/cands"
cp "$B/build_h8.py" "$W/"; cp "$O/rules/pred_t003_id1.csv" "$Z" "$H1/h1_rows.csv" "$I/"
for f in 01_h2 04_imp_D2 05_imp_S3 06_sr_S2 07_sr_D2 08_sr_S3 09_sr_D3 10_rul288 11_rul287 12_labF 13_rul279 14_readmit 15_rul216 \
         16b_rul455 19_rul007 20_dseg 21_rul431 22_rul430 23_rul218 24_rul408 26_sc3d_D3 27_sc3d_S3 28_rul261 29_rul263 \
         30_rul003 31_rul002 32_rul310 34_rul311; do
  if [ -f "$L/$f.csv" ]; then cp "$L/$f.csv" "$I/cands/"; else cp "$B/inputs/cands/$f.csv" "$I/cands/"; fi
done
cp "$D/16_xd3sim.csv" "$I/cands/"
python3 -B "$D/aud_tag.py" "$H1" "$O/rules/rows" "$L/02_aud_tag.csv"
python3 -B "$D/aud_geo.py" "$H1" "$W"
python3 -B "$D/mid.py" 03b "$H1/h1_rows.csv" "$Z" "$W/03_aud_geo.csv" "$L/03b_aud_geo_z.csv"
python3 -B "$D/mid.py" 33 "$H1/h1_rows.csv" "$W/rows.csv" "$L/33_len_jointtrack.csv"
cp "$L/02_aud_tag.csv" "$W/03_aud_geo.csv" "$L/03b_aud_geo_z.csv" "$L/33_len_jointtrack.csv" "$I/cands/"
(cd "$W" && python3 -B build_h8.py "$D/h8_layers.txt" pred_t016_id1.csv pred_t016_id1_src.csv > /dev/null)
python3 -B "$D/opt.py" "$I/pred_t003_id1.csv" "$Z" "$H1/h1_rows.csv" "$W/pred_t016_id1.csv" "$W/pred_t016_id1_src.csv" \
  "$L/opt_A_copyfix.csv" "$L/opt_B_h1chain_to_z1s2c.csv"
cp "$L/opt_A_copyfix.csv" "$L/opt_B_h1chain_to_z1s2c.csv" "$I/cands/"
(cd "$W" && python3 -B build_h8.py "$D/h16L_layers.txt" pred_t067_id1.csv pred_t067_id1_src.csv > /dev/null)
python3 -B "$D/mid.py" 35 "$H1/h1_rows.csv" "$W/rows.csv" "$W/pred_t067_id1.csv" "$D/gray_gate.csv" "$L/35_border.csv"
rm -r "$W"
