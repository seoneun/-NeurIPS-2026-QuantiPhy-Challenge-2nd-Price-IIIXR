#!/bin/bash
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd); cd "$R"
O=$R/output; rm -rf "$O"; mkdir -p "$O/logs"; export PYTHONDONTWRITEBYTECODE=1
P="@QP""_ROOT@"
{ grep -rlF "$P" --include='*.py' --include='*.sh' --include='*.json' --include='*.txt' --include='*.csv' --include='*.tsv' . || true; } | { grep -v '^\./run\.sh$' || true; } \
  | xargs -r -d '\n' sed -i "s#$P#$R#g"
SE=team/member1/trackB; SF=team/member1_v2/trackB/build/final

echo "1. stored model-stage outputs"
tar -xJf intermediates/intermediates.tar.xz

echo "2. measurement and fusion"
bash scripts/fusion.sh > "$O/logs/2.log" 2>&1

echo "3. class rule blocks"
PRED_T002="$R/submissions/pred_t002_id1.csv" bash scripts/rules.sh "$O/rules" > "$O/logs/3.log" 2>&1
cp "$O/rules/pred_t003_id1.csv" submissions/pred_t003_id1.csv
cp "$O/rules/pred_t003_id1.csv" "$SE/build/inputs/pred_t003_id1.csv"

echo "4. base table of the class layers"
bash "$SE/package/z1s2c/run_all.sh" "$O/z1s2c" > "$O/logs/4z.log" 2>&1
cp "$O/z1s2c/pred_t004_id1.csv" "$SE/build/inputs/pred_t004_id1.csv"
H1=$O/h1/a/b; mkdir -p "$H1/blk/analysis/v3/calib" "$H1/blk/submissions"
for f in pred_t006 pred_t007 pred_t008 pred_t009 pred_t010 pred_t011 pred_t003; do cp "$O/rules/rows/${f}_rows.csv" "$H1/blk/analysis/v3/calib/"; done
cp analysis/v3/calib/pred_t012_rows.csv "$H1/blk/analysis/v3/calib/"
for f in pred_t003 pred_t013 pred_t007 pred_t014; do cp "$O/rules/${f}_id1.csv" "$H1/blk/submissions/"; done
python3 -B "$SE/package/cands5/r4r5/build_r4r5_rows.py" "$O/rules/rows" analysis/v3/calib/r5/held_bundle_rows.csv "$SE/build/inputs/r4r5_rows.csv"
cp "$SE/build/inputs/rows_status.csv" "$SE/build/inputs/r4r5_rows.csv" "$H1/"; cp "$SE/build/inputs/pred_t004_id1.csv" "$O/h1/"
(cd "$H1" && python3 -I "$R/$SE/build/inputs/orig/h1.py" > "$O/logs/4.log" 2>&1)
cp "$H1/h1_rows.csv" "$SE/build/inputs/h1_rows.csv"

echo "5. class layers"
bash "$SE/package/layers/build_layers.sh" "$O" "$O/layers" > "$O/logs/5l.log" 2>&1
for f in 01_h2 06_sr_S2 07_sr_D2 09_sr_D3 12_labF 13_rul279 21_rul431 24_rul408 26_sc3d_D3 32_rul310 40_rul437 43_rul409 55_sport_S3 59_rul342_S2 64b_rul326 65_rul302 67_rul004 69_rul429 70_top2_A 75_xlv2 76_rul453 80_rul427 02_aud_tag 03b_aud_geo_z 33_len_jointtrack 35_border opt_A_copyfix opt_B_h1chain_to_z1s2c 45_sr2_S2 51_rul407_A 53_sport_S2 58_rul006_S2 61_rr_S2 62_rr_S3 04_imp_D2 44_rul292 52_rul407_B; do cp "$O/layers/$f.csv" "$SE/build/inputs/cands/"; done
python3 -B team/member3/pkg/chain/build_chain.py "$SE/package/z1s2c/1_s1_c2/inputs/orig/pred_t015_id1.csv" "$O/m3/chain" > "$O/logs/5c.log" 2>&1
cp "$O/m3/chain/74_member3_jm.csv" "$SE/build/inputs/cands/74_member3_jm.csv"; cp "$O/m3/chain/survey2_test.csv" analysis/v3/calib/r9/levers/survey2_test.csv
mkdir -p "$O/hline"; cp -r "$SE/build/." "$O/hline/build/"
(cd "$O/hline/build" && python3 -B build_h8.py inputs/layers_pred_t017.txt "$O/hline/pred_t017_id1.csv" "$O/hline/pred_t017_id1_src.csv" > "$O/logs/5.log")

echo "6. import rules"
bash strict/pred_t018/run.sh "$O" > "$O/logs/6s.log" 2>&1
python3 -B team/member2/trackB/package/audit/build_teamline.py "$O/rules" team/member2/trackB/audit/teamline_candidates.csv "$O/pred_t018/pred_t018_values.csv" > "$O/logs/6t.log" 2>&1
QP_ROOT="$R" QP_PRED_T018="$O/pred_t018/pred_t018_values.csv" QP_OUT="$O/pred_t019" QP_HBASE="$O/hline/pred_t017_id1.csv" QP_HSRC="$O/hline/pred_t017_id1_src.csv" \
  bash analysis/v3/ag/r9_main/build_r9.sh > "$O/logs/6.log"

echo "7. import rules on the layer file without layer 23"
QP_ROOT="$R" QP_PRED_T018="$O/pred_t018/pred_t018_values.csv" QP_OUT="$O" bash analysis/v3/ag/r9_main/build_r9g.sh > "$O/logs/7.log"

echo "8. step tables"
V=team/member2/trackB/package/vf; mkdir -p "$O/steps"; cp "$O/pred_t019/pred_t019_id1.csv" "$O/steps/"
python3 -B team/member3/pkg/s42/build_s42.py "$O/m3/chain/pred_t021_id1.csv" "$O/hline/pred_t017_id1.csv" "$O/hline/pred_t017_id1_src.csv" \
  "$O/pred_t019/pred_t019_id1.csv" "$SF/steps/s42_r9qpg.csv" "$O/pred_t018/pred_t018_values.csv" > "$O/logs/8s.log" 2>&1
python3 -B "$SE/package/stages/s37_rul309/rul309.py" "$O/pred_t019/pred_t019_id1.csv" "$O/hline/pred_t017_id1_src.csv" "$SF/steps/s37_rul309.csv"
python3 -B "$SF/apply_steps.py" "$O/steps" s37 > "$O/logs/8.log"
bash "$SE/package/stages/s38_rul443/s38.sh" "$O/w38" "$O/steps/pred_t022_id1.csv" "$SF/steps/s38_rul443_allowed2_vrest.csv" \
  "$SF/steps/s38_stored_rows.csv" > "$O/logs/8u.log" 2>&1
python3 -B "$SF/apply_steps.py" "$O/steps" s38 s39 s40 s41 s42 >> "$O/logs/8.log"
python3 -B "$V/build_pred_t023.py" "$O/rules/pred_t003_id1.csv" analysis/v3/calib/r7/cls/rows_status.csv "$O/hline/pred_t017_id1_src.csv" \
  "$O/steps/pred_t024_id1.csv" "$V/inputs/vggt_kin_rows.csv" "$O/steps/pred_t023_id1.csv" >> "$O/logs/8.log" 2>&1
python3 -B "$V/build_h42vfc.py" "$O/steps/pred_t023_id1.csv" "$SE/build/inputs/pred_t004_id1.csv" "$O/steps/pred_t025_id1.csv"
python3 -B "$SE/package/stages/s46_rul410/s46.py" "$O/steps/pred_t023_id1.csv" "$SE/build/inputs/pred_t004_id1.csv" analysis/v3/calib/r9/b89/b89x_test.csv \
  "$SF/steps/s46_rul410.csv"
python3 -B "$SF/apply_steps.py" "$O/steps" s46 >> "$O/logs/8.log"

echo "9. own-input re-measurements"
QP_OUT="$O" QP_PRED_T026="$O/steps/pred_t026_id1.csv" bash analysis/v3/ag/r9_main/build_r9h.sh > "$O/logs/9.log"

echo "10. refinement tables"
bash scripts/apply_refinements.sh "$O/refine" "$O/pred_t027_id1.csv" "$O/pred_t028_id1.csv" "$O/pred_t018/pred_t018_values.csv" > "$O/logs/10.log" 2>&1
cp "$O/refine/pred_t001_id1.csv" "$O/sub_xt_r9mF_id1.csv"
echo "predictions: $O/sub_xt_r9mF_id1.csv"
