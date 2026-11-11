#!/bin/bash
set -euo pipefail
O=$1; L=$2; D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$L"
H1=$O/h1/a/b/h1_rows.csv; Z=$O/h1/pred_t004_id1.csv
python3 -B "$D/01_h2/mk_h2.py" "$H1" "$D/01_h2/S_rows_verdict.csv" "$Z" "$L/01_h2.csv"
python3 -B "$D/12_labF/make_cand.py" "$D/12_labF/tab_fixed.pkl" "$D/12_labF/free_rows_model.csv" "$D/12_labF/pred_t084_id1.csv" "$D/12_labF/h5_src.csv" "$L/12_labF.csv"
python3 -B "$D/13_rul279/rul279.py" "$O/rules/pred_t048_id1.csv" "$O/rules/pred_t035_id1.csv" "$H1" "$L/13_rul279.csv"
python3 -B "$D/21_rul431/tb_halfframe.py" "$D/21_rul431/pred_t086_id1.csv" "$L/21_rul431.csv"
python3 -B "$D/24_rul408/make_24.py" "$D/24_rul408/restore_A_rows.csv" "$H1" "$L/24_rul408.csv"
python3 -B "$D/32_rul310/mk_32.py" "$H1" "$D/32_rul310/grayver_restore_B.csv" "$D/32_rul310/rs_r4r5_restore.csv" "$L/32_rul310.csv"
python3 -B "$D/40_rul437/ident_gvp1.py" "$D/40_rul437/parse_test_v2_rl0.jsonl" "$D/40_rul437/aud_geo_rows.csv" "$Z" "$L/40_rul437.csv"
python3 -B "$D/43_rul409/rul409_gm.py" "$D/43_rul409/main.jsonl" "$D/43_rul409/pred_t066_id1.csv" "$H1" "$L/43_rul409.csv"
python3 -B "$D/59_rul342_S2/rul342.py" "$D/59_rul342_S2/rows_all.csv" "$D/59_rul342_S2/pred_t071_id1.csv" "$Z" "$H1" "$L/59_cand_h24.csv" "$L/59_rul342_S2.csv"
rm "$L/59_cand_h24.csv"
python3 -B "$D/64b_rul326/labtab.py" "$Z" "$L/64_rul326.csv"
grep -v -E "^(2682|2688|2694)," "$L/64_rul326.csv" > "$L/64b_rul326.csv"; rm "$L/64_rul326.csv"
python3 -B "$D/67_rul004/make_cand.py" "$D/67_rul004/pred_t073_id1.csv" "$L/67_rul004.csv"
python3 -B "$D/69_rul429/make_cand.py" data/test/test_dataset.parquet "$D/69_rul429/pred_t074_id1.csv" "$L/69_rul429.csv"
python3 -B "$D/70_top2_A/top2A.py" "$D/70_top2_A/humscaleS.json" "$D/70_top2_A/pred_t075_id1.csv" "$D/70_top2_A/pred_t075_id1_src.csv" "$D/70_top2_A/ls1_rows.csv" "$H1" "$L/70_top2_A.csv"
python3 -B "$D/76_rul453/make_cand.py" "$D/76_rul453/pred_t076_id1.csv" "$H1" "$L/76_rul453.csv"
python3 -B "$D/80_rul427/regen_80.py" "$H1" "$D/80_rul427/rows_audit_1024.csv" "$D/../../build/inputs/cands/79V_rul449.csv" \
  "$D/80_rul427/pred_t078_id1.csv" "$D/80_rul427/pred_t078_id1_src.csv" "$L/80_rul427.csv"
SR=$D/sr_common; X=(env QP_WORK=$SR H1ROWS=$H1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2)
mkdir -p "$L/w06" "$L/w07" "$L/w09"
"${X[@]}" OUTDIR="$L/w06" python3 -B "$D/06_sr_S2/sr.py" > "$L/w06/sr.log"
python3 -B "$SR/mk_cand_cols.py" "$L/w06/all_changes.csv" "$L/w06/cand.csv"
python3 -B "$SR/mk_layer.py" "$L/w06/cand.csv" "$L/06_sr_S2.csv"
"${X[@]}" OUTDIR="$L/w07" python3 -B "$D/07_sr_D2/sr.py" > "$L/w07/sr.log"
python3 -B "$SR/mk_cand_cols.py" "$L/w07/cand_all.csv" "$L/w07/cand.csv" crlf
python3 -B "$SR/mk_layer.py" "$L/w07/cand.csv" "$L/07_sr_D2.csv"
"${X[@]}" OUTDIR="$L/w09" python3 -B "$D/09_sr_D3/srd3.py" > "$L/w09/srd3.log"
"${X[@]}" OUTDIR="$L/w09" python3 -B "$D/09_sr_D3/mk.py"
python3 -B "$SR/mk_layer.py" "$L/w09/cand.csv" "$L/09_sr_D3.csv"
rm -r "$L/w06" "$L/w07" "$L/w09"
python3 -B "$D/26_sc3d_D3/c1.py" "$D/26_sc3d_D3/m1" "$L/26_sc3d_D3.csv" "$D/43_rul409/pred_t066_id1.csv"
mkdir -p "$L/w55"; H1ROWS=$H1 OMP_NUM_THREADS=2 python3 -B "$D/55_sport_S3/fin.py" "$L/w55" > "$L/w55/fin.log"
grep -v "^1376," "$L/w55/cand.csv" > "$L/55_sport_S3.csv"; rm -r "$L/w55"
I=$D/65_rul302/inputs
python3 -B "$D/65_rul302/make_cand.py" "$I/pred_t032_detail.csv" "$I/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" geo/out/parse_test_v2.jsonl \
  "$I/groups_test.json" "$I/tracks_test_qv3ntmv_mq3" "$I/pred_t072_id1.csv" "$I/pred_t072_id1_src.csv" "$L/65_rul302.csv"
python3 -B "$D/75_xlv2/bird2.py" "$D/76_rul453/pred_t076_id1.csv" "$L/75_bird_min.csv"
python3 -B "$D/75_xlv2/pl2.py" "$D/76_rul453/pred_t076_id1.csv" "$L/75_pl_c.csv" > /dev/null
(cat "$L/75_bird_min.csv"; tail -n +2 "$L/75_pl_c.csv") > "$L/75_xlv2.csv"; rm "$L/75_bird_min.csv" "$L/75_pl_c.csv"
E45=$D/45_sr2_S2; mkdir -p "$L/w45"
python3 -B "$E45/s2mchain.py" "$E45/pred_t170_detail.csv" "$E45/pred_t170_id1.csv" "$D/65_rul302/inputs/test_geo_v4ntqdir3srngsr_d_vw_raw.csv" \
  "$E45/ag_qtrk_meas_test_m_qdir3.csv" "$E45/ag_qtrk_meas_test_v_qdir3.csv" geo/out/parse_test_v2.jsonl "$L/w45/s2mchain.json"
python3 -B "$E45/s2mdiscaw.py" "$E45/pred_t170_detail.csv" "$E45/pred_t170_id1.csv" geo/out/parse_test_v2.jsonl "$L/w45/s2mdiscaw.json"
python3 -B "$E45/make_cand.py" "$E45/pred_t068_id1.csv" "$E45/pred_t068_id1_src.csv" "$E45/pred_t170_detail.csv" "$L/45_sr2_S2.csv" \
  s2mdiscaw="$L/w45/s2mdiscaw.json" s2mchain="$L/w45/s2mchain.json" rul317="$E45/ag_rul317_patch_test.json" rul316="$E45/ag_rul316_patch_test.json"
rm -r "$L/w45"
mkdir -p "$L/w53"
MKL_CBWR=COMPATIBLE python3 -B "$D/53_sport_S2/rul434.py" analysis/v3/calib/audit_test.csv geo/out/parse_test_v2.jsonl geo/out/file_fps_test.json \
  "$D/53_sport_S2/tracks" data/test_480p "$L/w53/rul434.json"
python3 -B "$D/53_sport_S2/mk_53.py" "$L/w53/rul434.json" "$L/53_sport_S2.csv"
rm -r "$L/w53"
mkdir -p "$L/w51"
bash "$D/51_rul407_A/mkroot.sh" "$D/51_rul407_A/strict_inputs" "$D/51_rul407_A/from_R.txt" "$L/w51/qr"
E51=(env QP_ROOT="$PWD" RA_Q="$L/w51/qr" RA_OUT="$L/w51" RA_MODS="$D/51_rul407_A/mods")
"${E51[@]}" python3 -B "$D/51_rul407_A/py/flatw2own.py" pred_t092 > "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/c45own.py" pred_t105 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/c53own.py" pred_t149 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/c65own.py" pred_t100 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/c79own.py" pred_t107 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/rules/rul290.py" pred_t093 rj5sQ2 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/rules/rul206.py" pred_t108 rj5sQ2 >> "$L/w51/log.txt"
"${E51[@]}" python3 -B "$D/51_rul407_A/py/rules/rul023.py" pred_t094 rj5sQ2 >> "$L/w51/log.txt"
python3 -B "$D/51_rul407_A/mklayer.py" "$L/w51" "$L/51_rul407_A.csv"
rm -r "$L/w51"
bash "$D/61_62_common/rules.sh" "$L/w61" "$D/61_62_common/inputs/pred_t033_detail.csv" c80mastlenS2:0,1
python3 -B "$D/61_62_common/sel.py" "$L/w61" "$D/61_rr_S2/ids.csv" "$L/61_rr_S2.csv"
bash "$D/61_62_common/rules.sh" "$L/w62" "$D/65_rul302/inputs/pred_t032_detail.csv" c45wspanS2:0 disc3dS2:3
python3 -B "$D/61_62_common/sel.py" "$L/w62" "$D/62_rr_S3/ids.csv" "$L/62_rr_S3.csv"
bash "$D/58_rul006_S2/rul006.sh" "$L/w58" "$H1" "$L/58_rul006_S2.csv"
rm -r "$L/w61" "$L/w62" "$L/w58"
python3 -B "$D/04_imp_D2/mk.py" "$Z" "$D/04_imp_D2/strict_inputs/submissions/pred_t142_detail.csv" "$H1" "$L/04_imp_D2.csv"
mkdir -p "$L/w52"
bash "$D/51_rul407_A/mkroot.sh" "$D/52_rul407_B/strict_inputs" "$D/52_rul407_B/from_R.txt" "$L/w52/qr"
python3 -B "$D/52_rul407_B/final.py" "$L/w52/qr" "$Z" "$L/52_rul407_B.csv"
rm -r "$L/w52"
python3 -B "$D/44_rul292/cand.py" geo/out/parse_test_v2.jsonl "$H1" "$D/45_sr2_S2/pred_t068_id1_src.csv" \
  "$D/44_rul292/strict_inputs/geo/out/meas_test_v2_belt_test_v3ntmo.csv" "$D/44_rul292/strict_inputs/geo/out/tracks_test_v3ntmv" \
  geo/out/tracks_test_v3ntmv "$L/44_rul292.csv"
bash "$D/common_A/build_A.sh" "$O" "$L"
