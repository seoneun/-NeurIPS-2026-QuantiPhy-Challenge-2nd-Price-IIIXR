#!/bin/bash
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
O=$1; B=$2; N=$3; PRED_T018=$4; mkdir -p "$O"
L=analysis/v3/ag/r9_main/levers; T=analysis/v3/calib/r9/levers; W=output/work
C="python3 -B analysis/v3/ag/r8_main/compose.py"
python3 -B analysis/v3/calib/r9/focal/imp_focal.py "$N" $T/imp_focal_test.csv
$C "$O/pred_t171_id1.csv" "$B" $T/imp_focal_test.csv
$C "$O/pred_t172_id1.csv" "$O/pred_t171_id1.csv" $T/vfx_test.csv
python3 -B analysis/v3/calib/r9/b89imp/build_b89imp.py submissions/pred_t002_detail.csv $T/b89imp_test.csv
$C "$O/pred_t173_id1.csv" "$O/pred_t172_id1.csv" $T/b89imp_test.csv
mkdir -p $W/vbS
python3 -B analysis/v3/calib/r9/vb/S/code/tip17.py $W/vbS/tip17.txt
python3 -B analysis/v3/calib/r9/vb/S/code/build_vbS.py $W/vbS/tip17.txt $T/vbS_test.csv
$C "$O/pred_t174_id1.csv" "$O/pred_t173_id1.csv" $T/vbS_test.csv
mkdir -p $W/lev108
python3 -B team/member3/pkg/lev108/build_lev108.py team/member1/trackB/build/inputs/pred_t004_id1.csv $W/lev108 "$PRED_T018"
$C "$O/pred_t175_id1.csv" "$O/pred_t174_id1.csv" $T/h52m_test.csv $W/lev108/h52m_test.csv
mkdir -p $W/m3
python3 -B team/member3/pkg/h50s/build_h50s_rows.py team/member1/trackB/build/inputs/pred_t004_id1.csv team/member3/pkg/h50s/inputs/incl_values.csv $W/m3/s9m_part.csv $W/m3/s24m_part.csv "$PRED_T018"
$C "$O/pred_t176_id1.csv" "$O/pred_t175_id1.csv" $T/s9m_test.csv $W/m3/s9m_part.csv
mkdir -p $W/scale
python3 -B $L/scale/tip17u.py $W/scale/tip17u.txt
python3 -B $L/scale/build_scale.py $W/scale/tip17u.txt $T/scale_test.csv
mkdir -p $W/kin
for v in captured_0009s captured_0009x; do python3 -B $L/kin/orange_track.py $v > $W/kin/otrk_$v.csv; done
python3 -B $L/kin/build_kin.py $T/kin_test.csv
mkdir -p $W/labtime
python3 -B $L/labtime/otrack.py captured_0009 $W/labtime/trk_captured_0009.csv
python3 -B $L/labtime/build_labtime.py $T/labtime_test.csv
$C "$O/pred_t177_id1.csv" "$O/pred_t176_id1.csv" $T/scale_test.csv $T/kin_test.csv $T/labtime_test.csv
python3 -B team/member1/trackB/package/hline/aud_parse/apply.py team/member1/trackB/package/hline/aud_parse/inputs/pred_t085_id1.csv output/h1/a/b/h1_rows.csv $W/h62m_aud_parse.csv
$C "$O/pred_t178_id1.csv" "$O/pred_t177_id1.csv" $T/h62m_test.csv $W/lev108/h62m_test.csv $W/h62m_aud_parse.csv
python3 -B $L/line2/build_line2.py "$O/pred_t178_id1.csv" $T/line2_test.csv
$C "$O/pred_t179_id1.csv" "$O/pred_t178_id1.csv" $T/line2_test.csv
bash $L/revisit/run.sh "$O/pred_t178_id1.csv" $T/revisit_test.csv
$C "$O/pred_t180_id1.csv" "$O/pred_t179_id1.csv" $T/revisit_test.csv
python3 -B $L/line3/build_line3.py "$O/pred_t179_id1.csv" $T/line3_test.csv
mkdir -p $W/own
python3 -B team/member2/trackB/package/own/build_own.py $T $W/own "$PRED_T018"
$C "$O/pred_t181_id1.csv" "$O/pred_t180_id1.csv" $T/line3_test.csv $T/s24m_test.csv $T/s24m_own_test.csv $W/m3/s24m_part.csv
mkdir -p $W/labown
for v in captured_0009s captured_0009x; do python3 -B $L/labtime/otrack.py $v $W/labown/trk_$v.csv; done
python3 -B $L/labown/build_labown.py $T/labown_test.csv
$C "$O/pred_t182_id1.csv" "$O/pred_t181_id1.csv" $T/labown_test.csv
$C "$O/pred_t183_id1.csv" "$O/pred_t182_id1.csv" $T/rj16_test.csv
python3 -B $L/p2prep/build_p2kincD2.py "$O/pred_t183_id1.csv" $T/p2kincD2_test.csv
python3 -B $L/p2prep/floor_check.py "$O/pred_t183_id1.csv" $T/p2new_test.csv
python3 -B $L/shrink_nolab/build_shrink_nolab.py submissions/pred_t002_detail.csv $T/shrink_nolab_test.csv
python3 -B $L/sbfix/build_sbfix.py "$O/pred_t183_id1.csv"
python3 -B $L/trk54/build_trk54.py $T
$C "$O/pred_t184_id1.csv" "$O/pred_t183_id1.csv" $T/h63m_test.csv $T/p2kincD2_test.csv $T/p2new_test.csv $T/shrink_nolab_test.csv $T/sbfix_test.csv $T/trk54_test.csv $T/trk54g_test.csv
python3 -B team/member3/pkg/r18/ceil_rule.py $W/m3/tm19_part.csv
python3 -B $L/tm19/rocket_orig.py $T/tm19x_test.csv
python3 -B $L/kincons2/build_kc2.py
$C "$O/pred_t185_id1.csv" "$O/pred_t184_id1.csv" $T/trk54p_test.csv $T/tm19_test.csv $W/m3/tm19_part.csv $W/lev108/tm19_test.csv $T/tm19x_test.csv $T/kincons2_test.csv
mkdir -p $W/m3/r18
python3 -B team/member3/pkg/r18/build_r18_rows.py $W/m3/r18
$C "$O/pred_t186_id1.csv" "$O/pred_t185_id1.csv" $T/h74S_test.csv $T/h74S_own_test.csv $T/h74D2_test.csv $T/h74D2_own_test.csv $W/m3/r18/h74D2_test.csv $W/lev108/h74D2_test.csv $T/h74D3a_test.csv $T/h74D3a_own_test.csv $W/lev108/h74D3a_test.csv $T/h74D3b_test.csv $W/lev108/h74D3b_test.csv
$C "$O/pred_t187_id1.csv" "$O/pred_t186_id1.csv" $T/rwalkS3_test.csv $T/h78D2_test.csv $W/m3/r18/h78D2_test.csv $W/lev108/h78D2_test.csv $T/h78D3_test.csv $T/h78D3_own_test.csv $W/m3/r18/h78S2_test.csv
python3 -B $L/kincons3/build_kc3.py
python3 -B $L/sizecons/build_sizecons.py
$C "$O/pred_t188_id1.csv" "$O/pred_t187_id1.csv" $T/rwalkD_test.csv $T/kincons3_test.csv $T/sizecons_test.csv
$C "$O/pred_t189_id1.csv" "$O/pred_t188_id1.csv" $T/h78S3_test.csv $T/h78S3_own_test.csv $T/w40c14_test.csv $T/w40c14_own_test.csv
python3 -B $L/zdepth/build_zdepth.py
$C "$O/pred_t190_id1.csv" "$O/pred_t189_id1.csv" $T/zdepth_test.csv
python3 -B $L/r02/build.py "$O/pred_t190_id1.csv" "$O/pred_t186_id1.csv" $T/r02_test.csv
bash $L/v15r03/run.sh "$O/pred_t190_id1.csv" $T/v15r03_r2_test.csv
python3 -B $L/kimi2/kimi_values.py $L/kimi2/in/rows.txt $L/kimi2/in/raw2 $T/kimi2_test.csv
$C "$O/pred_t191_id1.csv" "$O/pred_t190_id1.csv" $T/sweepD3_test.csv $T/sweepS_r2_test.csv $T/r02_test.csv $T/v01v17_test.csv $T/v15r03_r2_test.csv $T/kimi2_test.csv
python3 -B $L/kimi2/kimi_values.py $L/kimi3/in/rows.txt $L/kimi3/in/raw2 $T/kimi3b_test.csv
python3 -B team/member3/pkg/flyer/flyer_len.py "$O/pred_t190_id1.csv" $T/sbflyer_test.csv
$C "$O/pred_t192_id1.csv" "$O/pred_t191_id1.csv" $T/s3rev2_test.csv $T/sw2S3b_test.csv $T/sw3S3_test.csv $T/kimi3b_test.csv $T/kvrev_test.csv $T/sbflyer_test.csv $T/sw2S2_test.csv $T/sw2D2Sb_test.csv $T/sw2D2I_test.csv $T/sw3D2S_test.csv $T/sw3D2I_test.csv $T/sw2D3Lb_test.csv $T/sw2D3S_test.csv
$C "$O/pred_t001_id1.csv" "$O/pred_t192_id1.csv" $T/sw4D2S_test.csv $T/sw4D2I_test.csv $T/sw4D3b_test.csv $T/sw5D2S_test.csv $T/sw5D2I_test.csv $T/sw5D3b_test.csv
