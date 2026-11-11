
X30=(--caw caw9b_orig_crop --rul263 caw --cawconf 0.1 --cawconf_rules d3 --geo_kinds dist2,avgspeed --var_caw _segmented,x
     --sib_caw test_sib --dis_motion speed,3d,nofov --event_geo weak --agree_w size:0.65:strong --kin sim:grav:before:vT
     --dis_size_gm all --caw_shrink caw_nogeo,dis_caw_routed,caw_c1:size:0.9 --cqc --phys_bounds extra3 --xrow_prior
     --c2_anchor "0.3,2,mix,2" --var_geo_plain 1.4:strong:video --var_ans_plain _segmented --lab_pool event:3 --scene_prior 3
     --d3lab_focal auto:1.3:gm --rul263_full --caw_override test_tshift30
     --rul263_scale simulation:1.1:-height,simulation:1.05:+height --caw_prescale simulation:D2:size:1.12
     --rul263_dimgm "simulation:width:0:2.6;simulation:length:0:1.6"
     --probe "S2:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;S3:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;D2:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D3:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D2:agree_wgm:size:w0.7"
     --phys_bounds_final --sruler_geo any --depth_copy --probe_kin_exempt)
X30_PL=()
_skip=0
for _a in "${X30[@]}"; do
  if [ $_skip = 1 ]; then _skip=0; continue; fi
  case "$_a" in --xrow_prior) continue ;; --scene_prior) _skip=1; continue ;; esac
  X30_PL+=("$_a")
done
unset _a _skip
FLB=(--scene_prior_ext dimpre --var_norm --disp_bound 4 --std_court --geo_only_modes 2d_gself --depth_bound)

P_NU="dim,lane,kin,hgt,rul264,jump"
P_NZ="$P_NU,second,rul001"
P_NZA="$P_NZ,rul439,wing"
P_NZB="$P_NZA,twin,roll,rul280,rul265,rul285,rul298,rul413,rul006,wlen,rul267,rul412"
P_NZC="$P_NZB,rul303,rul005,rul428"
P_NZD="$P_NZC,rul424"
P_NZE="$P_NZD,orbit,rul305,rul282,disc,rul269"
P_NZF="$P_NZE,rul323,rul432,rul324"
P_NZG="$P_NZF,p1pf,rul436,rul438,rul301,rul297"
P_NZH="$P_NZG,rul284,rul270,rul454,rul423,rul268,rul313,rul314,rul329"
P_NZI="$P_NZH,rul417,rul300,rul433,rul327,rul219,gap"
P_NZJ="$P_NZI,rul333,rul406,rul421"
P_NZK="$P_NZJ,rul319,rul321,rul318,rul322,rul320,rul315"
P_NZL="$P_NZK,rul420,rul291,rul425,rul435"
P_NZM="$P_NZL,rul426,rul334,rul304,rul328"
P_NZN="$P_NZM,rul332,rul419,rul289,rul271,rul286,rul330"
P_NZO="${P_NZN/,rul426,/,rul018,rul426,},rul013,rul012,rul015,rul011,rul014,rul016,rul020,rul019,rul017"
P_NZP="${P_NZO/,rul421,/,rul421,rul422,},rul021,rul025,rul026,rul022,rul024,rul029"
P_NZP="$P_NZP,rul028,rul030,rul031,rul027,rul034,rul032,rul033"
P_NZQ="$P_NZP,rul037,rul036,rul038,rul035,rul042,rul040,rul041,rul039"
P_NZR="$P_NZQ,rul043"
P_NZS="$P_NZR,rul044"
P_NZT="$P_NZS,rul045"
P_NZU="$P_NZT,rul047,rul048"
P_NZV="$P_NZU,rul046"
P_NZW="$P_NZV,rul049"
P_NZX="$P_NZW,rul057,rul054,rul055,rul050,rul053,rul052,rul051"
P_NZY="$P_NZX,rul058,rul061,rul059,rul060"
P_NZZ="$P_NZY,rul064,rul063,rul062"
P_PRED_T105="$P_NZZ,rul067,rul068,rul066,rul065"
P_PRED_T109="$P_PRED_T105,rul072,rul071,rul070,rul069,rul074,rul076,rul073,rul075"
P_PRED_T112="$P_PRED_T109,rul079,rul081,rul080"
P_PRED_T147="$P_PRED_T112,rul078,rul077,rul082,rul083,rul084"
P_PRED_T148="$P_PRED_T147,rul088,rul087,rul086"
P_PRED_T149="$P_PRED_T148,rul089,rul090,rul091,rul092"
P_PRED_T150="$P_PRED_T149,rul093,rul094"
P_PRED_T151="$P_PRED_T150,rul095,rul096,rul097"
P_PRED_T095="$P_PRED_T151,rul098,rul099,rul100"
P_PRED_T096="$P_PRED_T095,rul101,rul103,rul102"
P_PRED_T097="$P_PRED_T096,rul085,rul106,rul105,rul104"
P_PRED_T098="$P_PRED_T097,rul108,rul107"
_M14="rul109,rul110"
_M15="$_M14,rul111,rul112"
_M16="$_M15,rul113,rul115,rul114,rul118,rul117,rul116"
_M17="$_M16,rul120,rul119,rul121"
_M18="$_M17,rul122,rul124,rul123"
_M19="$_M18,rul125,rul126,rul127"
_M20="$_M19,rul130,rul131,rul129,rul128"
P_PRED_T099="$P_PRED_T098,$_M14,c63plainkey"
P_PRED_T100="$P_PRED_T098,$_M15,c63plainkey"
P_PRED_T101="$P_PRED_T098,$_M16,c63plainkey"
P_PRED_T102="$P_PRED_T098,$_M17,c63plainkey"
P_PRED_T103="$P_PRED_T098,$_M18,c63plainkey"
P_PRED_T104="$P_PRED_T098,$_M19,c63plainkey"
P_PRED_T106="$P_PRED_T098,$_M20,c63plainkey"
unset _M14 _M15 _M16 _M17 _M18 _M19 _M20
_H48="${P_PRED_T098/,rul439,/,rul441,}"
_H48="${_H48/,rul280,/,rul281,}"
_T48="rul132,rul134,rul133,rul137,rul135,rul136,rul138,rul139,rul141"
_T48="$_T48,rul140,rul142,rul143,rul144,rul145,rul147,rul148,rul151,rul149,rul150"
_T48="$_T48,rul146,rul155,rul154,rul160,rul157,rul158,rul159,rul165,rul164"
_T48="$_T48,rul163,rul161,rul162,rul186,rul191,rul193,rul190,rul188"
_T48="$_T48,rul189,rul187,rul153,rul152,rul156,rul173,rul174,rul177"
_T48="$_T48,rul176,rul175,rul168,rul170,rul172,rul171,rul169,rul166"
_T48="$_T48,rul167,rul181,rul182,rul179,rul185,rul178,rul184"
_T48="$_T48,rul180,rul194,rul195,rul196,rul200,rul198,rul199,rul204"
_T48="$_T48,rul205,rul202,rul203,rul183,rul197,rul201"
_T48="$_T48,rul209,rul208,rul207,rul211,rul210,rul212"
_T48="$_T48,rul213,rul214,rul215,rul222,rul221,rul224,rul227"
_T48="$_T48,rul225,rul223,rul228,rul232,rul233,rul235"
_T48="$_T48,rul236,rul237,rul239,rul238,rul240"
_T48="$_T48,rul246,rul241,rul242,rul245,rul244"
_T48="$_T48,rul243,rul247,rul248,rul249,rul250"
_T48="$_T48,rul252,rul251,rul257,rul256,rul254"
_T48="$_T48,rul255,rul253,rul229,rul231"
_T48="$_T48,rul230,rul234,rul192,rul338,rul339,rul275"
_T48="$_T48,rul274,rul276,rul278,rul277,rul226"
P_PRED_T002="$_H48,rul109,rul110,rul111,rul112,rul113,rul115,rul114,rul118,rul117"
P_PRED_T002="$P_PRED_T002,rul116,rul120,rul119,rul121,rul122,rul124,rul123,rul125,rul126"
P_PRED_T002="$P_PRED_T002,rul127,rul130,rul131,rul129,rul128,$_T48"
unset _H48 _T48
X48A=(--probe "S2:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;S3:caw_nogeo,dis_caw_routed,caw_c1:size:x1.1111;D2:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D3:caw_nogeo,dis_caw_routed,caw_c1:any:x0.9;D2:agree_wgm:size:w0.7;S2:agree_wgm,agree_gm:size:geo:simulation;S2:agree_wgm,agree_gm:size:geo:internet;D2:agree_wgm,agree_gm:size:geo:simulation;S3:agree_wgm,agree_gm:size:geo:simulation;D3:agree_wgm,agree_gm:size:geo:simulation"
      --rul263_scale simulation:1.1:-height,simulation:1.05:+height)

submit() {
  local g=$1 o=$2; shift 2
  QP_SANE=h1only $PY analysis/v3/submit_v3.py "$g" "$o" "${X30[@]}" "${FLB[@]}" "$@"
}
submit_pl() {
  local g=$1 o=$2; shift 2
  QP_SANE=h1only $PY analysis/v3/submit_v3.py "$g" "$o" "${X30_PL[@]}" "${FLB[@]}" "$@"
}

fusion_pair() {
  local nm=$1 t=$2 pl=${3:-}
  if [ $# -ge 3 ]; then shift 3; else shift $#; fi
  local px=(); [ -n "$pl" ] && px=(--patch_json "$pl")
  logged ${nm}_x.log submit geo_v4nt${t}srngsr_d_vw ${nm}_x ${px[@]+"${px[@]}"} "$@"
  logged ${nm}_xu.log submit geo_v4nt${t}usrngsr_d_vw ${nm}_xu ${px[@]+"${px[@]}"} "$@"
  logged mg_$nm.log $PY tools/mo_guard.py $nm ${nm}_x ${nm}_xu
}
