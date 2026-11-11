import sys, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
import analyze_caw as A
from lab_rows import labelled
pt = {int(k): v["value"] for k, v in json.load(open(f"{R}/cache/tmp/ag_c77ours_patch_test.json")).items()}
pv = {int(k): v["value"] for k, v in json.load(open(f"{R}/cache/tmp/ag_c77ours_patch_val.json")).items()}
V = pd.read_csv(f"{R}/submissions/sub_xt_Nzz23_valdetail.csv").set_index("Unnamed: 0")
new = V.answer.copy()
for i, v in pv.items(): new[i] = v
m0 = pd.Series(A.row_mra(V.answer.values, V.y.values), V.index); m1 = pd.Series(A.row_mra(new.values, V.y.values), V.index)
print("VAL rows changed", len(pv))
for i in pv: print(f"  val {i} {V.video_id[i]} {V.cat[i]} y={V.y[i]:.4g} old={V.answer[i]:.4g} ({V.answer_source[i]}) new={pv[i]:.4g}  mra {m0[i]:.2f}->{m1[i]:.2f}")
print("VAL macro %.4f -> %.4f" % (A.macro(m0, V.cat)[0], A.macro(m1, V.cat)[0]))
# friend's full routing on val (all scene3d/kin2d values routed by caw_nogeo/d3size_), for reference
s3 = pd.read_csv(f"{R}/analysis/v3/ag/rul143/wt/geo/out/scene3d_val_detS1.csv").set_index("id")
k2 = pd.read_csv(f"{R}/analysis/v3/ag/rul143/wt/geo/out/kin2d_val_v5dS1.csv").set_index("id")
fr = {}
for i, r in s3.iterrows():
    if r.get("vt") == "3D" and pd.notna(r.value) and (V.answer_source[i] == "caw_nogeo" or str(V.answer_source[i]).startswith("d3size_")) and 0.1 <= r.value / V.answer[i] <= 10: fr[i] = r.value
for i, r in k2.iterrows():
    if pd.notna(r.value) and V.answer_source[i] == "caw_nogeo" and 0.1 <= r.value / V.answer[i] <= 10: fr[i] = r.value
nf = V.answer.copy()
for i, v in fr.items(): nf[i] = v
mf = pd.Series(A.row_mra(nf.values, V.y.values), V.index)
print("VAL teammate routing on Nzz23 (tile check omitted):", len(fr), "rows; macro %.4f" % A.macro(mf, V.cat)[0])
for i in fr: print(f"  val {i} {V.video_id[i]} {V.cat[i]} y={V.y[i]:.4g} old={V.answer[i]:.4g} ({V.answer_source[i]}) mod={fr[i]:.4g} mra {m0[i]:.2f}->{mf[i]:.2f}")
# test: write eval submission and labelled copies
S = pd.read_csv(f"{R}/submissions/sub_xt_Nzz23.csv")
o = S.copy(); idx = o["Unnamed: 0"]
for i, v in pt.items(): o.loc[idx == i, "parsed_value"] = v
o.to_csv(f"{R}/submissions/sub_ag_c77ours_eval.csv", index=False)
L = labelled("sub_xt_Nzz23"); LT = L[L.split == "test"]
F = pd.read_csv(f"{R}/submissions/sub_friend_ours2.csv").set_index("Unnamed: 0").parsed_value
for i in sorted(set(pt) | set(range(0))):
    if i in LT.index:
        y = LT.y[i]; print(f"  labelled test copy {i}: y={y:.4g} Nzz23={LT.answer[i]:.4g} m={A.row_mra(np.array([LT.answer[i]]), np.array([y]))[0]:.2f} c77={pt[i]:.4g} m={A.row_mra(np.array([pt[i]]), np.array([y]))[0]:.2f}")
print("test cats changed:", pd.read_csv(f"{R}/submissions/sub_xt_Nzz23_detail.csv", index_col=0).cat.reindex(list(pt)).value_counts().to_dict())
