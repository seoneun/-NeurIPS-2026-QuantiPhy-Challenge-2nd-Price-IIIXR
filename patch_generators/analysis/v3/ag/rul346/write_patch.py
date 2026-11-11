"""latgeo2 patches (2026-10-20). Inputs: ext_rows.csv (ext.py: S3 lab motion rows, scored, unlabelled, not in rul345, whose
best21v answer is the value of a geo-led label-free patch (c80dupkeyfS2 / echoS2 / dkeyownS2) within 3 %, F >= 1.1; gated here to F >= 1.2) and blk.pkl
(decomp.py: rul345 rows with the median of the plain / s / x copies per question).
  ext : value = rule geo x F (F = remeasure.value(rec,1.3)/value(rec,1), strict-line record, as in rul345).
  cmed: captured_0033 row 2554 (plain copy; g 350 vs s / x copies 253 / 252) -> median of the three copies' block values.
Usage: python write_patch.py [write]"""
import sys, json, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; W2 = f"{R}/analysis/v3/ag/rul346"
E = pd.read_csv(f"{W2}/ext_rows.csv"); E = E[E.F >= 1.2]; B = pd.read_pickle(f"{W2}/blk.pkl")  # F >= 1.2: drops 0031 (F 1.15, EV < 0)
ext = {str(int(r.id0)): {"value": float(r.new), "why": f"r2_latgeo2_ext: lateral factor on the geo-led value of {r.tag}: geo {r.geo:.4g} x F {r.F:.4f}; was {r.b21:.4g}"} for r in E.itertuples()}
# s / x copies of an extension row (same recording, same question text, same best21v value, scored, not labelled, not in the block)
# get the plain row's value: same frames, one truth
A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv"); b21 = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv").parsed_value.values
L = pd.read_pickle(f"{R}/analysis/v3/ag/r2_lab_main/lr/L.pkl"); blk = {int(k) for k in json.load(open(f"{R}/cache/tmp/ag_r2_lab_latgeo_patch_test.json"))}
A["rec"] = A.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True); A["qk"] = A.question.str.lower().str.replace(r"\s+", " ", regex=True)
for k in list(ext):
    i = int(k); m = A[(A.rec == A.rec.iat[i]) & (A.qk == A.qk.iat[i]) & A.scored & ~A.index.isin(L.index) & ~A.index.isin(blk)]
    for j in m.index:
        if str(j) not in ext and np.isclose(b21[j], b21[i], rtol=0.005):
            ext[str(j)] = {"value": ext[k]["value"], "why": f"r2_latgeo2_ext: copy of row {i} (same frames): " + ext[k]["why"].split(": ", 1)[1]}
cm = {str(i): {"value": float(B.grp_med[i]), "why": f"rul347: median of the plain/s/x copies' rul345 values ({B.grp_med[i]:.4g}); own {B.new[i]:.4g} (g {B.g[i]:.4g} vs copies)"} for i in [2554]}
print("ext rows", len(ext), " cmed rows", len(cm))
if "write" in sys.argv[1:]:
    for nm, js in (("ext", ext), ("cmed", cm)):
        json.dump(js, open(f"{R}/cache/tmp/ag_r2_latgeo2_{nm}_patch_test.json", "w"), indent=1)
    S = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv"); print("columns", list(S.columns))
    for js in (ext, cm):
        for k, v in js.items(): S.loc[int(k), "parsed_value"] = v["value"]
    S.to_csv(f"{R}/cache/tmp/ag_r2_latgeo2_b21_id1.csv", index=False); print("wrote cache/tmp/ag_r2_latgeo2_b21_id1.csv")
