import sys, json, pickle, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
W2 = f"{R}/analysis/v3/ag/rul346"
T = pd.read_pickle(f"{W2}/tab.pkl"); Tt = T[T.split == "test"].set_index("id0"); Tv = T[T.split == "val"].set_index("id0")
L = pd.read_pickle(f"{R}/analysis/v3/ag/r2_lab_main/lr/L.pkl"); print("L columns", list(L.columns)[:12], len(L))
ycol = [c for c in L.columns if c in ("y", "label", "value")][0]
A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv")
A["rec"] = A.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True)
A["qk"] = A.question.str.lower().str.replace(r"\s+", " ", regex=True)
MOT = ["speed", "avgspeed", "disp", "path", "dist2"]
# (1) copies median vs own geo on labelled lab motion rows (test rows with labels), geo alone and geo x F
lab = A[A.video.str.startswith("captured") & A.kind.isin(MOT) & A.index.isin(L.index)].copy()
lab["y"] = L.loc[lab.index, ycol].astype(float); lab["F"] = Tt.F.reindex(lab.index).values
lab["gF"] = lab.g * lab.F.fillna(1.0)
for col in ("g", "gF"):
    lab[col + "_med"] = lab.groupby(["rec", "qk"])[col].transform("median")
lab["n"] = lab.groupby(["rec", "qk"]).g.transform("size")
m = lab[(lab.n >= 2) & lab.g.notna() & (lab.y > 0)]
print(f"(1) labelled lab motion rows in copy groups: {len(m)} rows, {m.groupby(['rec','qk']).ngroups} groups, recs {sorted(m.rec.unique())}")
for col in ("g", "gF"):
    a = FC.row_mra(m[col].values, m.y.values); b = FC.row_mra(m[col + "_med"].values, m.y.values)
    print(f"    {col}: own {a.mean():.3f} -> copies median {b.mean():.3f} (sum {b.sum()-a.sum():+.2f}); rows changed >3%: {(np.abs(m[col+'_med']/m[col]-1)>0.03).sum()}")
# (2) extension candidates: S3 lab motion rows, scored, not label-tagged, current value from a geo-led patch, F >= 1.1
b21 = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv").parsed_value.values
blk = {int(k) for k in json.load(open(f"{R}/cache/tmp/ag_r2_lab_latgeo_patch_test.json"))}
ext = {}
for tg, rule_geo in (("c80dupkeyfS2", "per-row meas"), ("echoS2", "own geo"), ("dkeyownS2", "per-row meas")):
    js = json.load(open(f"{R}/cache/tmp/ag_{tg}_patch_test.json"))
    for k, v in js.items():
        i = int(k)
        if i in blk or i in L.index or not A.scored.iat[i] or A.cat.iat[i] != "S3" or not A.video.iat[i].startswith("captured") or A.kind.iat[i] not in MOT: continue
        if not np.isclose(b21[i], v["value"], rtol=0.03): continue    # current answer must be this rule's value
        why = v.get("why", ""); import re
        mm = re.search(r"geo [\d.]+ -> ([\d.]+)", why) or re.search(r"own geo ([\d.]+)", why)
        geo = float(mm.group(1)) if mm else np.nan
        F = Tt.F.get(i, np.nan)
        if np.isfinite(geo) and np.isfinite(F) and F >= 1.1: ext[i] = (tg, geo, F)
print("(2) extension rows (current = rule value within 3 %, F >= 1.1):", len(ext))
E = pd.DataFrame([dict(id0=i, video=A.video.iat[i], kind=A.kind.iat[i], tag=tg, b21=b21[i], geo=geo, F=F, new=geo * F, q=A.question.iat[i][:55]) for i, (tg, geo, F) in sorted(ext.items())])
if len(E):
    print(E.round(3).to_string())
    r = FC.row_mra(E.new.values, E.new.values) - FC.row_mra(E.b21.values, E.new.values); w = FC.row_mra(E.new.values, E.b21.values) - FC.row_mra(E.b21.values, E.b21.values)
    print(f"    bound if right {r.sum():+.1f}, if wrong {w.sum():+.1f} (S3 row-MRA)")
    rng = np.random.default_rng(1)
    for sg in (0.25, 0.3):
        gr = np.mean([FC.row_mra(E.new.values, y).sum() - FC.row_mra(E.b21.values, y).sum() for y in E.new.values * np.exp(sg * rng.standard_normal((2000, len(E))))])
        gw = np.mean([FC.row_mra(E.new.values, y).sum() - FC.row_mra(E.b21.values, y).sum() for y in E.b21.values * np.exp(sg * rng.standard_normal((2000, len(E))))])
        print(f"    sigma {sg}: expected if right {gr:+.2f}, if wrong {gw:+.2f}")
    E.to_csv(f"{W2}/ext_rows.csv", index=False)
# (3) D3 lab motion rows: focal source and F; labelled evidence by focal source
D = T[(T.cat == "D3") & T.video.str.startswith("captured") & T.kind.isin(MOT) & T["mode"].notna()].copy()
D["src_f"] = np.where(D["mode"].astype(str).str.contains("fovdefault|knownsize"), "spec/knownsize", "calib(motion prior)")
D["lab"] = [(s == "val") or (i in L.index) for s, i in zip(D.split, D.id0)]
D["y2"] = [D.y.iat[k] if D.split.iat[k] == "val" else (float(L.loc[D.id0.iat[k], ycol]) if D.id0.iat[k] in L.index else np.nan) for k in range(len(D))]
print("(3) D3 lab motion rows by focal source (all splits):"); print(D.groupby(["split", "src_f"]).agg(n=("id0", "size"), nlab=("lab", "sum"), F=("F", "median")).to_string())
for s in ("spec/knownsize", "calib(motion prior)"):
    q = D[(D.src_f == s) & D.y2.notna() & D.g.notna() & D.F.notna() & (D.y2 > 0)]
    if len(q): print(f"    labelled {s}: n {len(q)} ({','.join(sorted(q.video.str[:13].unique()))}); g {FC.row_mra(q.g.values, q.y2.values).mean():.3f} -> g x F {FC.row_mra((q.g*q.F).values, q.y2.values).mean():.3f}; median y/g {np.median(q.y2/q.g):.3f}, median F {q.F.median():.3f}")
t3 = D[(D.split == "test") & (D.src_f == "spec/knownsize") & ~D.lab]
print(f"    unlabelled test D3 spec/knownsize rows: {len(t3)}, scored {int(A.scored.reindex(t3.id0).sum())}, F>=1.1: {(t3.F>=1.1).sum()}, videos {sorted(t3.video.str[:13].unique())}")
