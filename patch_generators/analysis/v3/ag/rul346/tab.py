"""latgeo2: per-row table for lab motion rows: record focal f/W, mode, own lateral factor F, velocity direction at t."""
import sys, json, pickle, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"
sys.path.insert(0, f"{R}/analysis/v3/ag/rul307"); import remeasure as RM
W2 = f"{R}/analysis/v3/ag/rul346"
def recs(split): return [pickle.load(open(f"{R}/cache/tmp/ag_labmo_traj_{split}_{v}_prodP_q3.pkl", "rb")) for v in ("v", "m")]
def F1(rec):
    try:
        a = RM.value(rec, 1.0); b = RM.value(rec, 1.3)
        return float(b / a) if a and b and np.isfinite(a) and a > 0 else np.nan
    except Exception: return np.nan
def vdir(c, t):
    """velocity vector (shown-camera coords) by a linear fit of P over t +-0.25 s; returns (vx, vy, vz)"""
    tt = np.asarray(c["tt"], float); P = np.asarray(c["P"], float)
    if t is None: m = np.ones(len(tt), bool)
    else: m = np.abs(tt - t) <= 0.25
    if m.sum() < 3: return (np.nan,) * 3
    return tuple(np.polyfit(tt[m], P[m, k], 1)[0] for k in range(3))
rows = []
for split in ("test", "val"):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{split}.csv")
    rr = recs(split)
    for i in range(len(A)):
        if not str(A.video.iat[i]).startswith("captured"): continue
        rec = None; var = None
        for vname, d in zip("vm", rr):
            if i in d and F1(d[i]) == F1(d[i]):
                rec = d[i]; var = vname; break
        if rec is None:
            for vname, d in zip("vm", rr):
                if i in d: rec = d[i]; var = vname; break
        res = (rec or {}).get("res", {}) or {}
        c = ((rec or {}).get("calls") or [None])[-1]
        F = F1(rec) if rec else np.nan
        vx = vy = vz = np.nan; key = ""
        if c is not None and c.get("fn") == "pxq":
            t = c.get("t"); t = t if t is not None else (None if c.get("t1") is None else 0.5 * (c["t1"] + c["t2"]))
            vx, vy, vz = vdir(c, t); key = c.get("key", "")
        rows.append(dict(split=split, id0=i, video=A.video.iat[i], cat=A.cat.iat[i], kind=A.kind.iat[i],
                         scored=A.scored.iat[i] if "scored" in A else True, var=var, mode=res.get("mode"), fW=res.get("f"),
                         F=F, vx=vx, vy=vy, vz=vz, key=key, g=A.g.iat[i], ans=A.answer.iat[i], src=A.source.iat[i],
                         y=A.y.iat[i] if "y" in A else np.nan, q=A.question.iat[i][:70]))
T = pd.DataFrame(rows)
T["lat"] = np.hypot(T.vx, T.vy) / np.sqrt(T.vx**2 + T.vy**2 + T.vz**2)
b21 = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv").parsed_value.values
b18 = pd.read_csv(f"{R}/submissions/sub_xt_best18v_id1.csv").parsed_value.values
T["b21"] = [b21[i] if s == "test" else np.nan for s, i in zip(T.split, T.id0)]
T["b18"] = [b18[i] if s == "test" else np.nan for s, i in zip(T.split, T.id0)]
lg = json.load(open(f"{R}/cache/tmp/ag_r2_lab_latgeo_patch_test.json"))
T["inblock"] = [s == "test" and str(i) in lg for s, i in zip(T.split, T.id0)]
T.to_pickle(f"{W2}/tab.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
T["rec"] = T.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True)
g = T[T["mode"].notna()].groupby(["rec", "split"]).agg(n=("id0", "size"), nblk=("inblock", "sum"), cats=("cat", lambda s: ",".join(sorted(set(s)))),
     modes=("mode", lambda s: ",".join(sorted(set(map(str, s))))[:60]), fW_med=("fW", "median"), fW_min=("fW", "min"), fW_max=("fW", "max"),
     F_med=("F", "median"), lat_med=("lat", "median"))
g.to_csv(f"{W2}/tab_rec.csv")
print(g.round(3).to_string())
