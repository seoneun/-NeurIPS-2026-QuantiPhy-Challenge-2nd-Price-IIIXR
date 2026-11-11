"""latgeo2: 0034 / 0035 re-measurement with the chain estimator (measure_v3.px_quantity) on the row's record, with the depth
held constant at the row's depth value at the question time (Zt), and lateral X, Y at a given f/W.  Prints values at spec focal."""
import sys, pickle, copy, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"
sys.path.insert(0, f"{R}/analysis/v3/ag/rul307"); import remeasure as RM
MV = RM.MV
yv = pd.read_csv(f"{R}/data/val/validation_dataset.csv").ground_truth_posterior
SPEC = 0.5959
def flat(rec, fW_new=None, zconst=True):
    c = rec["calls"][-1]; P = np.array(c["P"], float); fW = rec["res"]["f"]
    tt = np.asarray(c["tt"]); t = c["t"] if c["t"] is not None else (c["t1"] if c["t1"] is not None else 1.2)
    j = int(np.argmin(np.abs(tt - t))); Zc = P[j, 2]
    Q = P.copy()
    if zconst: Q[:, :2] *= (Zc / P[:, 2])[:, None]; Q[:, 2] = Zc
    if fW_new: Q[:, :2] *= fW / fW_new
    tr = RM._T(c["tt"], c["uv"], c.get("sg"))
    return MV.px_quantity(tr, c["kind"], c["t"], c["t1"], c["t2"], c["when"], P=Q)
out = []
for split, ids in (("val", range(138, 147)), ("test", [2563, 2564, 2568, 2569, 2570, 2571, 2590, 2591, 2595, 2596])):
    rr = [pickle.load(open(f"{R}/cache/tmp/ag_labmo_traj_{split}_{v}_prodP_q3.pkl", "rb")) for v in "vm"]
    for i in ids:
        rec = next((d[i] for d in rr if i in d and (d[i].get("calls") or [{}])[-1].get("fn") == "pxq"), None)
        if rec is None: continue
        c = rec["calls"][-1]
        v_chain = RM.value(rec, 1.0); v_flat = flat(rec); v_flat_spec = flat(rec, SPEC); v_flat_spec13 = v_flat_spec * 1.3 if v_flat_spec else None
        out.append(dict(split=split, id=i, key=c["key"], kind=c["kind"], t=c["t"], fW=rec["res"]["f"], Zt=float(np.asarray(c["P"])[int(np.argmin(np.abs(np.asarray(c["tt"]) - (c["t"] or 1.2)))), 2]),
                        chain=v_chain * 100, flat=v_flat * 100, flat_spec=v_flat_spec * 100, y=yv[i] if split == "val" else np.nan))
D = pd.DataFrame(out)
yw = {("left", "speed", 1.2): 82.93, ("right", "speed", 1.2): 99.47, ("left", "speed", 1.3): 84.54, ("right", "speed", 1.3): 92.33, ("left", "path", None): 57.1, ("right", "path", None): 70.62}
D["side"] = D.key.str.split("|").str[1]
D["y_worded"] = [yw.get((s, k, t if k == "speed" else None), np.nan) for s, k, t in zip(D.side, D.kind, D.t)]
D["y_swapped"] = [yw.get(("right" if s == "left" else "left", k, t if k == "speed" else None), np.nan) for s, k, t in zip(D.side, D.kind, D.t)]
D["r_worded"] = D.y_worded / D.flat_spec; D["r_swapped"] = D.y_swapped / D.flat_spec
pd.set_option("display.width", 250); print(D.round(3).to_string())
D.to_csv(f"{R}/analysis/v3/ag/rul346/flat34.csv", index=False)
