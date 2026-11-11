import sys, pickle, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"
sys.path.insert(0, f"{R}/analysis/v3/ag/rul307"); import remeasure as RM
T = pd.read_pickle(f"{R}/analysis/v3/ag/rul346/tab.pkl")
yv = pd.read_csv(f"{R}/data/val/validation_dataset.csv").ground_truth_posterior
for split, ids in (("val", range(138, 147)), ("test", list(range(2563, 2572)) + list(range(2590, 2597)) + list(range(2611, 2620)) + list(range(2638, 2647)))):
    rr = [pickle.load(open(f"{R}/cache/tmp/ag_labmo_traj_{split}_{v}_prodP_q3.pkl", "rb")) for v in "vm"]
    for i in ids:
        for vn, d in zip("vm", rr):
            if i not in d: continue
            rec = d[i]; c = (rec.get("calls") or [None])[-1]
            if c is None or c.get("fn") != "pxq": print(split, i, vn, "no pxq", (rec.get("res") or {}).get("mode")); continue
            tt = np.asarray(c["tt"]); uv = np.asarray(c["uv"]); P = np.asarray(c["P"])
            j0 = 0; j12 = int(np.argmin(np.abs(tt - (c["t"] if c["t"] is not None else 1.2))))
            pathpx = np.sum(np.hypot(*np.diff(uv, axis=0).T)); path3 = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
            v1 = RM.value(rec, 1.0); v13 = RM.value(rec, 1.3)
            row = T[(T.split == split) & (T.id0 == i)].iloc[0]
            y = yv[i] if split == "val" else np.nan
            print(f"{split} {i} {vn} {row.video} {row.q[12:60]:48s} key={c.get('key')} uv0=({uv[0,0]:.0f},{uv[0,1]:.0f}) uv_t=({uv[j12,0]:.0f},{uv[j12,1]:.0f}) Z0={P[0,2]:.3f} Zt={P[j12,2]:.3f} Zend={P[-1,2]:.3f} "
                  f"pathpx={pathpx:.0f} path3={path3*100:.1f}cm fW={rec['res'].get('f'):.3f} v1={v1*100 if v1 else np.nan:.1f} F={v13/v1 if v1 else np.nan:.3f} g={row.g:.1f} b21={row.b21:.1f} y={y}")
            break
