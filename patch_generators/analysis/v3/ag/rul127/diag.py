"""rul127 diag: tedge's uncut-run cross-check (rul433/diag_own.py run_speed) recomputed from BOTH the SAM2 mask centroid
tr.uv ('c', what tedge uses) and the box centre ((x1+x2)/2, (y1+y2)/2) over the SAME uncut run frames and fit window.
Rows: every row of cache/tmp/ag_tedge3_diagown_{split}.csv whose target is border-cut in the query-time fit frames (ncut > 0).
The run used is the one tedge's gates use: run (no persistent side) or run2 (frames free of the transient sides when a side is
touched in >= 90% of the track). writes cache/tmp/ag_c69segrun_diag_{split}.csv
usage: python analysis/v3/ag/rul127/diag.py <split>"""
import os, sys
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SPLIT = sys.argv[1]
sys.path.insert(0, R + "/analysis/v3/ag/rul433")
import diag_own as DO          # sets the same env defaults and MV config as the rul433 own-track diag (module guard on __main__)
MV = DO.MV


def run_fit(tr, cut, a, b, P):
    """DO.run_speed with the trajectory P as an argument (same run choice, same fit window); returns (vx, vy, nfit, i, j)"""
    t = tr.t
    best = None
    for i, j in DO.runs(cut, t):
        if j - i < DO.RUN_N or t[j - 1] - t[i] < DO.RUN_DUR - 1e-9:
            continue
        if t[j - 1] < a:
            gap, side = a - t[j - 1], "before"
        elif t[i] > b:
            gap, side = t[i] - b, "after"
        else:
            gap, side = 0.0, "inside"
        if best is None or gap < best[0]:
            best = (gap, side, i, j)
    if best is None:
        return None
    gap, side, i, j = best
    tt, PP = t[i:j], P[i:j]
    if side == "before":
        m = tt >= tt[-1] - DO.RUN_WIN
    elif side == "after":
        m = tt <= tt[0] + DO.RUN_WIN
    else:
        c = 0.5 * (a + b); m = np.abs(tt - c) <= DO.RUN_WIN
    if m.sum() < 3:
        m = np.ones(len(tt), bool)
    v = np.array([np.polyfit(tt[m], PP[m, k], 1)[0] for k in range(2)])
    return float(v[0]), float(v[1]), int(m.sum()), int(j - i)


if __name__ == "__main__":
    D0 = pd.read_csv(f"{R}/cache/tmp/ag_tedge3_diagown_{SPLIT}.csv", index_col=0)
    want = set(D0.index[D0.ncut.fillna(0) > 0])
    MV.TRACK_VER = "v2sc"
    byv, vjs = MV.load_split_tracks(SPLIT, "e")
    rows = []
    for v, recs in byv.items():
        rr_ = [r for r in recs if r["id"] in want]
        if not rr_ or v not in vjs:
            continue
        MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
        vj = vjs[v]; V = MV.Video(vj, "e")
        W, H = float(vj["W"]), float(vj["H"])
        for r in rr_:
            q = r["q"]
            tr = V.get(r["target"])
            d = dict(id=r["id"], vid=v)
            if tr is None or tr.n < 3:
                d["note"] = "no_target_track"; rows.append(d); continue
            w, win = DO.used_frames(tr, q["kind"], q)
            if w is None or len(w) == 0:
                d["note"] = "no_window"; rows.append(d); continue
            S = DO.sides(tr, W, H)
            pers = S.mean(0) >= 0.9
            cut = S[:, ~pers].any(1) if pers.any() else S.any(1)
            d["run_used"] = "run2" if pers.any() else "run"
            box = np.array([tr.st[j]["box"] for j in tr.okc], float)
            Pc = np.array(tr.uv, float)
            Pb = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
            fc = run_fit(tr, cut, *win, Pc); fb = run_fit(tr, cut, *win, Pb)
            if fc is None or fb is None:
                d["note"] = "no_run"; rows.append(d); continue
            d.update(c_vx=fc[0], c_vy=fc[1], b_vx=fb[0], b_vy=fb[1], run_nfit=fc[2], run_nrun=fc[3])
            k = 0 if abs(fc[0]) >= abs(fc[1]) else 1            # motion axis = the larger centroid slope component
            d["ax"] = "xy"[k]
            d["c_ax"] = [fc[0], fc[1]][k]; d["b_ax"] = [fb[0], fb[1]][k]
            vc = np.array([fc[0], fc[1]]); vb = vc.copy(); vb[k] = [fb[0], fb[1]][k]
            d["c_px"] = float(np.linalg.norm(vc)); d["b_px"] = float(np.linalg.norm(vb))   # box centre on the motion axis only
            d["slope_diff"] = abs(d["b_ax"]) / max(abs(d["c_ax"]), 1e-9) - 1
            rows.append(d)
    D = pd.DataFrame(rows).set_index("id")
    D.to_csv(f"{R}/cache/tmp/ag_c69segrun_diag_{SPLIT}.csv")
    print(SPLIT, "rows", len(D), "notes", D["note"].value_counts().to_dict() if "note" in D else {})
