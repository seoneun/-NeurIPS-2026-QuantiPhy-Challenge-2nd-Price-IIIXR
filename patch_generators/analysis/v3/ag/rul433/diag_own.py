"""rul433 diag: analysis/v3/ag/tedge/diag.py, unchanged except (1) TEDGE3_OWN=1 re-measures every row on the track file the
raw run itself loaded (MV.load_split_tracks with TRACK_VER=v2sc, which falls back to tracks_{split}/ (meas_tracks v1) when a video
has no v2sc file) instead of mapping meas_tracks v1 -> v3ntmv; (2) TEDGE3_VIDS restricts videos; (3) writes
cache/tmp/ag_tedge3_diag<TEDGE3_TAG>_{split}.csv. Generated from tedge/diag.py by rul433/make_diag.sh."""
"""tedge diagnostics: target speed when the target box is border-cut at the query time.
Rows: speed-at-time ("speed of X at t") and avgspeed windows ("average speed of X between t1 and t2").
measure_v3.px_quantity reads the target's speed from local quadratic fits of the SAM2 mask CENTROID (vel_at_c on the
piecewise-quadratic segment containing t0, half window 0.25 s widened to 0.4 / 0.625 s until it holds 5 frames;
avgspeed = mean |v| of 0.15 s fits at every frame in [t1, t2]). While the box touches the image border (within 2 px) the
centroid follows only the visible part: an object entering through the left border has its right (free) edge moving at
the true speed and its left edge pinned at x=0, so the centroid moves at about half the speed.
For each candidate row this script records:
  - old_px: px_quantity exactly as measure_v3 computes it (2D check: old_px * meas_mpp == meas_value)
  - the frames the fit at the query time uses (same windows / segment), and the border sides touched there
  - new_px: the same fit with the cut axis read from the FREE edge (x1 if cut on the left, x0 if cut on the right,
    y1 if cut on top, y0 if cut at the bottom) for the frames of the window; the uncut axis stays the centroid.
    An axis cut on both sides has no free edge -> no measurement.
  - run_px: the centroid speed on the nearest uncut run of the same track (>= 5 frames, >= 0.25 s; linear fit over the
    0.5 s of the run next to the query time) -> the physical check: the free-edge speed should match it.
writes cache/tmp/ag_tedge_diag_{split}.csv"""
import os, sys, json, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 40)
SPLIT = sys.argv[1]
TOL = 2.0
RUN_N, RUN_DUR, RUN_WIN = 5, 0.25, 0.5


def window_idx(t, P, t0, sg, half):
    """indices (into t) of the frames vel_at_c(t, P, t0, sg, half) fits on"""
    s = MV._sub(sg, t, P, t0)
    if s is None:
        i, j = 0, len(t)
    else:
        k = MV.seg_index(sg, t, t0); i, j = sg["segs"][k]
    ts = t[i:j]
    t0c = min(max(t0, ts.min()), ts.max())
    for deg, mp in ((2, 5), (1, 3)):
        for h in (half, half * 1.6, half * 2.5):
            m = np.abs(ts - t0c) <= h + 1e-9
            if m.sum() >= mp:
                return i + np.where(m)[0]
    return np.array([], int)


def used_frames(tr, kind, q):
    t, P, sg = tr.t, tr.uv, tr.segs()
    lo, hi = t.min(), t.max()
    if kind == "speed":
        t0 = MV.clamp_time(tr, q["t"])
        if t0 is None:
            return None, None
        return window_idx(t, P, t0, sg, MV.HALF_V), (t0, t0)
    a = lo if q.get("t1") is None else max(q["t1"], lo)
    b = hi if q.get("t2") is None else min(q["t2"], hi)
    m = np.where((t >= a - 1e-9) & (t <= b + 1e-9))[0]
    if len(m) < 3:
        return None, None
    ws = set()
    for i in m:
        ws.update(window_idx(t, P, t[i], sg, 0.15).tolist())
    return np.array(sorted(ws), int), (a, b)


def sides(tr, W, H):
    """per okc frame: (L, R, T, B) border contact within TOL px"""
    out = []
    for j in tr.okc:
        b = tr.st[j]["box"]
        out.append((b[0] <= TOL, b[2] >= W - 1 - TOL, b[1] <= TOL, b[3] >= H - 1 - TOL))
    return np.array(out, bool).reshape(-1, 4)


def free_edge_P(tr, S, w):
    """centroid trajectory with the cut axes of the window frames read from the free edge; None if an axis is cut on both sides"""
    Ls, Rs, Ts, Bs = S[w].any(0)
    P = np.array(tr.uv, float).copy()
    box = np.array([tr.st[j]["box"] for j in tr.okc], float)
    ax = []
    if Ls and Rs:
        return None, "x_both"
    if Ts and Bs:
        return None, "y_both"
    if Ls:
        P[w, 0] = box[w, 2]; ax.append("L>x1")
    elif Rs:
        P[w, 0] = box[w, 0]; ax.append("R>x0")
    if Ts:
        P[w, 1] = box[w, 3]; ax.append("T>y1")
    elif Bs:
        P[w, 1] = box[w, 1]; ax.append("B>y0")
    return P, "+".join(ax)


def runs(cut, t):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out, s = [], None
    for k in range(len(cut)):
        brk = k > 0 and (t[k] - t[k - 1]) > 3 * dt
        if cut[k] or brk:
            if s is not None:
                out.append((s, k)); s = None
        if not cut[k] and s is None:
            s = k
    if s is not None:
        out.append((s, len(cut)))
    return out


def run_speed(tr, cut, a, b):
    """centroid speed on the uncut run nearest to [a, b]: linear fit over the RUN_WIN s of the run next to the window"""
    t, P = tr.t, tr.uv
    best = None
    for i, j in runs(cut, t):
        if j - i < RUN_N or t[j - 1] - t[i] < RUN_DUR - 1e-9:
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
        return dict(run_px=np.nan, run_gap=np.nan, run_side="none")
    gap, side, i, j = best
    tt, PP = t[i:j], P[i:j]
    if side == "before":
        m = tt >= tt[-1] - RUN_WIN
    elif side == "after":
        m = tt <= tt[0] + RUN_WIN
    else:
        c = 0.5 * (a + b); m = np.abs(tt - c) <= RUN_WIN
    if m.sum() < 3:
        m = np.ones(len(tt), bool)
    v = np.array([np.polyfit(tt[m], PP[m, k], 1)[0] for k in range(2)])
    return dict(run_px=float(np.linalg.norm(v)), run_gap=round(float(gap), 3), run_side=side, run_n=int(j - i),
                run_t0=round(float(tt[0]), 2), run_t1=round(float(tt[-1]), 2), run_nfit=int(m.sum()),
                run_vx=float(v[0]), run_vy=float(v[1]))


def avg_surgical(tr, q, P2, S):
    """avgspeed as measure_v3.avg_speed_c (mean |v| of 0.15 s fits at every frame in [a, b]), where only the fits whose own
    window holds a border-cut frame read the free-edge trajectory P2; every other frame keeps its centroid fit"""
    t, P, sg = tr.t, tr.uv, tr.segs()
    lo, hi = t.min(), t.max()
    a = lo if q.get("t1") is None else max(q["t1"], lo)
    b = hi if q.get("t2") is None else min(q["t2"], hi)
    if b - a < 0.1:
        return None, 0
    m = np.where((t >= a - 1e-9) & (t <= b + 1e-9))[0]
    if len(m) < 3:
        return None, 0
    cut = S.any(1)
    vs, nch = [], 0
    for i in m:
        wi = window_idx(t, P, t[i], sg, 0.15)
        use = P2 if (len(wi) and cut[wi].any()) else P
        nch += int(use is P2)
        v = MV.vel_at_c(t, use, t[i], sg, 0.15)
        if v is not None:
            vs.append(float(np.linalg.norm(v)))
    return (float(np.mean(vs)) if vs else None), nch


def px_on(tr, kind, q, P, sg=None):
    t = tr.t
    sg = tr.segs() if sg is None else sg
    lo, hi = t.min(), t.max()
    if kind == "speed":
        t0 = MV.clamp_time(tr, q["t"])
        v = MV.vel_at_c(t, P, t0, sg, MV.HALF_V)
        return None if v is None else float(np.linalg.norm(v))
    a = lo if q.get("t1") is None else max(q["t1"], lo)
    b = hi if q.get("t2") is None else min(q["t2"], hi)
    if b - a < 0.1:
        return None
    return MV.avg_speed_c(t, P, a, b, sg)


if __name__ == "__main__":
    raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    MV.TRACK_VER = "v2sc"
    byv, vjs = MV.load_split_tracks(SPLIT, "e")
    MV.TRACK_VER = "v3ntmv"
    _, vjs3 = MV.load_split_tracks(SPLIT, "e")
    rows = []
    VIDS_ = set(os.environ.get("TEDGE3_VIDS", "").split(",")) - {""}
    for v, recs in byv.items():
        if VIDS_ and v not in VIDS_:
            continue
        cand = [r for r in recs if r["q"]["kind"] in ("speed", "avgspeed") and not (r["q"]["kind"] == "speed" and r["q"].get("t") is None)]
        if not cand:
            continue
        MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
        VV = {}
        for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
            if v in src_:
                VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
        for r in cand:
            q = r["q"]
            d = dict(id=r["id"], vid=v, cat=r["cat"], qk=q["kind"], qt=q.get("t"), qt1=q.get("t1"), qt2=q.get("t2"),
                     tgt=(r.get("target") or {}).get("phrase"))
            rr = raw.loc[r["id"]] if r["id"] in raw.index else None
            if rr is None:
                d["note"] = "no_raw"; rows.append(d); continue
            d.update(mode=str(rr.meas_mode), mtr=rr.meas_tracks, mpp=rr.meas_mpp, geo=rr.geo_value, meas=rr.meas_value, valid=rr.meas_valid)
            mt_ = "v2" if os.environ.get("TEDGE3_OWN") == "1" else (rr.meas_tracks if rr.meas_tracks in VV else "v2")
            if mt_ not in VV:
                d["note"] = "no_track_file"; rows.append(d); continue
            vj, V = VV[mt_]
            W, H = float(vj["W"]), float(vj["H"])
            tr = V.get(r["target"])
            if tr is None or tr.n < 3:
                d["note"] = "no_target_track"; rows.append(d); continue
            d.update(tkey=tr.q.get("key"), trn=tr.n, W=W, H=H)
            old = MV.px_quantity(tr, q["kind"], q.get("t"), q.get("t1"), q.get("t2"), q.get("when"))
            d["old_px"] = old
            if old is not None and np.isfinite(rr.meas_mpp) and rr.meas_mpp > 0 and np.isfinite(rr.meas_value):
                d["rep_err"] = old * rr.meas_mpp / rr.meas_value - 1
            w, win = used_frames(tr, q["kind"], q)
            if w is None or len(w) == 0:
                d["note"] = "no_window"; rows.append(d); continue
            S = sides(tr, W, H)
            cutw = S[w].any(1)
            d.update(nw=len(w), ncut=int(cutw.sum()), fcut=round(float(cutw.mean()), 3), w_t0=round(float(tr.t[w].min()), 2),
                     w_t1=round(float(tr.t[w].max()), 2), sidesw="".join(c for c, f in zip("LRTB", S[w].any(0)) if f))
            cut_all = S.any(1)
            d["clip_track"] = round(float(cut_all.mean()), 3)
            if not cutw.any():
                rows.append(d); continue
            P2, how = free_edge_P(tr, S, w)
            d["edge_how"] = how
            if P2 is not None:
                d["new_px"] = px_on(tr, q["kind"], q, P2)
                if q["kind"] == "avgspeed":          # surgical: only the per-frame fits that see a cut frame change
                    d["new_px_all"] = d["new_px"]
                    d["new_px"], d["n_fits_changed"] = avg_surgical(tr, q, P2, S)
                sg2 = MV.segment(tr.t, P2, tr.size_px)
                d["new_px_seg2"] = px_on(tr, q["kind"], q, P2, sg2)
                box = np.array([tr.st[j]["box"] for j in tr.okc], float)
                Pb = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
                d["boxc_px"] = px_on(tr, q["kind"], q, Pb)
            d.update(run_speed(tr, cut_all, *win))
            # sides the box touches in >= 90% of the track (e.g. a boat whose top touches the frame top all along): the run
            # check then uses the frames free of the other (transient) sides; valid only when the motion is perpendicular
            # to the persistent side's axis (checked in rule.py with run2_vx / run2_vy)
            pers = S.mean(0) >= 0.9
            d["pers"] = "".join(c for c, f in zip("LRTB", pers) if f)
            if pers.any():
                cut_tr = S[:, ~pers].any(1)
                d.update({k.replace("run_", "run2_"): v_ for k, v_ in run_speed(tr, cut_tr, *win).items()})
            # dynamic prior (D rows): is the prior's own px kinematics read from border-cut boxes too? Then the scale carries
            # the same centroid bias and correcting only the target would break a cancellation -> rule.py skips the row.
            pr = r["priors"][0] if r.get("priors") else None
            if pr is not None and pr.get("kind") in ("speed", "avgspeed", "accel") and not pr.get("gravity"):
                ptr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
                d["p_kind"] = pr.get("kind"); d["p_t"] = pr.get("t"); d["p_t1"] = pr.get("t1"); d["p_t2"] = pr.get("t2")
                if ptr is not None and ptr.n >= 3:
                    d["p_same"] = bool(ptr is tr or (ptr.q.get("obj") is not None and ptr.q.get("obj") == tr.q.get("obj")))
                    Sp = sides(ptr, W, H)
                    persp = Sp.mean(0) >= 0.9
                    cutp = Sp[:, ~persp].any(1)
                    if pr.get("t") is not None:
                        t0p = MV.clamp_time(ptr, pr["t"])
                        wp = window_idx(ptr.t, ptr.uv, t0p, ptr.segs(), MV.HALF_A if pr.get("kind") == "accel" else MV.HALF_V) if t0p is not None else np.array([], int)
                    elif pr.get("t1") is not None:
                        wp = np.where((ptr.t >= pr["t1"] - 0.15) & (ptr.t <= (pr.get("t2") if pr.get("t2") is not None else ptr.t.max()) + 0.15))[0]
                    else:
                        wp = np.arange(ptr.n)
                    d["p_ncut"] = int(cutp[wp].sum()) if len(wp) else -1
                    d["p_nw"] = int(len(wp))
                else:
                    d["p_ncut"] = -2
            d["size_px"] = round(tr.size_px, 1)
            box = np.array([tr.st[j]["box"] for j in tr.okc], float)
            unc = np.where(~cut_all)[0]
            if len(unc):
                d["wfull"] = float(np.median(box[unc, 2] - box[unc, 0])); d["hfull"] = float(np.median(box[unc, 3] - box[unc, 1]))
            d["wwin"] = float(np.median(box[w, 2] - box[w, 0])); d["hwin"] = float(np.median(box[w, 3] - box[w, 1]))
            rows.append(d)
    D = pd.DataFrame(rows).set_index("id")
    D.to_csv(f"{R}/cache/tmp/ag_tedge3_diag{os.environ.get('TEDGE3_TAG', 'own')}_{SPLIT}.csv")
    nc = D["ncut"] if "ncut" in D else pd.Series(0, index=D.index)
    print(SPLIT, "candidate rows", len(D), "videos", D.vid.nunique(), "| cut in window:", int((nc > 0).sum()),
          "| notes:", D["note"].value_counts().to_dict() if "note" in D else {})
    print("rep_err |.|>1%:", int((D.get("rep_err", pd.Series(dtype=float)).abs() > 0.01).sum()), "of", int(D.get("rep_err", pd.Series(dtype=float)).notna().sum()))
    X = D[nc > 0]
    cols = [c for c in ["vid", "cat", "qk", "qt", "qt1", "qt2", "mode", "old_px", "rep_err", "nw", "ncut", "sidesw", "edge_how", "new_px",
                        "new_px_seg2", "boxc_px", "run_px", "run_gap", "run_side", "clip_track", "wwin", "wfull"] if c in X]
    print(X[cols].round(3).to_string())
