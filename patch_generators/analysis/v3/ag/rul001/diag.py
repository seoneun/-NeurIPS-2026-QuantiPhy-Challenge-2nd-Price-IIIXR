"""rul001 diagnostics: for rows whose prior is an acceleration (incl. gravity) and whose geo mode is 2d_ratio, find the
prior track and the frames the acceleration fit uses; report border clipping, frame count and quadratic R^2 there."""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 40)
SPLIT = sys.argv[1]


def border_flags(tr, idx, W, H, tol=2.0):
    out = []
    for k in idx:
        j = tr.okc[k]
        s = tr.st[j]
        b = s["box"]
        cut = (b[0] <= tol) or (b[1] <= tol) or (b[2] >= W - 1 - tol) or (b[3] >= H - 1 - tol)
        out.append(bool(cut or s.get("border", 0) != 0))
    return np.array(out, bool)


def quad_r2(t, P):
    if len(t) < 4:
        return np.nan, np.nan
    tc = t - t.mean()
    cs = [np.polyfit(tc, P[:, k], 2) for k in range(P.shape[1])]
    res = sum(float(np.sum((P[:, k] - np.polyval(cs[k], tc)) ** 2)) for k in range(P.shape[1]))
    tot = sum(float(np.sum((P[:, k] - P[:, k].mean()) ** 2)) for k in range(P.shape[1]))
    # share of the straight-line (constant velocity) residual that the curvature explains
    cl = [np.polyfit(tc, P[:, k], 1) for k in range(P.shape[1])]
    resl = sum(float(np.sum((P[:, k] - np.polyval(cl[k], tc)) ** 2)) for k in range(P.shape[1]))
    return (1 - res / tot if tot > 0 else 0.0), (1 - res / resl if resl > 0 else 0.0)


def accel_window(tr, pr):
    """indices (into tr.t) used by prior_kin for a non-gravity acceleration prior"""
    t, P = tr.t, tr.uv
    sg = tr.segs()
    if pr.get("t") is not None and pr.get("t1") is None:
        t0 = pr["t"] + (0.1 if re.search(r"\bafter\b", pr.get("raw", "")) else 0.0)
        t0 = MV.clamp_time(tr, t0)
        if t0 is None:
            return None, "clamp_fail"
        k = MV.seg_index(sg, t, t0)
        if k is None:
            return np.arange(len(t)), "noseg"
        i, j = sg["segs"][k]
        ts = t[i:j]; t0c = min(max(t0, ts.min()), ts.max())
        for h in (MV.HALF_A, MV.HALF_A * 1.6, MV.HALF_A * 2.5):
            m = np.abs(ts - t0c) <= h + 1e-9
            if m.sum() >= 7:
                return i + np.where(m)[0], f"t_win{h:.2f}_seg{k}"
        return np.arange(i, j), f"t_wholeseg{k}"
    if pr.get("t1") is not None:
        a, b = pr["t1"], pr.get("t2")
        lo, hi = t.min(), t.max()
        a = max(a, lo); b = hi if b is None else min(b, hi)
        ks = sorted(set(k for k, (i, j) in enumerate(sg["segs"]) if t[j - 1] >= a - 1e-9 and t[i] <= b + 1e-9))
        if not ks:
            return None, "no_seg_in_interval"
        if len(ks) == 1:
            i, j = sg["segs"][ks[0]]
            m = (t[i:j] >= a - 1e-9) & (t[i:j] <= b + 1e-9)
            if m.sum() < 6:
                return np.arange(i, j), "int_wholeseg"
            return i + np.where(m)[0], "int_inseg"
        m = (t >= a - MV.HALF_V) & (t <= b + MV.HALF_V)
        return np.where(m)[0], f"int_{len(ks)}segs_dv"
    best, bl = None, 0.0
    for (i, j), rest in zip(sg["segs"], sg["rest"]):
        if rest or j - i < 5:
            continue
        L = t[j - 1] - t[i]
        if L > bl:
            best, bl = (i, j), L
    if best is None:
        return None, "no_moving_seg"
    return np.arange(*best), "longest_seg"


def ff_pieces(tr, a=None, b=None):
    """free-fall pieces accepted by MV.freefall (re-implemented ff_test, same thresholds)"""
    t = tr.t; Q = tr.uv; sg = tr.segs(); sig = sg["sigma"]
    ff_short = [float(x) for x in os.environ["QP_FF_SHORT"].split(":")]
    ff_sj = float(os.environ.get("QP_FF_SIZEJIT", "0") or 0)

    def ff_test(idx):
        n = len(idx)
        if n < 6:
            return None
        dur = t[idx[-1]] - t[idx[0]]
        short = dur < float(os.environ.get("QP_FF_DUR", 0.15))
        if short and (ff_short is None or dur < ff_short[2]):
            return None
        tt = t[idx] - t[idx].mean()
        cy, se, s_r, keep = MV.robust_quad(tt, Q[idx, 1], sig)
        ay = 2 * cy[0]
        if not (ay > 0 and cy[0] / se >= float(os.environ.get("QP_FF_SIG", 5))):
            return None
        if short and not (cy[0] / se >= ff_short[0] and s_r <= ff_short[1]):
            return None
        ah = [2 * np.polyfit(tt[keep], Q[idx, k][keep], 2)[0] for k in range(Q.shape[1]) if k != 1]
        if math.sqrt(sum(x * x for x in ah)) > float(os.environ.get("QP_FF_H", 0.35)) * ay:
            return None
        lim = 3 * max(sig, 0.5, ff_sj * getattr(tr, "size_px", 0.0)) if ff_sj > 0 else 3 * max(sig, 0.5)
        if s_r > lim:
            return None
        if n >= 12:
            h = n // 2
            a1 = 2 * np.polyfit(tt[:h], Q[idx[:h], 1], 2)[0]
            a2 = 2 * np.polyfit(tt[h:], Q[idx[h:], 1], 2)[0]
            if not (0.5 * ay <= a1 <= 1.5 * ay and 0.5 * ay <= a2 <= 1.5 * ay):
                return None
        return ay, n, dur
    acc, pieces = [], []
    for (i, j), rest in zip(sg["segs"], sg["rest"]):
        idx = np.arange(i, j)
        if a is not None and b is not None:
            idx = idx[(t[idx] >= a - 0.05) & (t[idx] <= b + 0.05)]
        if len(idx) < 6 or rest:
            continue
        pieces.append(idx)
        r_ = ff_test(idx)
        if r_ is not None:
            acc.append((r_, idx))
    if not acc and str(MV.CUR_VID.get("src", "")).startswith("simulation") and MV.CUR_VID.get("is2d"):
        best_ = None
        for idx in pieces:
            n = len(idx); st_ = max(1, n // 30)
            for i0 in range(0, n - 5, st_):
                for i1 in range(n, i0 + 5, -st_):
                    if best_ is not None and (t[idx[i1 - 1]] - t[idx[i0]]) <= best_[0][2]:
                        break
                    rr = ff_test(idx[i0:i1])
                    if rr is not None:
                        best_ = (rr, idx[i0:i1])
                        break
        if best_ is not None:
            acc.append(best_)
    if not acc:
        return []
    vals = np.array([x[0][0] for x in acc]); med = float(np.median(vals))
    return [x for x, v in zip(acc, vals) if med / 1.5 <= v <= med * 1.5]


raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
MV.TRACK_VER = "v2sc"
byv, vjs = MV.load_split_tracks(SPLIT, "e")
MV.TRACK_VER = "v3ntmv"
_, vjs3 = MV.load_split_tracks(SPLIT, "e")
rows = []
for v, recs in byv.items():
    cand = [r for r in recs if r.get("priors") and (r["priors"][0].get("kind") == "accel" or r["priors"][0].get("gravity"))
            and r["id"] in raw.index and (str(raw.loc[r["id"], "meas_mode"]).startswith("2d_ratio")
                                          or (not r["priors"][0].get("gravity") and "3d" in str(raw.loc[r["id"], "meas_mode"])))]
    if not cand:
        continue
    MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=recs[0]["video_type"][1] == "2")
    VV = {}
    for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
        if v in src_:
            VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
    for r in cand:
        mt_ = raw.loc[r["id"], "meas_tracks"]
        if mt_ not in VV:
            rows.append(dict(id=r["id"], vid=v, note="no_track_file_" + str(mt_))); continue
        vj, V = VV[mt_]
        W, H = float(vj["W"]), float(vj["H"])
        pr = r["priors"][0]
        d = dict(id=r["id"], vid=v, cat=r["cat"], qk=r["q"]["kind"], grav=bool(pr.get("gravity")), pt=pr.get("t"), pt1=pr.get("t1"), pt2=pr.get("t2"),
                 pobj=(r.get("prior_obj") or {}).get("phrase"), tgt=(r.get("target") or {}).get("phrase"), fps=vj.get("fps"),
                 mtr=raw.loc[r["id"], "meas_tracks"], msrc=raw.loc[r["id"], "merge_src"], mode=str(raw.loc[r["id"], "meas_mode"]))
        prior_tr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
        if pr.get("gravity"):
            cands = [prior_tr] if prior_tr is not None else [trk for trk in {id(x): x for x in V.tr.values()}.values() if MV.track_ok(trk, True)[0]]
            best, bev = None, None
            for trk in cands:
                if trk is None:
                    continue
                g_px, inf = MV.freefall(trk, a=pr.get("t1"), b=pr.get("t2"))
                if g_px is not None:
                    ev = (0 if inf.get("sub") else 1, inf["dur"])
                    if bev is None or ev > bev:
                        best, bev = trk, ev
            if best is None:
                d["note"] = "grav_no_ff"; rows.append(d); continue
            pcs = ff_pieces(best, pr.get("t1"), pr.get("t2"))
            if not pcs:
                d["note"] = "grav_pieces_mismatch"; rows.append(d); continue
            tr = best
            idx = max(pcs, key=lambda x: x[0][1])[1]
            d["win"] = f"ff{len(pcs)}"
            d["pkey"] = tr.q.get("key")
        else:
            tr = prior_tr
            if tr is None:
                d["note"] = "no_prior_track"; rows.append(d); continue
            idx, how = accel_window(tr, pr)
            d["win"] = how; d["pkey"] = tr.q.get("key")
            if idx is None:
                d["note"] = how; rows.append(d); continue
        t, P = tr.t[idx], tr.uv[idx]
        bf = border_flags(tr, idx, W, H)
        r2, r2c = quad_r2(t, P)
        tall = np.array(vj["fidx"], float) / vj["fps"]
        nfidx = int(((tall >= t[0] - 1e-9) & (tall <= t[-1] + 1e-9)).sum()) if len(t) else 0
        d.update(n=len(idx), nfidx=nfidx, t0=round(float(t[0]), 2), t1w=round(float(t[-1]), 2), clip=round(float(bf.mean()), 2),
                 nclip=int(bf.sum()), r2=round(r2, 4), r2c=round(r2c, 3), trn=tr.n, trclip=round(float(border_flags(tr, np.arange(tr.n), W, H).mean()), 2),
                 span=round(float(np.linalg.norm(P.max(0) - P.min(0))), 1), size=round(tr.size_px, 1),
                 a_px=(MV.prior_kin(tr, pr) if not pr.get("gravity") else None))
        rows.append(d)
D = pd.DataFrame(rows).set_index("id")
D.to_csv(f"{R}/cache/tmp/ag_accguard_diag_{SPLIT}.csv")
G = D.drop_duplicates([c for c in ["vid", "pkey", "win", "pt", "pt1"] if c in D]) if "pkey" in D else D
print(SPLIT, "rows", len(D), "videos", D.vid.nunique(), "groups", len(G))
cols = [c for c in ["vid", "grav", "pt", "pt1", "pt2", "pobj", "pkey", "win", "n", "nfidx", "t0", "t1w", "clip", "nclip", "r2", "r2c", "trn", "trclip", "span", "size", "fps", "mtr", "msrc", "note"] if c in G]
print(G[cols + ["mode"]].to_string())
