import os, sys, re, json, math, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import UNIT_SI

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCENE = r"\b(lanes?|road|street|court|tiles?|floor|ceiling|room|wall|field|track|pool|river|sidewalk|crosswalk|ground|sky|water|ruler|calibre|window frame|markings?|gap|opening|surface|deck|lattice|frame of)\b"
PART = r"\b(shoulders?|arms?|legs?|heads?|hands?|feet|foot|fingers?|fingertip|tip|edges?|corners?|rim|underside|handle|lid|neck|tail|ears?|knee|waist|chest|breadth)\b"
WINGED = r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft)\b"
DYN = {"speed", "avgspeed", "hvel", "accel", "disp", "path", "maxheight", "orbitdiam"}


class Track:
    def __init__(self, q, vid):
        self.q = q
        fps = vid["fps"]
        t_all = np.array(vid["fidx"], float) / fps
        st = [f["st"] for f in q["frames"]]
        val = np.array([s is not None for s in st])
        area = np.array([s["area"] if s else 0 for s in st], float)
        med = np.median(area[val]) if val.any() else 0
        ok = val & (area > 0.4 * med) & (area < 2.5 * med) if med > 0 else val
        border = np.array([s["border"] if s else 9 for s in st])
        self.okc = np.where(ok)[0]
        self.oke = np.where(ok & (border == 0))[0]
        if len(self.oke) < max(2, 0.3 * len(self.okc)):
            self.oke = self.okc
        self.st = st
        self.t_all = t_all
        self.t = t_all[self.okc]
        self.uv = np.array([st[j]["c"] for j in self.okc], float).reshape(-1, 2)
        self.best = q.get("best_score", 0.0)
        ious = q.get("det_iou") or []
        self.iou = float(np.median(ious)) if ious else 0.0
        self.n = len(self.okc)

    def ext_px(self, dim, phrase):
        js = self.oke
        if len(js) == 0:
            return np.array([]), js
        box = np.array([self.st[j]["box"] for j in js], float)
        rr = np.array([self.st[j]["rr"][:2] for j in js], float)
        bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
        winged = bool(re.search(WINGED, phrase or ""))
        if dim == "height":
            e = bh
        elif dim in ("length", "wingspan", "size"):
            e = rr[:, 0]
        elif dim == "width":
            e = rr[:, 0] if winged else rr[:, 1]
        elif dim in ("thickness", "side"):
            e = rr[:, 1]
        elif dim in ("diameter", "radius"):
            e = rr[:, 1]
        else:
            e = rr[:, 0]
        if dim == "radius":
            e = e / 2
        return e, js


def agg_ext(e, dim):
    if len(e) == 0:
        return np.nan
    return float(np.percentile(e, 75)) if dim in ("diameter", "radius") else float(np.median(e))


def _fit(t, P, t0, half, deg, minpts):
    for h in (half, half * 1.6, half * 2.5):
        m = np.abs(t - t0) <= h + 1e-9
        if m.sum() >= minpts:
            break
    if m.sum() < minpts:
        return None
    tt = t[m] - t0
    deg = min(deg, m.sum() - 1)
    return [np.polyfit(tt, P[m, k], deg) for k in range(P.shape[1])], deg


def pos_at(t, P, t0):
    r = _fit(t, P, t0, 0.08, 1, 2)
    if r is None:
        return None
    return np.array([c[-1] for c in r[0]])


def vel_at(t, P, t0, half=0.25):
    r = _fit(t, P, t0, half, 2, 5)
    if r is None:
        r = _fit(t, P, t0, half, 1, 3)
        if r is None:
            return None
    cs, deg = r
    return np.array([c[-2] for c in cs])


def acc_at(t, P, t0, half=0.4):
    r = _fit(t, P, t0, half, 2, 7)
    if r is None or r[1] < 2:
        return None
    return np.array([2 * c[-3] for c in r[0]])


def acc_interval(t, P, a, b):
    m = (t >= a - 1e-9) & (t <= b + 1e-9)
    if m.sum() < 6:
        return None
    return np.array([2 * np.polyfit(t[m], P[m, k], 2)[0] for k in range(P.shape[1])])


def avg_speed(t, P, a, b):
    m = np.where((t >= a - 1e-9) & (t <= b + 1e-9))[0]
    if len(m) < 3:
        return None
    vs = [vel_at(t, P, t[i], 0.15) for i in m]
    vs = [np.linalg.norm(v) for v in vs if v is not None]
    return float(np.mean(vs)) if vs else None


def clamp_time(tr, t0, tol=0.25):
    if tr.n == 0:
        return None
    lo, hi = tr.t.min(), tr.t.max()
    if t0 < lo - tol or t0 > hi + tol:
        return None
    return float(min(max(t0, lo), hi))


def depth_fn(depth, keys):
    ent = [d for d in depth if d["key"] in keys]
    if not ent:
        return None
    per = {}
    for d in ent:
        per.setdefault(d["key"], []).append(d)
    fns = []
    for k, es in per.items():
        timed = sorted([(e["t"], e["d"]) for e in es if e["t"] is not None])
        untimed = [e["d"] for e in es if e["t"] is None]
        if len(timed) >= 2:
            ts = np.array([x[0] for x in timed]); ds = np.array([x[1] for x in timed])
            lo, hi = ds.min() * 0.6, ds.max() * 1.6
            def f(t, ts=ts, ds=ds, lo=lo, hi=hi):
                t = np.atleast_1d(np.asarray(t, float))
                out = np.interp(t, ts, ds)
                s0 = (ds[1] - ds[0]) / (ts[1] - ts[0] + 1e-9); s1 = (ds[-1] - ds[-2]) / (ts[-1] - ts[-2] + 1e-9)
                out = np.where(t < ts[0], ds[0] + s0 * (t - ts[0]), out)
                out = np.where(t > ts[-1], ds[-1] + s1 * (t - ts[-1]), out)
                return np.clip(out, lo, hi)
        else:
            v = timed[0][1] if timed else float(np.mean(untimed))
            def f(t, v=v):
                return np.full(np.atleast_1d(np.asarray(t, float)).shape, v)
        fns.append(f)
    return lambda t: np.mean([f(t) for f in fns], axis=0)


SCALE_DEPTH = True


def track_depth_fn(depth, keys, trk):
    base = depth_fn(depth, keys)
    if base is None or trk is None or trk.n < 5 or not SCALE_DEPTH:
        return base
    ent = [d for d in depth if d["key"] in keys]
    timed = [d for d in ent if d["t"] is not None]
    per_key_timed = max([sum(1 for d in timed if d["key"] == k) for k in keys] + [0])
    if per_key_timed >= 2:
        return base
    s = np.sqrt(np.array([trk.st[j]["area"] for j in trk.okc], float))
    if len(s) >= 5:
        s = np.array([np.median(s[max(0, i - 2):i + 3]) for i in range(len(s))])
    lo, hi = np.percentile(s, 10), np.percentile(s, 90)
    if hi / max(lo, 1e-6) < 1.15:
        return base
    if timed:
        t0 = timed[0]["t"]
        s0 = float(np.interp(t0, trk.t, s))
        D0 = float(np.mean([d["d"] for d in timed]))
    else:
        s0 = float(np.median(s)); D0 = float(np.mean([d["d"] for d in ent]))
    Dtr = np.clip(D0 * s0 / s, D0 / 3, D0 * 3)
    tt = trk.t.copy()
    return lambda t: np.interp(np.atleast_1d(np.asarray(t, float)), tt, Dtr)


def to3d(uv, D, f, cx, cy):
    r = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(len(uv))], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True) * D[:, None], np.linalg.norm(r, axis=1)


def gravity_ay(tr, a=None, b=None):
    if tr is None or tr.n < 7:
        return None
    t, P = tr.t, tr.uv
    if a is not None and b is not None:
        r = acc_interval(t, P, a, b)
        if r is not None:
            return float(r[1])
    vals = []
    for t0 in np.arange(t.min() + 0.15, t.max() - 0.15 + 1e-9, 0.05):
        r = _fit(t, P, t0, 0.2, 2, 7)
        if r and r[1] == 2:
            vals.append(2 * r[0][1][0])
    if len(vals) < 3:
        return None
    return float(np.median(vals))


class Video:
    def __init__(self, vj):
        self.vj = vj
        self.W, self.H = vj["W"], vj["H"]
        self.tr = {q["key"]: Track(q, vj) for q in vj["queries"]}
        cam = np.array(vj["cam"], float)
        self.cam_shift = np.cumsum(cam[:, :2], 0) if len(cam) else np.zeros((1, 2))
        self.cam_mag = float(np.abs(self.cam_shift).max()) if len(cam) else 0.0

    def get(self, ent):
        if not ent or not ent.get("phrase"):
            return None
        k = ent["phrase"] + "|" + ",".join(ent.get("where") or [])
        return self.tr.get(k)


def track_ok(tr, dynamic, depth_moves=False):
    if tr is None or tr.n == 0:
        return False, "no_track"
    if tr.best < 0.3:
        return False, f"low_det({tr.best:.2f})"
    if tr.iou and tr.iou < 0.5:
        return False, f"det_track_iou({tr.iou:.2f})"
    if tr.n < (6 if dynamic else 3):
        return False, f"few_frames({tr.n})"
    if dynamic and not depth_moves:
        size = np.median([max(tr.st[j]["rr"][0], 1.0) for j in tr.okc])
        rng = float(np.linalg.norm(tr.uv.max(0) - tr.uv.min(0)))
        if rng < 0.25 * size:
            return False, f"static_track({rng:.0f}px<{0.25 * size:.0f})"
    return True, ""


CONTRAST = {"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray",
            "silver", "big", "small", "large", "little", "left", "right", "upper", "lower", "top", "bottom", "front",
            "back", "near", "far", "tall", "short", "first", "second", "third", "middle", "center"}


def same_object(a, b):
    if a is None or b is None or a is b:
        return a is b and a is not None
    qa, qb = a.q.get("anchor"), b.q.get("anchor")
    if not qa or not qb or qa[0] != qb[0]:
        return False
    x = qa[1]; y = qb[1]
    ix = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
    ua = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - ix
    return ua > 0 and ix / ua > 0.9


def contrastive(pa, pb, heads=None):
    wa = set(re.findall(r"[a-z]+", pa or "")) & CONTRAST
    wb = set(re.findall(r"[a-z]+", pb or "")) & CONTRAST
    ha, hb = heads if heads else (re.findall(r"[a-z]+", pa or "")[-1:], re.findall(r"[a-z]+", pb or "")[-1:])
    if ha != hb:
        return True
    return bool(wa) and bool(wb) and wa != wb


def px_quantity(tr, kind, dim, phrase, t=None, t1=None, t2=None, when=None, tr2=None, P=None, horizontal=False):
    tt = tr.t
    PP = tr.uv if P is None else P
    if kind == "size":
        return None
    lo, hi = (tt.min(), tt.max()) if tr.n else (0, 0)
    a = lo if t1 is None else max(t1, lo)
    b = hi if t2 is None else min(t2, hi)
    if kind in ("speed",):
        if t is None:
            return avg_speed(tt, PP, lo, hi)
        t0 = clamp_time(tr, t)
        if t0 is None:
            return None
        v = vel_at(tt, PP, t0)
        return None if v is None else float(np.linalg.norm(v))
    if kind == "avgspeed":
        if b - a < 0.1:
            return None
        return avg_speed(tt, PP, a, b)
    if kind == "hvel":
        pa, pb = pos_at(tt, PP, a), pos_at(tt, PP, b)
        if pa is None or pb is None or b - a < 0.1:
            return None
        return abs(pb[0] - pa[0]) / (b - a)
    if kind == "accel":
        if t is not None:
            t0 = clamp_time(tr, t)
            if t0 is None:
                return None
            ac = acc_at(tt, PP, t0)
        elif t1 is not None:
            ac = acc_interval(tt, PP, a, b)
        else:
            ac = acc_interval(tt, PP, lo, hi)
        return None if ac is None else float(np.linalg.norm(ac))
    if kind in ("disp",):
        if t1 is None:
            a, b = lo, hi
        ca, cb = clamp_time(tr, a), clamp_time(tr, b)
        if ca is None or cb is None:
            return None
        pa, pb = pos_at(tt, PP, ca), pos_at(tt, PP, cb)
        return None if pa is None or pb is None else float(np.linalg.norm(pb - pa))
    if kind == "path":
        s = avg_speed(tt, PP, a, b)
        return None if s is None else s * (b - a)
    if kind == "maxheight":
        return float(PP[:, 1].max() - PP[:, 1].min()) if tr.n >= 3 else None
    if kind == "orbitdiam":
        if tr.n < 5:
            return None
        d = np.linalg.norm(PP[:, None, :] - PP[None, :, :], axis=2)
        return float(d.max())
    return None


def dist2_px(tr1, tr2, t, when, P1=None, P2=None):
    if tr1 is None or tr2 is None or tr1.n == 0 or tr2.n == 0:
        return None
    A = tr1.uv if P1 is None else P1
    B = tr2.uv if P2 is None else P2
    if t is None and when is None:
        common = sorted(set(np.round(tr1.t, 4)) & set(np.round(tr2.t, 4)))
        if not common:
            return None
        ds = [np.linalg.norm(pos_at(tr1.t, A, c) - pos_at(tr2.t, B, c)) for c in common]
        return float(np.median(ds))
    if when == "start":
        t = max(tr1.t.min(), tr2.t.min())
    elif when == "end":
        t = min(tr1.t.max(), tr2.t.max())
    c1, c2 = clamp_time(tr1, t), clamp_time(tr2, t)
    if c1 is None or c2 is None:
        return None
    p1, p2 = pos_at(tr1.t, A, c1), pos_at(tr2.t, B, c2)
    return None if p1 is None or p2 is None else float(np.linalg.norm(p1 - p2))


def measure_video(recs, vj, fdef):
    V = Video(vj)
    cx, cy = V.W / 2.0, V.H / 2.0
    out = []
    calib_cache = {}
    for r in recs:
        res = dict(id=r["id"], kind=r["q"]["kind"], ok=False, why="", val_si=np.nan, mpp=np.nan, f=np.nan, mode="")
        q = r["q"]
        kind = q["kind"]
        is3d = r["video_type"][1] == "3"
        pr = r["priors"][0] if r["priors"] else None
        tgt = V.get(r["target"])
        tgt2 = V.get(r["target2"]) if r.get("target2") else None
        prior_tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
        if kind == "other" or not q.get("target"):
            res["why"] = "unparsed_or_other"; out.append(res); continue
        if re.search(SCENE, r["target"]["raw"] or "") or (r.get("target2") and re.search(SCENE, (r["target2"].get("raw") or ""))):
            res["why"] = "scene_target"; out.append(res); continue
        if re.search(PART, re.sub(r"\([^)]*\)", " ", r["question"].lower())) and kind == "size":
            res["why"] = "part_target"; out.append(res); continue
        if pr is None or pr.get("value_si") is None:
            res["why"] = "no_prior"; out.append(res); continue
        if (pr.get("obj") and re.search(SCENE, pr["obj"])):
            res["why"] = "scene_prior"; out.append(res); continue
        dyn = kind in DYN
        def depth_moves(keys):
            ds = [d["d"] for d in r["depth"] if d["key"] in keys and d["t"] is not None]
            return len(ds) >= 2 and (max(ds) - min(ds)) > 0.05 * max(ds)
        okt, why = track_ok(tgt, dyn, is3d and depth_moves(r.get("target_dkeys") or []))
        if not okt:
            res["why"] = "target_" + why; out.append(res); continue
        if kind == "dist2":
            ok2, why2 = track_ok(tgt2, False)
            if not ok2:
                res["why"] = "target2_" + why2; out.append(res); continue
        gravity = pr.get("gravity", False)
        if gravity and prior_tr is None:
            best, ba = None, 0
            for k, trk in V.tr.items():
                if not track_ok(trk, True)[0]:
                    continue
                ay = gravity_ay(trk, pr.get("t1"), pr.get("t2"))
                if ay is not None and ay > ba:
                    best, ba = trk, ay
            prior_tr = best
        okp, whyp = track_ok(prior_tr, pr["kind"] != "size", is3d and depth_moves(r.get("prior_dkeys") or []))
        if not okp:
            res["why"] = "prior_" + whyp; out.append(res); continue
        tph = (r["target"]["phrase"] or "") + " " + " ".join(r["target"].get("where") or [])
        pph = (r["prior_obj"].get("phrase") or "") + " " + " ".join(r["prior_obj"].get("where") or [])
        if prior_tr is not tgt and same_object(prior_tr, tgt) and r["prior_obj"].get("phrase") and contrastive(tph, pph, (re.findall(r"[a-z]+", r["target"]["phrase"] or "")[-1:], re.findall(r"[a-z]+", r["prior_obj"]["phrase"] or "")[-1:])):
            res["why"] = "target=prior_object"; out.append(res); continue
        if kind == "dist2" and same_object(tgt, tgt2):
            res["why"] = "target=target2"; out.append(res); continue
        pdim = pr.get("dim") or "length"
        tk, pk = r.get("target_dkeys") or [], r.get("prior_dkeys") or []
        Dt = track_depth_fn(r["depth"], tk, tgt) if is3d else None
        Dp = track_depth_fn(r["depth"], pk, prior_tr) if is3d else None
        same_obj = prior_tr is tgt
        use3d = is3d and (Dt is not None or Dp is not None) and not (same_obj and Dp is None)

        def prior_px(P=None):
            if pr["kind"] == "size":
                e, js = prior_tr.ext_px(pdim, r["prior_obj"]["phrase"])
                return agg_ext(e, pdim) if P is None else None
            if gravity:
                ay = gravity_ay(prior_tr, pr.get("t1"), pr.get("t2")) if P is None else None
                return ay
            kd = "speed" if pr["kind"] == "speed" else "accel"
            return px_quantity(prior_tr, kd, None, None, t=pr.get("t"), t1=pr.get("t1"), t2=pr.get("t2"), P=P)

        if not use3d:
            ppx = prior_px()
            if ppx is None or not np.isfinite(ppx) or ppx <= 1e-6:
                res["why"] = "prior_px_fail"; out.append(res); continue
            k_prior = {"size": 0, "speed": 1, "accel": 2}[pr["kind"]]
            mpp = pr["value_si"] / ppx
            if kind == "size":
                e, js = tgt.ext_px(q["dim"], r["target"]["phrase"])
                v = agg_ext(e, q["dim"])
            elif kind == "dist2":
                v = dist2_px(tgt, tgt2, q["t"], q["when"])
            else:
                v = px_quantity(tgt, kind, q["dim"], r["target"]["phrase"], q["t"], q["t1"], q["t2"], q["when"])
            if v is None or not np.isfinite(v) or v <= 0:
                res["why"] = "target_px_fail"; out.append(res); continue
            res.update(val_si=v * mpp, mpp=mpp, mode="2d_ratio" if not is3d else "3d_samedepth", ok=True)
            out.append(res); continue
        Wd = V.W

        def traj3d(trk, Dfn, f):
            D = Dfn(trk.t)
            P, nr = to3d(trk.uv, D, f, cx, cy)
            return P, D / nr

        def prior_metric(f):
            Dfn = Dp if Dp is not None else Dt
            if pr["kind"] == "size":
                e, js = prior_tr.ext_px(pdim, r["prior_obj"]["phrase"])
                if len(e) == 0:
                    return None
                uv = np.array([prior_tr.st[j]["c"] for j in js], float)
                D = Dfn(prior_tr.t_all[js])
                _, nr = to3d(uv, D, f, cx, cy)
                return agg_ext(e * (D / nr) / f, pdim)
            P, _ = traj3d(prior_tr, Dfn, f)
            if gravity:
                a, b = pr.get("t1"), pr.get("t2")
                if a is not None:
                    ac = acc_interval(prior_tr.t, P, a, b)
                    return None if ac is None else abs(float(ac[1]))
                vals = []
                for t0 in np.arange(prior_tr.t.min() + 0.15, prior_tr.t.max() - 0.15 + 1e-9, 0.05):
                    rr_ = _fit(prior_tr.t, P, t0, 0.2, 2, 7)
                    if rr_ and rr_[1] == 2:
                        vals.append(2 * rr_[0][1][0])
                return abs(float(np.median(vals))) if len(vals) >= 3 else None
            kd = "speed" if pr["kind"] == "speed" else "accel"
            return px_quantity(prior_tr, kd, None, None, t=pr.get("t"), t1=pr.get("t1"), t2=pr.get("t2"), P=P)

        f_cal, mode = None, ""
        ckey = (id(prior_tr), tuple(pk), tuple(tk) if (Dp is None and same_obj) else ())
        if ckey in calib_cache:
            f_cal, mode = calib_cache[ckey]
        elif Dp is not None or (Dt is not None and same_obj):
            grid = Wd * np.logspace(np.log10(0.25), np.log10(12), 120)
            vals = []
            for f in grid:
                m = prior_metric(f)
                vals.append(np.nan if (m is None or not np.isfinite(m) or m <= 0) else m)
            vals = np.array(vals)
            if np.isfinite(vals).sum() >= 3:
                err = np.abs(np.log(vals / pr["value_si"]))
                err[~np.isfinite(err)] = np.inf
                i = int(np.argmin(err))
                if err[i] < 0.2 and 0 < i < len(grid) - 1:
                    f_cal, mode = float(grid[i]), "3d_calib"
            calib_cache[ckey] = (f_cal, mode)
        fmed = fdef.get(r["video_source"])
        if f_cal is not None and fmed is not None and not (0.5 <= f_cal / fmed <= 2.0):
            f_cal, mode = None, "3d_calibrejected"
        if f_cal is None:
            f_cal = fmed if fmed is not None else 1.0 * Wd
            mode = "3d_fovdefault" if mode == "" else mode + "_fovdefault"
        Dfn_t = Dt if Dt is not None else Dp
        if kind == "size":
            e, js = tgt.ext_px(q["dim"], r["target"]["phrase"])
            if len(e) == 0:
                res["why"] = "target_px_fail"; out.append(res); continue
            uv = np.array([tgt.st[j]["c"] for j in js], float)
            D = Dfn_t(tgt.t_all[js])
            _, nr = to3d(uv, D, f_cal, cx, cy)
            v = agg_ext(e * (D / nr) / f_cal, q["dim"])
        elif kind == "dist2":
            D2 = track_depth_fn(r["depth"], r.get("target2_dkeys") or [], tgt2) or Dfn_t
            P1, _ = traj3d(tgt, Dfn_t, f_cal)
            P2, _ = traj3d(tgt2, D2, f_cal)
            v = dist2_px(tgt, tgt2, q["t"], q["when"], P1, P2)
        else:
            P, Z = traj3d(tgt, Dfn_t, f_cal)
            if kind == "maxheight":
                Y = (tgt.uv[:, 1] - cy) * Z / f_cal
                v = float(Y.max() - Y.min()) if tgt.n >= 3 else None
            else:
                v = px_quantity(tgt, kind, q["dim"], r["target"]["phrase"], q["t"], q["t1"], q["t2"], q["when"], P=P)
        if v is None or not np.isfinite(v) or v <= 0:
            res["why"] = "target_px_fail"; out.append(res); continue
        res.update(val_si=float(v), f=f_cal / Wd, mode=mode + ("" if Dt is not None else "_tdepth=prior"), ok=True)
        out.append(res)
    return out, V


PLAUS = {0: (1e-4, 500.0), 1: (1e-4, 400.0), 2: (1e-3, 300.0)}


def load_split_tracks(split):
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}.jsonl")]
    byv = {}
    for r in recs:
        r["video_source"] = r["video_source"].replace("segmentaion", "segmentation")
        byv.setdefault(r["video_id"], []).append(r)
    tdir = f"{R}/geo/out/tracks_{split}"
    vjs = {}
    for v in byv:
        p = f"{tdir}/{v}.json"
        if os.path.exists(p):
            vjs[v] = json.load(open(p))
    return byv, vjs


def main():
    global SCALE_DEPTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--no_scale_depth", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    SCALE_DEPTH = not a.no_scale_depth
    byv, vjs = load_split_tracks(a.split)
    fmed = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
    print("fallback focal f/W per source:", {s: round(f, 3) for s, f in fmed.items()})
    allf = float(np.median(list(fmed.values()))) if fmed else 1.0
    rows = []
    for v, rs in byv.items():
        if v not in vjs:
            for r in rs:
                rows.append(dict(id=r["id"], kind=r["q"]["kind"], ok=False, why="no_track_file"))
            continue
        W = vjs[v]["W"]
        fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * W for r in rs}
        o, V = measure_video(rs, vjs[v], fdd)
        for x, r in zip(o, rs):
            q = r["q"]
            u = q["unit"]
            k = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(u, 2 if "^2" in u else 1)
            x["unit"] = u
            x["value"] = x["val_si"] / UNIT_SI[u] if x.get("ok") else np.nan
            if x.get("ok"):
                lo, hi = PLAUS[k]
                if not (lo <= x["val_si"] <= hi):
                    x["ok"] = False; x["why"] = f"implausible({x['val_si']:.3g})"
            x.update(video_id=v, cat=r["cat"], source=r["video_source"], cam_mag=V.cam_mag / V.W)
            rows.append(x)
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/geo/out/meas_{a.split}{a.tag}.csv", index=False)
    print("measured share", round(df.ok.mean(), 3), df.groupby("kind").ok.mean().round(2).to_dict())
    print(df.why.value_counts().head(25).to_dict())
    print(df["mode"].value_counts().to_dict() if "mode" in df else "")


if __name__ == "__main__":
    main()
