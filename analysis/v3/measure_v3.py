import os, sys, re, json, math, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import UNIT_SI
import measure as M1
import track_common as tc
from measure import (SCENE, PART, WINGED, DYN, agg_ext, _fit, pos_at, vel_at, acc_at, acc_interval, avg_speed, clamp_time,
                     depth_fn, to3d, PLAUS)

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
FIX = set()
T_OWL = 0.2
G = 9.8
TRACK_VER = "v2"
MOGE = {}
DA3D = {}
SCENE_RULER = {}
ROT_CUR = None
CUR_VID = {}
MOGE_STATS = []
HALF_V = 0.25
MEASURE_SCENE = False
_SC_LOC = r"\s+(?:in|on|at|near|from|with|along|across|over|under|above|below|behind|beside|between|next to|inside|outside|into|onto|through|around|that|which|who|where|whose|the|a|an)\s+|,"
_SC_GENERIC = {"line", "lines", "height", "side", "sides", "section", "part", "area", "edge", "edges", "surface", "top", "bottom",
               "end", "length", "width", "distance", "gap", "space", "segment", "portion", "corner", "center", "centre", "middle"}


def scene_phrase(raw):
    raw = (raw or "").lower()
    if os.environ.get("QP_SCENE_HEAD", "") != "1":
        return bool(re.search(SCENE, raw))
    s_ = re.sub(r"^(?:the\s+)?(?:(?:height|width|length|size|diameter|side|top|bottom|surface|edge|part|section)\s+of\s+(?:the\s+|a\s+|an\s+)?)+", "", raw.strip())
    w = re.findall(r"[a-z][a-z'\-]*", re.split(_SC_LOC, s_)[0])
    while w and w[-1] in ("of", "to", "for", "and", "or"):
        w = w[:-1]
    if not w or w[-1] in _SC_GENERIC:
        return bool(re.search(SCENE, raw))
    return bool(re.search(SCENE, raw)) and bool(re.search(SCENE, w[-1]) or re.fullmatch(r"(curbs?|kerbs?|piers?|bridges?|buildings?|stairs?|steps?|ramps?|walls?)", w[-1]))
WINGED_LEN_SHORT = False
AGG3D_P = None
STATIC_FRAC = 0.25
PARSE_TAG = ""
FOCAL_MIX = ""
HALF_A = 0.4
MOGE_OPTS = {"reject", "fallback", "tdepth", "known"}
MOGE_CORR = {}


def agg3(e, dim):
    if AGG3D_P is None or len(e) == 0:
        return agg_ext(e, dim)
    return float(np.percentile(e, max(AGG3D_P, 75) if dim in ("diameter", "radius") else AGG3D_P))


def moge_depth_at(mg, k, box):
    d = mg["depth"][k]
    x0, y0, x1, y1 = [float(v) / 2.0 for v in box]
    w, h = x1 - x0, y1 - y0
    xa, xb = int(x0 + 0.3 * w), int(math.ceil(x1 - 0.3 * w))
    ya, yb = int(y0 + 0.3 * h), int(math.ceil(y1 - 0.3 * h))
    patch = d[max(ya, 0):max(yb, ya + 1), max(xa, 0):max(xb, xa + 1)].astype(float)
    patch = patch[np.isfinite(patch)]
    return float(np.median(patch)) if patch.size else None


def moge_ratio(mg, tra, trb):
    if mg is None or tra is None or trb is None:
        return None
    rs = []
    for k, j in enumerate(mg["J"]):
        if j >= len(tra.st) or j >= len(trb.st):
            continue
        sa, sb = tra.st[j], trb.st[j]
        if not sa or not sb:
            continue
        da, db = moge_depth_at(mg, k, sa["box"]), moge_depth_at(mg, k, sb["box"])
        if da and db and da > 0 and db > 0:
            rs.append(da / db)
    return float(np.median(rs)) if rs else None


class Track(M1.Track):
    def ext_px(self, dim, phrase):
        if ROT_CUR and dim == "height":
            js = self.oke
            if len(js) == 0:
                return np.array([]), js
            box = np.array([self.st[j]["box"] for j in js], float)
            return box[:, 2] - box[:, 0], js
        if WINGED_LEN_SHORT and dim == "length" and re.search(WINGED, phrase or ""):
            js = self.oke
            if len(js) == 0:
                return np.array([]), js
            rr = np.array([self.st[j]["rr"][:2] for j in js], float)
            return rr[:, 1], js
        return super().ext_px(dim, phrase)

    def __init__(self, q, vid, frames=None):
        q2 = dict(q)
        if frames is not None:
            q2["frames"] = frames
        super().__init__(q2, vid)
        self.q = q2
        self.best_gd = q.get("best_gd")
        self.best_owl = q.get("best_owl")
        self.v2 = "best_gd" in q
        self._segs = None
        self.size_px = float(np.median([max(self.st[j]["rr"][0], 1.0) for j in self.oke])) if len(self.oke) else 0.0
        bd_ = [(self.st[j] or {}).get("border", 9) for j in self.okc]
        n_unc = sum(1 for b_ in bd_ if b_ == 0)
        self.clipped_extent = bool(len(self.okc) and n_unc < max(2, 0.3 * len(self.okc)))
        if os.environ.get("QP_EDGE_FIX") == "1":
            self._edge_fix(vid)

    def _edge_fix(self, vid):
        Wv, Hv = float(vid.get("W") or 0), float(vid.get("H") or 0)
        if Wv <= 0 or Hv <= 0 or len(self.okc) == 0 or len(self.oke) < max(3, 0.2 * len(self.okc)) or len(self.oke) == len(self.okc):
            return
        bx = np.array([self.st[j]["box"] for j in self.oke], float)
        cc = np.array([self.st[j]["c"] for j in self.oke], float)
        wf, hf = float(np.median(bx[:, 2] - bx[:, 0])), float(np.median(bx[:, 3] - bx[:, 1]))
        ox = float(np.median(cc[:, 0] - (bx[:, 0] + bx[:, 2]) / 2))
        oy = float(np.median(cc[:, 1] - (bx[:, 1] + bx[:, 3]) / 2))
        uv = np.array(self.uv, float).copy()
        EFRAC = float(os.environ.get("QP_EDGE_FRAC", "0.95"))
        nfix = 0
        for k, j in enumerate(self.okc):
            x0, y0, x1, y1 = [float(v) for v in self.st[j]["box"]]
            L_, R_, T_, B_ = x0 <= 1, x1 >= Wv - 1, y0 <= 1, y1 >= Hv - 1
            if L_ != R_ and (x1 - x0) < EFRAC * wf:
                uv[k, 0] = (x1 - wf / 2 if L_ else x0 + wf / 2) + ox; nfix += 1
            if T_ != B_ and (y1 - y0) < EFRAC * hf:
                uv[k, 1] = (y1 - hf / 2 if T_ else y0 + hf / 2) + oy; nfix += 1
        if nfix:
            self.uv = uv
            self.edge_fixed = nfix

    def segs(self):
        if self._segs is None:
            self._segs = segment(self.t, self.uv, self.size_px)
        return self._segs


class Video:
    def __init__(self, vj, cfg):
        self.vj = vj
        self.W, self.H = vj["W"], vj["H"]
        if vj.get("version") == "v2":
            qs = vj["queries"] if cfg == "e" else vj["queries_d"]
            nfr = len(vj["fidx"])
            self.tr = {}
            for q in qs:
                fr = vj["objs"][str(q["obj"])] if q.get("obj") is not None else [dict(st=None, lg=-99.0)] * nfr
                self.tr[q["key"]] = Track(q, vj, fr)
        else:
            self.tr = {q["key"]: Track(q, vj) for q in vj["queries"]}
        self.colour_swaps = []
        if cfg == "e" and vj.get("version") == "v2":
            self.colour_assign(vj)
        if os.environ.get("QP_SHARED_WEAK") == "1":
            self.flag_shared()
        cam = np.array(vj["cam"], float)
        self.cam_shift = np.cumsum(cam[:, :2], 0) if len(cam) else np.zeros((1, 2))
        self.cam_mag = float(np.abs(self.cam_shift).max()) if len(cam) else 0.0

    def flag_shared(self):
        stop = {"the", "a", "an", "of", "in", "on"}
        def words(p_):
            return {w for w in re.findall(r"[a-z]+", (p_ or "").lower()) if w not in stop}
        by = {}
        for k, trk in self.tr.items():
            o = trk.q.get("obj")
            if o is None or "#second" in (trk.q.get("where") or []) or (trk.q.get("where") or []):
                continue
            by.setdefault(o, []).append(k)
        for o, ks in by.items():
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    wa, wb = words(self.tr[ks[i]].q.get("phrase")), words(self.tr[ks[j]].q.get("phrase"))
                    if wa and wb and not (wa <= wb or wb <= wa):
                        self.tr[ks[i]].shared = True
                        self.tr[ks[j]].shared = True

    def colour_assign(self, vj):
        import itertools
        groups = {}
        for k, trk in self.tr.items():
            ph = trk.q.get("phrase") or ""
            cw = [c for c in tc.COLOR_WORDS if re.search(r"\b" + c + r"\b", ph)]
            if len(cw) != 1 or trk.q.get("obj") is None:
                continue
            groups.setdefault(tc.head_noun(ph), []).append((k, cw[0], trk.q["obj"]))
        for h, mem in groups.items():
            objs = sorted(set(o for _, _, o in mem))
            if len(mem) < 2 or len(objs) < len(mem) or len(mem) > 6:
                continue
            col = {}
            for o in objs:
                c = next((f["col"] for f in vj["objs"][str(o)] if f.get("col")), None)
                if c is None:
                    break
                col[o] = c
            if len(col) < len(objs):
                continue
            cur = [o for _, _, o in mem]
            score = lambda assign: sum(col[o][c] for (_, c, _), o in zip(mem, assign))
            best = max(itertools.permutations(objs, len(mem)), key=score)
            if list(best) != cur and score(best) >= score(cur) + 0.2:
                trks = {o: next(self.tr[k] for k, _, oo in mem if oo == o) for o in cur}
                newtr = {}
                for (k, c, _), o in zip(mem, best):
                    src = trks.get(o)
                    if src is None:
                        break
                    newtr[k] = src
                else:
                    self.tr.update(newtr)
                    self.colour_swaps.append((h, [k for k, _, _ in mem], cur, list(best)))

    @staticmethod
    def key(ent):
        if not ent or not ent.get("phrase"):
            return None
        rel = ent.get("rel")
        return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")

    def get(self, ent):
        k = self.key(ent)
        return self.tr.get(k) if k else None


def noise_sigma(t, P, floor=0.5):
    if len(t) < 5:
        return floor
    dt1, dt2 = t[1:-1] - t[:-2], t[2:] - t[1:-1]
    ok = np.abs(dt1 - dt2) < 0.25 * np.minimum(dt1, dt2) + 1e-9
    d2 = (P[2:] - 2 * P[1:-1] + P[:-2])[ok]
    if len(d2) < 3:
        return floor
    s = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / math.sqrt(6)
    return float(max(np.mean(s), floor))


def segment(t, P, size_px=0.0, min_pts=5):
    N = len(t)
    sig = noise_sigma(t, P)
    if N < 2 * min_pts:
        segs = [(0, N)] if N else []
    else:
        C = np.full((N + 1, N + 1), np.inf)
        for i in range(N - min_pts + 1):
            tt = t[i:] - t[i]
            sc = max(tt[-1], 1e-6)
            tt = tt / sc
            S = [np.cumsum(tt ** k) for k in range(5)]
            A = np.stack([np.stack([S[0], S[1], S[2]], -1), np.stack([S[1], S[2], S[3]], -1), np.stack([S[2], S[3], S[4]], -1)], -2)
            n = np.arange(1, len(tt) + 1)
            v = n >= min_pts
            tot = np.zeros(v.sum())
            for k in range(P.shape[1]):
                y = P[i:, k]
                b = np.stack([np.cumsum(y), np.cumsum(y * tt), np.cumsum(y * tt * tt)], -1)
                yy = np.cumsum(y * y)
                coef = np.linalg.solve(A[v] + 1e-12 * np.eye(3), b[v][..., None])[..., 0]
                tot += np.clip(yy[v] - (coef * b[v]).sum(-1), 0, None)
            C[i, i + n[v]] = tot
        beta = 7.0 * sig * sig * math.log(N)
        Fv = np.full(N + 1, np.inf); Fv[0] = 0.0
        back = np.zeros(N + 1, int)
        for j in range(min_pts, N + 1):
            cand = Fv[:j - min_pts + 1] + C[:j - min_pts + 1, j] + beta
            i = int(np.argmin(cand)); Fv[j] = cand[i]; back[j] = i
        segs = []
        j = N
        while j > 0:
            i = back[j]; segs.append((i, j)); j = i
        segs = segs[::-1]

    def is_rest(i, j):
        Q = P[i:j]
        spread = float(np.max(np.linalg.norm(Q - Q.mean(0), axis=1))) if j > i else 0.0
        return spread < max(2 * sig, 0.03 * size_px, 1.0)

    def vel_end(i, j, tb):
        return np.array([np.polyval(np.polyder(np.polyfit(t[i:j] - tb, P[i:j, k], 2)), 0.0) for k in range(P.shape[1])])
    dt = float(np.median(np.diff(t))) if N > 1 else 1.0
    k = 0
    while k < len(segs) - 1:
        (i, m), (m2, j) = segs[k], segs[k + 1]
        tb = 0.5 * (t[m - 1] + t[m])
        keep = is_rest(i, m) != is_rest(m2, j)
        if not keep and not (is_rest(i, m) and is_rest(m2, j)):
            vl, vr = vel_end(i, m, tb), vel_end(m2, j, tb)
            jump = float(np.linalg.norm(vl - vr))
            keep = jump > max(0.3 * max(np.linalg.norm(vl), np.linalg.norm(vr)), 4 * sig / dt)
        if keep:
            k += 1
        else:
            segs[k:k + 2] = [(i, j)]
    rest = [is_rest(i, j) for i, j in segs]
    return dict(segs=segs, rest=rest, sigma=sig)


def seg_index(sg, t, t0):
    best, bd = None, 1e9
    for k, (i, j) in enumerate(sg["segs"]):
        if t[i] - 1e-9 <= t0 <= t[j - 1] + 1e-9:
            return k
        d = min(abs(t0 - t[i]), abs(t0 - t[j - 1]))
        if d < bd:
            best, bd = k, d
    return best


def _sub(sg, t, P, t0):
    k = seg_index(sg, t, t0)
    if k is None:
        return None
    i, j = sg["segs"][k]
    return t[i:j], P[i:j], k


def vel_at_c(t, P, t0, sg, half=None):
    if half is None:
        half = HALF_V
    s = _sub(sg, t, P, t0)
    if s is None:
        return vel_at(t, P, t0, half)
    ts, Ps, _ = s
    t0c = min(max(t0, ts.min()), ts.max())
    r = _fit(ts, Ps, t0c, half, 2, 5) or _fit(ts, Ps, t0c, half, 1, 3)
    if r is None:
        return vel_at(t, P, t0, half)
    cs, deg = r
    if t0c != t0 and deg >= 1:
        return np.array([np.polyval(np.polyder(c), t0 - t0c) for c in cs])
    return np.array([c[-2] for c in cs])


def acc_at_c(t, P, t0, sg, half=None):
    if half is None:
        half = HALF_A
    s = _sub(sg, t, P, t0)
    if s is None:
        return acc_at(t, P, t0, half)
    ts, Ps, _ = s
    t0c = min(max(t0, ts.min()), ts.max())
    r = _fit(ts, Ps, t0c, half, 2, 7)
    if r is not None and r[1] == 2:
        return np.array([2 * c[-3] for c in r[0]])
    if len(ts) >= 5:
        return np.array([2 * np.polyfit(ts - t0c, Ps[:, k], 2)[0] for k in range(Ps.shape[1])])
    return None


def pos_at_c(t, P, t0, sg):
    s = _sub(sg, t, P, t0)
    if s is None:
        return pos_at(t, P, t0)
    ts, Ps, _ = s
    r = _fit(ts, Ps, min(max(t0, ts.min()), ts.max()), 0.08, 1, 2)
    if r is None:
        return pos_at(t, P, t0)
    t0c = min(max(t0, ts.min()), ts.max())
    return np.array([np.polyval(c, t0 - t0c) for c in r[0]])


def avg_speed_c(t, P, a, b, sg):
    m = np.where((t >= a - 1e-9) & (t <= b + 1e-9))[0]
    if len(m) < 3:
        return None
    vs = [vel_at_c(t, P, t[i], sg, 0.15) for i in m]
    vs = [np.linalg.norm(v) for v in vs if v is not None]
    return float(np.mean(vs)) if vs else None


def acc_interval_c(t, P, a, b, sg):
    ks = sorted(set(k for k, (i, j) in enumerate(sg["segs"]) if t[j - 1] >= a - 1e-9 and t[i] <= b + 1e-9))
    if not ks:
        return None
    if len(ks) == 1:
        i, j = sg["segs"][ks[0]]
        m = (t[i:j] >= a - 1e-9) & (t[i:j] <= b + 1e-9)
        ts, Ps = t[i:j][m], P[i:j][m]
        if len(ts) < 6:
            ts, Ps = t[i:j], P[i:j]
        if len(ts) < 5:
            return None
        return np.array([2 * np.polyfit(ts, Ps[:, k], 2)[0] for k in range(P.shape[1])])
    va, vb = vel_at_c(t, P, a, sg), vel_at_c(t, P, b, sg)
    if va is None or vb is None or b - a < 0.05:
        return None
    return (vb - va) / (b - a)


def acc_whole_c(t, P, sg):
    best, bl = None, 0.0
    for (i, j), rest in zip(sg["segs"], sg["rest"]):
        if rest or j - i < 5:
            continue
        L = t[j - 1] - t[i]
        if L > bl:
            best, bl = (i, j), L
    if best is None:
        return None
    i, j = best
    return np.array([2 * np.polyfit(t[i:j], P[i:j, k], 2)[0] for k in range(P.shape[1])])


def robust_quad(tt, y, sig):
    c = np.polyfit(tt, y, 2)
    r = y - np.polyval(c, tt)
    s = max(1.4826 * np.median(np.abs(r - np.median(r))), sig, 1e-6 * (np.ptp(y) + 1e-9))
    keep = np.abs(r) <= 3 * s
    if keep.sum() < 6:
        keep = np.ones(len(y), bool)
    elif keep.sum() < len(y):
        c = np.polyfit(tt[keep], y[keep], 2)
    X = np.stack([tt ** 2, tt, np.ones_like(tt)], 1)[keep]
    res = y[keep] - X @ c
    dof = max(int(keep.sum()) - 3, 1)
    s_r = math.sqrt(max(float(res @ res) / dof, 1e-12))
    try:
        se2 = s_r * math.sqrt(np.linalg.inv(X.T @ X)[0, 0])
    except np.linalg.LinAlgError:
        se2 = np.inf
    return c, se2, s_r, keep


def freefall(tr, P=None, a=None, b=None):
    if tr is None or tr.n < 6:
        return None, "few_frames"
    t = tr.t
    Q = tr.uv if P is None else P
    sg = tr.segs()
    sig = sg["sigma"] if P is None else 0.0
    ff_short = [float(x) for x in os.environ.get("QP_FF_SHORT", "").split(":")] if os.environ.get("QP_FF_SHORT") else None
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
        cy, se, s_r, keep = robust_quad(tt, Q[idx, 1], sig)
        ay = 2 * cy[0]
        if not (ay > 0 and cy[0] / se >= float(os.environ.get("QP_FF_SIG", 5))):
            return None
        if short and not (cy[0] / se >= ff_short[0] and s_r <= ff_short[1]):
            return None
        ah = [2 * np.polyfit(tt[keep], Q[idx, k][keep], 2)[0] for k in range(Q.shape[1]) if k != 1]
        if math.sqrt(sum(x * x for x in ah)) > float(os.environ.get("QP_FF_H", 0.35)) * ay:
            return None
        lim = 3 * max(sig, 0.5, ff_sj * getattr(tr, "size_px", 0.0)) if ff_sj > 0 else 3 * max(sig, 0.5)
        if P is None and s_r > lim:
            return None
        if n >= 12:
            h = n // 2
            a1 = 2 * np.polyfit(tt[:h], Q[idx[:h], 1], 2)[0]
            a2 = 2 * np.polyfit(tt[h:], Q[idx[h:], 1], 2)[0]
            if not (0.5 * ay <= a1 <= 1.5 * ay and 0.5 * ay <= a2 <= 1.5 * ay):
                return None
        return ay, n, dur

    acc = []
    pieces = []
    sub_ = False
    for (i, j), rest in zip(sg["segs"], sg["rest"]):
        idx = np.arange(i, j)
        if a is not None and b is not None:
            idx = idx[(t[idx] >= a - 0.05) & (t[idx] <= b + 0.05)]
        if len(idx) < 6 or rest:
            continue
        pieces.append(idx)
        r_ = ff_test(idx)
        if r_ is not None:
            acc.append(r_)
    sub_src_ = os.environ.get("QP_FF_SUB_SRC", "")
    if not acc and os.environ.get("QP_FF_SUB") and (not sub_src_ or (str(CUR_VID.get("src", "")).startswith(sub_src_) and CUR_VID.get("is2d"))):
        best_ = None
        for idx in pieces:
            n = len(idx); st_ = max(1, n // 30)
            for i0 in range(0, n - 5, st_):
                for i1 in range(n, i0 + 5, -st_):
                    if best_ is not None and (t[idx[i1 - 1]] - t[idx[i0]]) <= best_[2]:
                        break
                    rr = ff_test(idx[i0:i1])
                    if rr is not None:
                        best_ = rr
                        break
        if best_ is not None:
            acc.append(best_)
            sub_ = True
    if not acc:
        return None, "no_freefall_segment"
    vals = np.array([x[0] for x in acc]); w = np.array([x[1] for x in acc], float)
    med = float(np.median(vals))
    ok = (vals >= med / 1.5) & (vals <= med * 1.5)
    if ok.sum() < max(1, 0.5 * len(acc)):
        return None, "freefall_segments_disagree"
    o = np.argsort(vals[ok]); cw = np.cumsum(w[ok][o])
    g = float(vals[ok][o][np.searchsorted(cw, cw[-1] / 2)])
    return g, dict(n_seg=int(ok.sum()), dur=float(sum(x[2] for x, k in zip(acc, ok) if k)), sub=sub_)


CLASS_RANGE = [(r"^(?:person|human|man|woman|boy|girl|player|people|pedestrian|walker|lady|kid|child|runner)$", (0.5, 2.5)),
               (r"ball$", (0.01, 1.0)), (r"^(?:car|truck|van|bus|suv|taxi)$", (1.5, 12.0))]


def gravity_scale_check(mpp, fall_tr, V):
    fh = V.H * mpp
    if not (0.05 <= fh <= 50.0):
        return False, f"grav_frameH({fh:.3g}m)"
    if fall_tr is not None and fall_tr.n:
        s = fall_tr.size_px * mpp
        if not (1e-3 <= s <= 5.0):
            return False, f"grav_objsize({s:.3g}m)"
    bad, n = [], 0
    for k, trk in V.tr.items():
        if trk.n < 3 or trk.best < 0.3 or not len(trk.oke):
            continue
        h = tc.head_noun(trk.q.get("phrase"))
        for pat, (lo, hi) in CLASS_RANGE:
            if re.search(pat, h):
                n += 1
                s = trk.size_px * mpp
                if not (lo <= s <= hi):
                    bad.append(f"{h}={s:.3g}m")
                break
    if n and len(bad) > n / 2:
        return False, "grav_class(" + ",".join(bad[:3]) + ")"
    return True, f"frameH={fh:.3g}m"


def track_ok(tr, dynamic, depth_moves=False):
    if tr is None or tr.n == 0:
        return False, "no_track"
    if "e" in FIX and tr.v2:
        det_ok = (tr.best_gd or 0) >= float(os.environ.get("QP_GD_HI", 0.3)) or ((tr.best_gd or 0) >= float(os.environ.get("QP_GD_LO", 0.15)) and (tr.best_owl or 0) >= T_OWL * float(os.environ.get("QP_OWL_F", 1.0)))
        if not det_ok:
            return False, f"low_det({tr.best_gd or 0:.2f}/owl{tr.best_owl or 0:.2f})"
    elif tr.best < 0.3:
        return False, f"low_det({tr.best:.2f})"
    if tr.iou and tr.iou < 0.5:
        return False, f"det_track_iou({tr.iou:.2f})"
    if tr.n < (6 if dynamic else 3):
        return False, f"few_frames({tr.n})"
    if dynamic and not depth_moves:
        size = np.median([max(tr.st[j]["rr"][0], 1.0) for j in tr.okc])
        rng = float(np.linalg.norm(tr.uv.max(0) - tr.uv.min(0)))
        thr = STATIC_FRAC * size
        if os.environ.get("QP_STATIC_CAP") and rng < thr:
            st_ = np.linalg.norm(np.diff(tr.uv, axis=0), axis=1) if tr.n >= 2 else np.zeros(1)
            if rng >= float(os.environ["QP_STATIC_CAP"]) and st_.max() <= 0.3 * rng:
                thr = float(os.environ["QP_STATIC_CAP"])
        if rng < thr:
            return False, f"static_track({rng:.0f}px<{thr:.0f})"
    return True, ""


def contrastive_row(r, tgt_ent, pr_ent):
    tph = (tgt_ent["phrase"] or "")
    pph = (pr_ent.get("phrase") or "")
    if "d" in FIX:
        return tc.contrastive_v2(tph, tgt_ent.get("where"), pph, pr_ent.get("where"))
    a = tph + " " + " ".join(tgt_ent.get("where") or [])
    b = pph + " " + " ".join(pr_ent.get("where") or [])
    return M1.contrastive(a, b, (re.findall(r"[a-z]+", tgt_ent["phrase"] or "")[-1:], re.findall(r"[a-z]+", pr_ent["phrase"] or "")[-1:]))


def px_quantity(tr, kind, t=None, t1=None, t2=None, when=None, P=None):
    tt = tr.t
    PP = tr.uv if P is None else P
    c = "c" in FIX
    sg = tr.segs() if c else None
    lo, hi = (tt.min(), tt.max()) if tr.n else (0, 0)
    a = lo if t1 is None else max(t1, lo)
    b = hi if t2 is None else min(t2, hi)
    V_ = (lambda t0, h=None: vel_at_c(tt, PP, t0, sg, HALF_V if h is None else h)) if c else (lambda t0, h=None: vel_at(tt, PP, t0, HALF_V if h is None else h))
    POS = (lambda t0: pos_at_c(tt, PP, t0, sg)) if c else (lambda t0: pos_at(tt, PP, t0))
    AVG = (lambda a_, b_: avg_speed_c(tt, PP, a_, b_, sg)) if c else (lambda a_, b_: avg_speed(tt, PP, a_, b_))
    if kind == "speed":
        if t is None:
            return AVG(lo, hi)
        t0 = clamp_time(tr, t)
        if t0 is None:
            return None
        v = V_(t0)
        return None if v is None else float(np.linalg.norm(v))
    if kind == "avgspeed":
        if b - a < 0.1:
            return None
        return AVG(a, b)
    if kind == "hvel":
        pa, pb = POS(a), POS(b)
        if pa is None or pb is None or b - a < 0.1:
            return None
        return abs(pb[0] - pa[0]) / (b - a)
    if kind == "accel":
        if t is not None:
            t0 = clamp_time(tr, t)
            if t0 is None:
                return None
            ac = acc_at_c(tt, PP, t0, sg) if c else acc_at(tt, PP, t0)
        elif t1 is not None:
            ac = acc_interval_c(tt, PP, a, b, sg) if c else acc_interval(tt, PP, a, b)
        else:
            ac = acc_whole_c(tt, PP, sg) if c else acc_interval(tt, PP, lo, hi)
        return None if ac is None else float(np.linalg.norm(ac))
    if kind == "disp":
        if t1 is None:
            a, b = lo, hi
        ca, cb = clamp_time(tr, a), clamp_time(tr, b)
        if ca is None or cb is None:
            return None
        pa, pb = POS(ca), POS(cb)
        return None if pa is None or pb is None else float(np.linalg.norm(pb - pa))
    if kind == "path":
        s = AVG(a, b)
        return None if s is None else s * (b - a)
    if kind == "maxheight":
        return float(PP[:, 1].max() - PP[:, 1].min()) if tr.n >= 3 else None
    if kind == "orbitdiam":
        if tr.n < 5:
            return None
        d = np.linalg.norm(PP[:, None, :] - PP[None, :, :], axis=2)
        return float(d.max())
    return None


PRIOR_DRIFT = float(os.environ.get("QP_PRIOR_DRIFT", "0") or 0)


def drift_factor(tr):
    t, P = tr.t, tr.uv
    if PRIOR_DRIFT <= 0 or len(t) < 8:
        return 1.0
    T = float(t[-1] - t[0])
    s = avg_speed_c(t, P, t[0], t[-1], tr.segs()) if "c" in FIX else avg_speed(t, P, t[0], t[-1])
    lin = float(np.hypot(np.polyfit(t, P[:, 0], 1)[0], np.polyfit(t, P[:, 1], 1)[0]))
    if s is None or lin <= 0 or T < 0.5:
        return 1.0
    if s / lin > PRIOR_DRIFT and lin * T >= max(20.0, 0.3 * tr.size_px):
        DRIFT_LOG.append((tr.q.get("key"), round(s, 2), round(lin, 2)))
        return lin / s
    return 1.0


DRIFT_LOG = []
PRIOR_ACCGUARD = float(os.environ.get("QP_PRIOR_ACCGUARD", "0") or 0)


def acc_whole_factor(tr, pr):
    t, P = tr.t, tr.uv
    if PRIOR_ACCGUARD <= 0 or len(t) < 12 or pr.get("t") is None or pr.get("t1") is not None or re.search(r"\b(?:after|before)\b", pr.get("raw", "")):
        return 1.0
    a_loc = px_quantity(tr, "accel", t=pr.get("t"))
    if a_loc is None or a_loc <= 0:
        return 1.0
    tc = t - t.mean()
    cs = [np.polyfit(tc, P[:, k], 2) for k in range(2)]
    a_w = 2 * float(np.hypot(cs[0][0], cs[1][0]))
    ss_res = sum(float(np.sum((P[:, k] - np.polyval(cs[k], tc)) ** 2)) for k in range(2))
    ss_tot = sum(float(np.sum((P[:, k] - P[:, k].mean()) ** 2)) for k in range(2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    span = float(np.linalg.norm(P[-1] - P[0]))
    if a_w > 0 and r2 >= 0.98 and span >= max(20.0, 0.3 * tr.size_px) and a_loc / a_w > PRIOR_ACCGUARD:
        DRIFT_LOG.append(("acc", tr.q.get("key"), round(a_loc, 1), round(a_w, 1), round(r2, 4)))
        return a_w / a_loc
    return 1.0


def prior_kin(tr, pr, P=None):
    if "c" not in FIX:
        kd = "speed" if pr["kind"] == "speed" else "accel"
        return px_quantity(tr, kd, t=pr.get("t"), t1=pr.get("t1"), t2=pr.get("t2"), P=P)
    if pr["kind"] == "speed":
        if pr.get("t1") is None:
            s = px_quantity(tr, "speed", t=pr.get("t"), t1=pr.get("t1"), t2=pr.get("t2"), P=P)
            if s is not None and pr.get("t") is None and PRIOR_DRIFT > 0:
                s = s * drift_factor(tr)
            return s
        return px_quantity(tr, "avgspeed", t1=pr.get("t1"), t2=pr.get("t2"), P=P)
    if "c" in FIX and pr.get("t") is not None and pr.get("t1") is None:
        t0 = pr["t"] + (0.1 if re.search(r"\bafter\b", pr.get("raw", "")) else 0.0)
        a_ = px_quantity(tr, "accel", t=t0, P=P)
    else:
        a_ = px_quantity(tr, "accel", t=pr.get("t"), t1=pr.get("t1"), t2=pr.get("t2"), P=P)
    if a_ is not None and PRIOR_ACCGUARD > 0 and not pr.get("gravity"):
        a_ = a_ * acc_whole_factor(tr, pr)
    return a_


def _poly_dist(A, B):
    import cv2
    best = (np.inf, None, None)
    for pa in A:
        a = np.asarray(pa, float)
        for pb in B:
            b = np.asarray(pb, float)
            if len(a) >= 3 and len(b) >= 3:
                if any(cv2.pointPolygonTest(b.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in a[::max(1, len(a) // 20)]) or \
                   any(cv2.pointPolygonTest(a.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in b[::max(1, len(b) // 20)]):
                    return 0.0, None, None
            for P_, S_ in ((a, b), (b, a)):
                s0, s1 = S_, np.roll(S_, -1, 0)
                d = s1 - s0
                L = np.maximum((d * d).sum(1), 1e-9)
                u = np.clip(((P_[:, None, :] - s0[None]) * d[None]).sum(-1) / L[None], 0, 1)
                proj = s0[None] + u[..., None] * d[None]
                dist = np.linalg.norm(P_[:, None, :] - proj, axis=2)
                k = np.unravel_index(np.argmin(dist), dist.shape)
                if dist[k] < best[0]:
                    pp, qq = P_[k[0]], proj[k]
                    best = (float(dist[k]), (pp if S_ is b else qq), (qq if S_ is b else pp))
    return best


def dist2_px(tr1, tr2, t, when, P1=None, P2=None, D1=None, D2=None, f=None, cx=0, cy=0, minimum=False):
    v = M1.dist2_px(tr1, tr2, t, when, P1, P2)
    if "f" not in FIX or v is None:
        return v, False
    fr1, fr2 = tr1.q.get("frames"), tr2.q.get("frames")
    if not fr1 or not fr2:
        return v, False
    T = tr1.t_all
    both = [j for j in range(len(T)) if fr1[j].get("poly") and fr2[j].get("poly") and fr1[j].get("st") and fr2[j].get("st")]
    if not both:
        return v, False
    if t is not None or when is not None:
        if when == "start":
            t = max(tr1.t.min(), tr2.t.min())
        elif when == "end":
            t = min(tr1.t.max(), tr2.t.max())
        js = [min(both, key=lambda j: abs(T[j] - t))]
        if abs(T[js[0]] - t) > 0.25:
            return v, False
    else:
        js = both
    gaps, cens = [], []
    for j in js:
        g, pa, pb = _poly_dist(fr1[j]["poly"], fr2[j]["poly"])
        c1, c2 = np.array(fr1[j]["st"]["c"]), np.array(fr2[j]["st"]["c"])
        if D1 is not None and f is not None:
            z1, z2 = float(D1(T[j])[0]), float(D2(T[j])[0])
            def bp(u, z):
                r = np.array([(u[0] - cx) / f, (u[1] - cy) / f, 1.0]); return r / np.linalg.norm(r) * z
            cen = float(np.linalg.norm(bp(c1, z1) - bp(c2, z2)))
            gg = float(np.linalg.norm(bp(pa, z1) - bp(pb, z2))) if pa is not None else 0.0
        else:
            cen = float(np.linalg.norm(c1 - c2)); gg = g
        gaps.append(gg); cens.append(cen)
    g, c = float(np.median(gaps)), float(np.median(cens))
    if g > 0 and (g < 0.5 * c or minimum):
        return v * g / c if c > 0 else v, True
    return v, False


def _rot_uv(uv, d):
    return np.stack([-uv[:, 1], uv[:, 0]], 1) if d == "+x" else np.stack([uv[:, 1], -uv[:, 0]], 1)


def _bottom_up(box):
    return box[2] if ROT_CUR == "+x" else (-box[0] if ROT_CUR == "-x" else box[3])


def floor_level(V):
    lim = V.W if ROT_CUR == "+x" else (0.0 if ROT_CUR == "-x" else V.H)
    bs = []
    for trk in list({id(x): x for x in V.tr.values() if x is not None}.values()):
        for j in trk.oke:
            b = trk.st[j]["box"]
            edge = b[2] if ROT_CUR == "+x" else (b[0] if ROT_CUR == "-x" else b[3])
            if abs(edge - lim) <= 2:
                continue
            bs.append(_bottom_up(b))
    return float(np.percentile(bs, 99)) if len(bs) >= 5 else None


def hfloor_px(tgt, V, t):
    fl = floor_level(V)
    if fl is None or tgt is None or len(tgt.oke) == 0:
        return None
    js = list(tgt.oke)
    if t is not None:
        tt = np.array([tgt.t_all[j] for j in js])
        k = int(np.argmin(np.abs(tt - t)))
        if abs(tt[k] - t) > 0.12:
            return None
        bt = _bottom_up(tgt.st[js[k]]["box"])
    else:
        bt = float(np.median([_bottom_up(tgt.st[j]["box"]) for j in js]))
    return fl - bt


def track_depth_fn_v3(depth, keys, trk):
    mode = os.environ.get("QP_SD_MODE", "")
    base = M1.depth_fn(depth, keys)
    if base is None or trk is None or trk.n < 5:
        return base
    ent = [d for d in depth if d["key"] in keys]
    timed = [d for d in ent if d["t"] is not None]
    per_key_timed = max([sum(1 for d in timed if d["key"] == k) for k in keys] + [0])
    if per_key_timed >= 2 or mode == "off":
        return base
    s = np.sqrt(np.array([trk.st[j]["area"] for j in trk.okc], float))
    tt = trk.t.copy()
    if len(s) >= 5:
        s = np.array([np.median(s[max(0, i - 2):i + 3]) for i in range(len(s))])
    raw_ratio = float(np.percentile(s, 90) / max(np.percentile(s, 10), 1e-6))
    if mode == "robust" and len(s) >= 8 and len(s) == len(tt):
        y = np.log(np.maximum(s, 1e-6)); w = np.ones(len(y))
        c = np.polyfit(tt, y, 2)
        for _ in range(6):
            r_ = y - np.polyval(c, tt)
            sig = 1.4826 * float(np.median(np.abs(r_[w > 0]))) + 1e-3
            w_new = (np.abs(r_) <= 2.5 * sig).astype(float)
            if w_new.sum() < 6 or np.array_equal(w_new, w):
                break
            w = w_new
            c = np.polyfit(tt[w > 0], y[w > 0], 2)
        s = np.exp(np.polyval(c, tt))
    lo, hi = np.percentile(s, 10), np.percentile(s, 90)
    SD_LOG.append((trk.q.get("key"), round(raw_ratio, 3), round(float(hi / max(lo, 1e-6)), 3)))
    if hi / max(lo, 1e-6) < 1.15:
        return base
    if timed:
        t0 = timed[0]["t"]
        s0 = float(np.interp(t0, tt, s))
        D0 = float(np.mean([d["d"] for d in timed]))
    else:
        s0 = float(np.median(s)); D0 = float(np.mean([d["d"] for d in ent]))
    Dtr = np.clip(D0 * s0 / s, D0 / 3, D0 * 3)
    return lambda t: np.interp(np.atleast_1d(np.asarray(t, float)), tt, Dtr)


SD_LOG = []


def second_instance(V, vj, tgt):
    T_ = tgt.t_all
    base_ = {j: np.array(tgt.st[j]["c"], float) for j in range(len(T_)) if tgt.st[j]}
    if len(base_) < 3:
        return None
    a0_ = float(np.median([tgt.st[j]["area"] for j in base_]))
    best_, bn_ = None, 0
    for oid_, fr_ in vj["objs"].items():
        if len(fr_) != len(T_):
            continue
        ok_ = [j for j in range(len(fr_)) if fr_[j] and fr_[j].get("st")]
        if len(ok_) < 0.5 * len(fr_):
            continue
        com_ = [j for j in ok_ if j in base_]
        if len(com_) < 3:
            continue
        dd_ = float(np.median([np.linalg.norm(np.array(fr_[j]["st"]["c"], float) - base_[j]) for j in com_]))
        a1_ = float(np.median([fr_[j]["st"]["area"] for j in ok_]))
        if dd_ <= max(20.0, 1.5 * tgt.size_px) or not (0.25 <= a1_ / max(a0_, 1.0) <= 4.0):
            continue
        if len(ok_) > bn_:
            best_, bn_ = oid_, len(ok_)
    if best_ is None:
        return None
    q_ = dict(tgt.q, obj=int(best_), key=str(tgt.q.get("key")) + "#alt", anchor=None)
    q_.pop("frames", None)
    return Track(q_, vj, vj["objs"][best_])


def nearest_cands(V, r, tgt2):
    if os.environ.get("QP_NEAREST") != "1" or tgt2 is None:
        return [tgt2]
    m_ = re.search(r"\b(?:nearest|closest)\s+([a-z]+)", r["question"].lower())
    if not m_:
        return [tgt2]
    head = re.sub(r"(?:es|s)$", "", m_.group(1))
    out = {id(tgt2): tgt2}
    for k, trk in V.tr.items():
        ph = str(k).split("|")[0].strip().lower()
        if trk is not None and trk.n and re.search(r"\b" + re.escape(head) + r"(?:s|es)?\b", ph):
            out[id(trk)] = trk
    return list(out.values())


def measure_video(recs, vj, fdef, cfg):
    global ROT_CUR
    ROT_CUR = None
    CUR_VID.clear()
    if recs:
        CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=recs[0]["video_type"][1] == "2")
    V = Video(vj, cfg)
    if (os.environ.get("QP_FF_ROT") == "1" and recs and str(recs[0].get("video_source", "")).startswith("internet")
            and recs[0]["video_type"][1] == "2" and any((r["priors"] or [{}])[0].get("gravity") for r in recs)):
        trs = list({id(t_): t_ for t_ in V.tr.values() if t_ is not None and t_.n >= 6}.values())
        if not any(freefall(t_)[0] is not None for t_ in trs):
            for d_ in ("+x", "-x"):
                hit_ = []
                for t_ in trs:
                    u0_ = t_.uv
                    t_.uv = _rot_uv(u0_, d_)
                    hit_.append(freefall(t_)[0] is not None)
                    t_.uv = u0_
                if any(hit_):
                    ROT_CUR = d_
                    for t_ in list({id(x_): x_ for x_ in V.tr.values() if x_ is not None and getattr(x_, "uv", None) is not None}.values()):
                        t_.uv = _rot_uv(np.asarray(t_.uv, float).reshape(-1, 2), d_); t_._segs = None
                    print('', flush=True)
                    break
    cx, cy = V.W / 2.0, V.H / 2.0
    mg = MOGE.get(vj["video_id"])
    dg = DA3D.get(vj["video_id"])
    f_moge = mg["f"] if (mg is not None and np.isfinite(mg["f"]) and mg["f"] > 0) else None
    if f_moge is not None and recs:
        f_moge *= MOGE_CORR.get(recs[0]["video_source"].replace("segmentaion", "segmentation"), 1.0)
        f_moge *= V.W / float(mg.get("W", V.W))
    out = []
    calib_cache = {}
    qkeys = {}
    for r in recs:
        for ent, dk in ((r["target"], r.get("target_dkeys")), (r.get("target2"), r.get("target2_dkeys")), (r["prior_obj"], r.get("prior_dkeys"))):
            trk = V.get(ent) if ent else None
            if trk is not None and dk:
                qkeys.setdefault(id(trk), dk)
    gcache = {}
    dep_all = recs[0].get("depth") or [] if recs else []

    def ground_fit(f):
        key = round(float(f), 1)
        if key in gcache:
            return gcache[key]
        obs, ntr = [], set()
        for trk in V.tr.values():
            keys = qkeys.get(id(trk))
            if not keys or trk is None or trk.n == 0:
                continue
            for e in dep_all:
                if e["key"] not in keys:
                    continue
                js = list(trk.okc) if e["t"] is None else [j for j in trk.okc if abs(trk.t_all[j] - e["t"]) <= 0.06]
                for j in js[::max(1, len(js) // 8)]:
                    b = trk.st[j]["box"]
                    if b[3] >= V.H - 2:
                        continue
                    u_, v_ = trk.st[j]["c"]
                    Z_ = e["d"] / math.sqrt(1 + ((u_ - cx) ** 2 + (v_ - cy) ** 2) / f ** 2)
                    obs.append((float(b[3]), Z_)); ntr.add(id(trk))
        out_ = None
        if len(ntr) >= 2 and len(obs) >= 4:
            Y_ = np.array([o[0] for o in obs]); Zs_ = np.array([o[1] for o in obs])
            X_ = np.stack([np.ones_like(Zs_), f / Zs_], 1)
            coef_, *_ = np.linalg.lstsq(X_, Y_, rcond=None)
            yh_, h_ = float(coef_[0]), float(coef_[1])
            if h_ > 0.05 and np.median(np.abs(Y_ - X_ @ coef_)) <= 0.03 * V.H:
                out_ = (yh_, h_)
        gcache[key] = out_
        return out_

    for r in recs:
        res = dict(id=r["id"], kind=r["q"]["kind"], ok=False, why="", val_si=np.nan, mpp=np.nan, f=np.nan, mode="", grav="")
        q = r["q"]
        kind = q["kind"]
        is3d = r["video_type"][1] == "3"
        pr = r["priors"][0] if r["priors"] else None
        tgt = V.get(r["target"])
        tgt2 = V.get(r["target2"]) if r.get("target2") else None
        if (os.environ.get("QP_SECOND_FIX") == "1" and tgt is not None and tgt2 is not None and q.get("kind") == "dist2"
                and str(vj.get("video_id", "")).startswith("simulation")
                and (r.get("target2") or {}).get("where") == ["#second"] and vj.get("objs")
                and (tgt2 is tgt or M1.same_object(tgt, tgt2)
                     or (tgt.q.get("obj") is not None and tgt.q.get("obj") == tgt2.q.get("obj")))):
            alt_ = second_instance(V, vj, tgt)
            if alt_ is not None:
                tgt2 = alt_; res["grav"] = "second_fix"
        prior_tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
        if os.environ.get("QP_SA2VA_WEAK") == "1" and any(t_ is not None and (getattr(t_, "q", None) or {}).get("sa2va_weak") for t_ in (tgt, tgt2, prior_tr)):
            res["sa2va_weak"] = True
        if os.environ.get("QP_SHARED_WEAK") == "1" and any(t_ is not None and getattr(t_, "shared", False) for t_ in (tgt, tgt2, prior_tr)):
            res["shared"] = True
        if os.environ.get("QP_CLIPPED_WEAK") == "1" and (
                (kind == "size" and tgt is not None and getattr(tgt, "clipped_extent", False))
                or (pr is not None and pr.get("kind") == "size" and prior_tr is not None and getattr(prior_tr, "clipped_extent", False))):
            res["clipped"] = True
        if kind == "other" or not q.get("target"):
            res["why"] = "unparsed_or_other"; out.append(res); continue
        scene_t = bool(scene_phrase(r["target"]["raw"]) or (r.get("target2") and scene_phrase(r["target2"].get("raw"))))
        sa_scene = bool(os.environ.get("QP_SCENE_SA2VA") == "1" and tgt is not None and (getattr(tgt, "q", None) or {}).get("sa2va")
                        and (not r.get("target2") or (tgt2 is not None and (getattr(tgt2, "q", None) or {}).get("sa2va"))))
        if scene_t and not MEASURE_SCENE and not sa_scene:
            res["why"] = "scene_target"; out.append(res); continue
        if scene_t and sa_scene:
            res["sa2va_scene"] = True
        if (re.search(PART, re.sub(r"\([^)]*\)", " ", r["question"].lower())) or q.get("part")) and kind == "size" and r.get("side_fix") != "rim":
            res["why"] = "part_target"; out.append(res); continue
        if pr is None or pr.get("value_si") is None:
            res["why"] = "no_prior"; out.append(res); continue
        scene_ppx = None
        sr_ = SCENE_RULER.get(vj["video_id"])
        if sr_ and sr_.get("ok") and not is3d and pr.get("kind") == "size" and pr.get("obj") and re.search(sr_.get("prior_re", r"\blanes?\b"), pr["obj"]):
            scene_ppx = float(sr_["frac_w"]) * V.W
        elif (pr.get("obj") and re.search(SCENE, pr["obj"])):
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
        if scene_ppx is not None:
            sr_ = SCENE_RULER[vj["video_id"]]
            if sr_.get("lines_w"):
                th_ = math.radians(float(sr_["theta"]))
                bands_ = sr_.get("bands_w") or [[min(sr_["lines_w"]), max(sr_["lines_w"])]]
                pad_ = 0.15 * float(sr_["frac_w"])

                def inband(trk):
                    c_ = float(np.median((-trk.uv[:, 0] * math.sin(th_) + trk.uv[:, 1] * math.cos(th_)) / V.W))
                    return any(b_[0] - pad_ <= c_ <= b_[1] + pad_ for b_ in bands_)
                if not inband(tgt) or (kind == "dist2" and tgt2 is not None and not inband(tgt2)):
                    res["why"] = "scene_ruler_offband"; out.append(res); continue
            mpp = pr["value_si"] / scene_ppx
            if kind == "size":
                e, js = tgt.ext_px(q["dim"], r["target"]["phrase"])
                v = agg_ext(e, q["dim"])
            elif kind == "dist2":
                vs_ = [dist2_px(tgt, t2_, q["t"], q["when"], minimum=bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", r["question"].lower())))
                       for t2_ in nearest_cands(V, r, tgt2)]
                vs_ = [x_ for x_ in vs_ if x_[0] is not None and np.isfinite(x_[0]) and x_[0] > 0] or vs_[:1]
                v, gap = min(vs_, key=lambda x_: x_[0] if (x_[0] is not None and np.isfinite(x_[0])) else np.inf)
                res["gap"] = gap
            else:
                v = px_quantity(tgt, kind, q["t"], q["t1"], q["t2"], q["when"])
            if v is None or not np.isfinite(v) or v <= 0:
                res["why"] = "target_px_fail"; out.append(res); continue
            res.update(val_si=v * mpp, mpp=mpp, mode="2d_ratio", ok=True, sruler=SCENE_RULER[vj["video_id"]].get("kind", "scene"))
            out.append(res); continue
        gravity = pr.get("gravity", False)
        g_info = None
        if gravity and "a" in FIX:
            cands = [prior_tr] if prior_tr is not None else [trk for trk in V.tr.values() if track_ok(trk, True)[0]]
            best, bev = None, None
            for trk in cands:
                if trk is None:
                    continue
                g_px, inf = freefall(trk, a=pr.get("t1"), b=pr.get("t2"))
                if g_px is not None:
                    ev = (0 if inf.get("sub") else 1, inf["dur"])
                    if bev is None or ev > bev:
                        best, bev, g_info = trk, ev, (g_px, inf)
            if best is None:
                if not (is3d and Dfn_available(r, tgt, V)):
                    res["why"] = "prior_no_freefall"; out.append(res); continue
                g_info = None
            prior_tr = best if best is not None else prior_tr
        elif gravity and prior_tr is None:
            best, ba = None, 0
            for k, trk in V.tr.items():
                if not track_ok(trk, True)[0]:
                    continue
                ay = M1.gravity_ay(trk, pr.get("t1"), pr.get("t2"))
                if ay is not None and ay > ba:
                    best, ba = trk, ay
            prior_tr = best
        grav_fail3d = gravity and "a" in FIX and g_info is None
        belt_ppx = None
        if (os.environ.get("QP_BELT") == "1" and not is3d and pr.get("kind") == "speed" and pr.get("t") is None
                and re.search(r"\b(?:conveyor|belt|escalator|treadmill|travelator|walkway)\b", str(pr.get("obj") or ""))):
            okp_, _ = track_ok(prior_tr, True) if prior_tr is not None else (False, "")
            if not okp_:
                sp_ = []
                for trk_ in list({id(x_): x_ for x_ in V.tr.values() if x_ is not None and x_ is not prior_tr}.values()):
                    if track_ok(trk_, True)[0]:
                        v_ = px_quantity(trk_, "avgspeed", None, None, None, None)
                        if v_ is not None and np.isfinite(v_) and v_ > 0:
                            sp_.append(float(v_))
                if len(sp_) >= 2 and max(sp_) / min(sp_) <= 1.5:
                    belt_ppx = float(np.median(sp_)); res["grav"] = f"belt_from_{len(sp_)}_objects"
        if not grav_fail3d and belt_ppx is None:
            global STATIC_FRAC
            _sf = STATIC_FRAC
            if os.environ.get("QP_STATIC_PRIOR"):
                STATIC_FRAC = float(os.environ["QP_STATIC_PRIOR"])
            okp, whyp = track_ok(prior_tr, pr["kind"] != "size", is3d and depth_moves(r.get("prior_dkeys") or []))
            STATIC_FRAC = _sf
            if not okp:
                res["why"] = "prior_" + whyp; out.append(res); continue
        if prior_tr is not None and prior_tr is not tgt and M1.same_object(prior_tr, tgt) and r["prior_obj"].get("phrase") and contrastive_row(r, r["target"], r["prior_obj"]):
            skip = "d" in FIX and kind == "dist2" and (r.get("target2") or {}).get("where") == ["#second"]
            if not skip:
                res["why"] = "target=prior_object"; out.append(res); continue
        if kind == "dist2" and M1.same_object(tgt, tgt2) and ("d" not in FIX or tgt is tgt2 or tc.contrastive_v2(r["target"]["phrase"], r["target"].get("where"), r["target2"]["phrase"], r["target2"].get("where"))):
            res["why"] = "target=target2"; out.append(res); continue
        pdim = pr.get("dim") or "length"
        res["owlvote"] = bool(any(x is not None and x.v2 and "e" in FIX and (x.best_gd or 0) < 0.3 for x in (tgt, tgt2, prior_tr)))
        res["nseg"] = len(tgt.segs()["segs"]) if ("c" in FIX and tgt is not None and tgt.n) else np.nan
        sw = set(k for _, ks, cur, best in V.colour_swaps for k, c0, b0 in zip(ks, cur, best) if c0 != b0)
        res["colswap"] = bool(sw and any(V.key(e) in sw for e in (r["target"], r.get("target2"), r["prior_obj"]) if e))
        tk = r.get("target_dkeys") or []
        pk = r.get("prior_dkeys") or []
        if gravity and "a" in FIX and prior_tr is not None and not pk:
            pk = qkeys.get(id(prior_tr), [])
        Dt = M1.track_depth_fn(r["depth"], tk, tgt) if is3d else None
        Dp = M1.track_depth_fn(r["depth"], pk, prior_tr) if (is3d and prior_tr is not None) else None
        same_obj = prior_tr is tgt
        if "d" in FIX and prior_tr is not None and not same_obj and M1.same_object(prior_tr, tgt):
            same_obj = True
        if is3d and "b" in FIX and Dt is None and same_obj and Dp is None and pr["kind"] in ("size", "speed", "accel")                 and not (gravity and "a" not in FIX):
            r_ = known_size(r, q, pr, tgt, V, fdef, gravity, g_info, cx, cy, f_override=f_moge if "known" in MOGE_OPTS else None)
            res.update(r_)
            out.append(res); continue
        use3d = is3d and (Dt is not None or Dp is not None) and not (same_obj and Dp is None)
        if kind == "hfloor" and (use3d or grav_fail3d):
            res["why"] = "hfloor_3d"; out.append(res); continue
        if grav_fail3d:
            if Dt is None:
                res["why"] = "prior_no_freefall"; out.append(res); continue
            use3d = True
        if not use3d:
            if gravity and "a" in FIX:
                ppx = g_info[0]
            elif belt_ppx is not None:
                ppx = belt_ppx
            else:
                ppx = prior_px(pr, prior_tr, pdim, r, gravity)
            if (os.environ.get("QP_SAMECLASS") == "1" and pr["kind"] == "size" and not gravity
                    and re.search(r"\b(?:one|each|every|any)\b", str(pr.get("raw") or "").lower())):
                hn_ = tc.head_noun(r["prior_obj"].get("phrase"))
                alts_ = []
                for trk_ in {id(x_): x_ for x_ in V.tr.values() if x_ is not None}.values():
                    if trk_ is prior_tr or not trk_.n or tc.head_noun(trk_.q.get("phrase")) != hn_ or not track_ok(trk_, False)[0]:
                        continue
                    e_, _ = trk_.ext_px(pdim, trk_.q.get("phrase"))
                    a_ = agg_ext(e_, pdim) if len(e_) else None
                    if a_ is not None and np.isfinite(a_) and a_ > 0:
                        alts_.append(float(a_))
                if alts_:
                    am_ = float(np.median(alts_))
                    if ppx is None or not np.isfinite(ppx) or ppx <= 0 or max(ppx / am_, am_ / ppx) > 2.0:
                        res["grav"] = f"sameclass_{len(alts_)}"; ppx = am_
            if ppx is None or not np.isfinite(ppx) or ppx <= 1e-6:
                res["why"] = "prior_px_fail"; out.append(res); continue
            mpp = pr["value_si"] / ppx
            if gravity and "a" in FIX:
                okg, wg = gravity_scale_check(mpp, prior_tr, V)
                res["grav"] = wg
                if not okg:
                    res["why"] = wg; out.append(res); continue
            if kind == "size":
                e, js = tgt.ext_px(q["dim"], r["target"]["phrase"])
                v = agg_ext(e, q["dim"])
            elif kind == "dist2":
                vs_ = [dist2_px(tgt, t2_, q["t"], q["when"], minimum=bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", r["question"].lower())))
                       for t2_ in nearest_cands(V, r, tgt2)]
                vs_ = [x_ for x_ in vs_ if x_[0] is not None and np.isfinite(x_[0]) and x_[0] > 0] or vs_[:1]
                v, gap = min(vs_, key=lambda x_: x_[0] if (x_[0] is not None and np.isfinite(x_[0])) else np.inf)
                res["gap"] = gap
            elif kind == "hfloor":
                v = hfloor_px(tgt, V, q["t"])
            else:
                v = px_quantity(tgt, kind, q["t"], q["t1"], q["t2"], q["when"])
            if v is None or not np.isfinite(v) or v <= 0:
                res["why"] = "target_px_fail"; out.append(res); continue
            if "m2d" in MOGE_OPTS and mg is not None and not is3d and prior_tr is not None and not same_obj:
                rts = [moge_ratio(mg, tgt, prior_tr)] + ([moge_ratio(mg, tgt2, prior_tr)] if (kind == "dist2" and tgt2 is not None) else [])
                rts = [x for x in rts if x is not None and np.isfinite(x) and x > 0]
                if rts:
                    rt = float(np.exp(np.mean(np.log(rts))))
                    res["m2d"] = rt
                    if 1 / 3 <= rt <= 3:
                        v = v * rt
            res.update(val_si=v * mpp, mpp=mpp, mode=("2d_ratio" if not is3d else "3d_samedepth") + ("_scene" if scene_t else ""), ok=True)
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
                return agg3(e * (D / nr) / f, pdim)
            P, _ = traj3d(prior_tr, Dfn, f)
            if gravity:
                if "a" in FIX:
                    g3, _ = freefall(prior_tr, P=P, a=pr.get("t1"), b=pr.get("t2"))
                    return g3
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
            return prior_kin(prior_tr, pr, P=P)

        f_cal, mode, cal_slope = None, "", np.nan
        ckey = (id(prior_tr), tuple(pk), tuple(tk) if (Dp is None and same_obj) else ())
        if grav_fail3d:
            mode = "3d_gravnofall"
        elif ckey in calib_cache:
            f_cal, mode, cal_slope = calib_cache[ckey]
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
                    lo_, hi_ = max(0, i - 6), min(len(grid) - 1, i + 6)
                    if np.isfinite(vals[lo_]) and np.isfinite(vals[hi_]) and vals[lo_] > 0 and vals[hi_] > 0:
                        cal_slope = float((np.log(vals[hi_]) - np.log(vals[lo_])) / (np.log(grid[hi_]) - np.log(grid[lo_])))
            calib_cache[ckey] = (f_cal, mode, cal_slope)
        fmed = fdef.get(r["video_source"])
        fref = f_moge if (f_moge is not None and "reject" in MOGE_OPTS) else fmed
        if f_cal is not None and f_moge is not None:
            MOGE_STATS.append(dict(video_id=vj["video_id"], src=r["video_source"], f_cal=f_cal / Wd, f_moge=f_moge / Wd, id=r["id"]))
        if f_cal is not None and fref is not None and not (float(os.environ.get("QP_REJ_LO", 0.5)) <= f_cal / fref <= float(os.environ.get("QP_REJ_HI", 2.0))):
            f_cal, mode = None, "3d_calibrejected"
        if f_cal is None:
            if f_moge is not None and ("fallback" in MOGE_OPTS or ("fbrej" in MOGE_OPTS and mode == "3d_calibrejected")):
                f_cal = f_moge
                mode = "3d_mogefov" if mode == "" else mode + "_mogefov"
            else:
                f_cal = fmed if fmed is not None else 1.0 * Wd
                mode = "3d_fovdefault" if mode == "" else mode + "_fovdefault"
        if FOCAL_MIX and mode == "3d_calib" and f_moge is not None and pr["kind"] != "size":
            f_cal = math.sqrt(f_cal * f_moge) if FOCAL_MIX == "gm" else f_moge
        if os.environ.get("QP_FOCAL_LAB") and mode == "3d_calib" and fmed is not None and pr["kind"] != "size" and str(r["video_source"]).startswith("lab"):
            f_cal = math.sqrt(f_cal * fmed)
        if gravity and "a" in FIX and g_info is None and Dt is not None:
            res["grav"] = "no_freefall->depth+focal"
        Dfn_t = Dt if Dt is not None else Dp
        tdep = "" if Dt is not None else "_tdepth=prior"
        if Dt is None and Dp is not None and mg is not None and not same_obj and "tdepth" in MOGE_OPTS:
            rt = moge_ratio(mg, tgt, prior_tr)
            if rt is not None and 0.05 <= rt <= 20:
                Dfn_t = (lambda tt, _D=Dp, _r=rt: _D(tt) * _r)
                tdep = "_tdepth=moge"
        if Dt is None and Dp is not None and dg is not None and not same_obj and os.environ.get("QP_TDEPTH_SRC") == "da3":
            rt = moge_ratio(dg, tgt, prior_tr)
            if rt is not None and 0.05 <= rt <= 20:
                Dfn_t = (lambda tt, _D=Dp, _r=rt: _D(tt) * _r)
                tdep = "_tdepth=da3"
        if Dt is None and Dp is not None and not same_obj and os.environ.get("QP_GROUND") and tgt is not None and tgt.n:
            gf = ground_fit(f_cal)
            if gf is not None:
                yh_, h_ = gf
                js_ = np.array([j for j in tgt.okc if V.H - 2 > tgt.st[j]["box"][3] > yh_ + 3])
                if len(js_) >= 3:
                    yb_ = np.array([tgt.st[j]["box"][3] for j in js_], float)
                    Zg_ = f_cal * h_ / (yb_ - yh_)
                    uvg_ = np.array([tgt.st[j]["c"] for j in js_], float)
                    Dg_ = Zg_ * np.sqrt(1 + ((uvg_[:, 0] - cx) ** 2 + (uvg_[:, 1] - cy) ** 2) / f_cal ** 2)
                    if len(Dg_) >= 5:
                        Dg_ = np.array([np.median(Dg_[max(0, i_ - 2):i_ + 3]) for i_ in range(len(Dg_))])
                    dref_ = float(np.median(Dp(tgt.t_all[js_])))
                    if 0.2 <= np.median(Dg_) / dref_ <= 5:
                        tg_ = tgt.t_all[js_].astype(float)
                        Dfn_t = (lambda tt, _t=tg_, _d=Dg_: np.interp(np.atleast_1d(np.asarray(tt, float)), _t, _d))
                        tdep = "_tdepth=ground"
        if kind == "size":
            e, js = tgt.ext_px(q["dim"], r["target"]["phrase"])
            if len(e) == 0:
                res["why"] = "target_px_fail"; out.append(res); continue
            uv = np.array([tgt.st[j]["c"] for j in js], float)
            D = Dfn_t(tgt.t_all[js])
            _, nr = to3d(uv, D, f_cal, cx, cy)
            v = agg3(e * (D / nr) / f_cal, q["dim"])
        elif kind == "dist2":
            D2 = M1.track_depth_fn(r["depth"], r.get("target2_dkeys") or [], tgt2)
            if D2 is None and mg is not None and tgt2 is not None and tgt2 is not tgt and "tdepth" in MOGE_OPTS:
                rt2 = moge_ratio(mg, tgt2, tgt)
                if rt2 is not None and 0.05 <= rt2 <= 20:
                    D2 = (lambda tt, _D=Dfn_t, _r=rt2: _D(tt) * _r)
            if D2 is None and dg is not None and tgt2 is not None and tgt2 is not tgt and os.environ.get("QP_TDEPTH_SRC") == "da3":
                rt2 = moge_ratio(dg, tgt2, tgt)
                if rt2 is not None and 0.05 <= rt2 <= 20:
                    D2 = (lambda tt, _D=Dfn_t, _r=rt2: _D(tt) * _r)
            D2 = D2 or Dfn_t
            P1, _ = traj3d(tgt, Dfn_t, f_cal)
            P2, _ = traj3d(tgt2, D2, f_cal)
            v, gap = dist2_px(tgt, tgt2, q["t"], q["when"], P1, P2, D1=Dfn_t, D2=D2, f=f_cal, cx=cx, cy=cy,
                              minimum=bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", r["question"].lower())))
            res["gap"] = gap
        else:
            P, Z = traj3d(tgt, Dfn_t, f_cal)
            if kind == "maxheight":
                Y = (tgt.uv[:, 1] - cy) * Z / f_cal
                v = float(Y.max() - Y.min()) if tgt.n >= 3 else None
            else:
                v = px_quantity(tgt, kind, q["t"], q["t1"], q["t2"], q["when"], P=P)
        if v is None or not np.isfinite(v) or v <= 0:
            res["why"] = "target_px_fail"; out.append(res); continue
        res.update(val_si=float(v), f=f_cal / Wd, mode=mode + tdep + ("_scene" if scene_t else ""), ok=True, cal_slope=cal_slope)
        out.append(res)
    if os.environ.get("QP_SA2VA_WEAK") == "1":
        for res_ in out:
            if res_.get("sa2va_weak") and res_.get("ok") and res_.get("mode"):
                res_["mode"] = str(res_["mode"]) + "_sa2vaweak"
    if os.environ.get("QP_CLIPPED_WEAK") == "1":
        for res_ in out:
            if res_.get("clipped") and res_.get("ok") and res_.get("mode"):
                res_["mode"] = str(res_["mode"]) + "_clipped"
    if os.environ.get("QP_SHARED_WEAK") == "1":
        for res_ in out:
            if res_.get("shared") and res_.get("ok") and res_.get("mode"):
                res_["mode"] = str(res_["mode"]) + "_shared"
    return out, V


def Dfn_available(r, tgt, V):
    return bool(r.get("target_dkeys"))


def prior_px(pr, prior_tr, pdim, r, gravity):
    if pr["kind"] == "size":
        e, js = prior_tr.ext_px(pdim, r["prior_obj"]["phrase"])
        return agg_ext(e, pdim)
    if gravity:
        return M1.gravity_ay(prior_tr, pr.get("t1"), pr.get("t2"))
    return prior_kin(prior_tr, pr)


def known_size(r, q, pr, tgt, V, fdef, gravity, g_info, cx, cy, f_override=None):
    res = dict(ok=False)
    fmed = f_override if f_override is not None else fdef.get(r["video_source"])
    if fmed is None:
        res["why"] = "knownsize_no_focal"; return res
    f = fmed
    if tgt.n < 5 or len(tgt.oke) < 5:
        res["why"] = "knownsize_few_frames"; return res
    js = tgt.oke
    if pr["kind"] == "size":
        pdim = pr.get("dim") or "length"
        e_ref, _ = tgt.ext_px(pdim, r["prior_obj"]["phrase"])
        S = pr["value_si"]
    else:
        e_ref = np.sqrt(4 * np.array([tgt.st[j]["area"] for j in js], float) / np.pi)
        S = 1.0
    e_ref = np.asarray(e_ref, float)
    if len(e_ref) != len(js) or np.any(e_ref <= 0):
        res["why"] = "knownsize_extent"; return res
    e_s = np.array([np.median(e_ref[max(0, i - 2):i + 3]) for i in range(len(e_ref))])
    t_e = tgt.t_all[js]

    def Zf(tt, S_):
        return np.interp(np.atleast_1d(tt), t_e, f * S_ / e_s)

    def traj(S_):
        Z = Zf(tgt.t, S_)
        P = np.stack([(tgt.uv[:, 0] - cx) * Z / f, (tgt.uv[:, 1] - cy) * Z / f, Z], 1)
        return P, Z
    if pr["kind"] != "size":
        P1, _ = traj(1.0)
        if gravity:
            q1, _ = freefall(tgt, P=P1, a=pr.get("t1"), b=pr.get("t2")) if "a" in FIX else (None, None)
            if q1 is None:
                res["why"] = "knownsize_no_freefall"; return res
        else:
            q1 = prior_kin(tgt, pr, P=P1)
        if q1 is None or not np.isfinite(q1) or q1 <= 0:
            res["why"] = "knownsize_prior_fail"; return res
        S = pr["value_si"] / q1
    if not (1e-3 <= S <= 20):
        res["why"] = f"knownsize_S({S:.3g})"; return res
    P, Z = traj(S)
    if not (0.05 <= np.median(Z) <= 200):
        res["why"] = f"knownsize_Z({np.median(Z):.3g})"; return res
    kind = q["kind"]
    if kind == "size":
        e, js2 = tgt.ext_px(q["dim"], r["target"]["phrase"])
        if len(e) == 0:
            res["why"] = "target_px_fail"; return res
        v = agg_ext(np.asarray(e) * Zf(tgt.t_all[js2], S) / f, q["dim"])
    elif kind == "dist2":
        res["why"] = "knownsize_dist2"; return res
    elif kind == "maxheight":
        v = float(P[:, 1].max() - P[:, 1].min())
    else:
        v = px_quantity(tgt, kind, q["t"], q["t1"], q["t2"], q["when"], P=P)
    if v is None or not np.isfinite(v) or v <= 0:
        res["why"] = "target_px_fail"; return res
    return dict(ok=True, why="", val_si=float(v), f=f / V.W, mode="3d_knownsize" + ("_moge" if f_override is not None else ""),
                grav=f"S={S:.3g}m")


def dedup_depth(dep):
    from collections import OrderedDict
    by_t = OrderedDict()
    for d in dep:
        by_t.setdefault(d.get("t"), []).append(dict(d))
    keys_all = []
    for ds in by_t.values():
        for d in ds:
            if d["key"] not in keys_all:
                keys_all.append(d["key"])
    order = next((list(d["key"] for d in ds) for ds in by_t.values() if len(set(d["key"] for d in ds)) == len(ds) and len(ds) > 1), None)
    out = []
    for t, ds in by_t.items():
        ks = [d["key"] for d in ds]
        dup = [k for k in set(ks) if ks.count(k) > 1]
        if not dup:
            out += ds; continue
        missing = [k for k in keys_all if k not in ks]
        if len(dup) == 1 and ks.count(dup[0]) == 2 and len(missing) == 1 and order and len(order) == len(ds):
            for d, k in zip(ds, order):
                d["key"] = k if k in (dup[0], missing[0]) else d["key"]
            if sorted(d["key"] for d in ds) == sorted(order):
                out += ds; continue
        out += [d for d in ds if d["key"] not in dup]
    return out


def load_split_tracks(split, cfg):
    use_v2 = "d" in FIX
    ppath = f"{R}/geo/out/parse_{split}_v2{PARSE_TAG}.jsonl" if use_v2 else f"{R}/geo/out/parse_{split}.jsonl"
    recs = [json.loads(l) for l in open(ppath)]
    byv = {}
    for r in recs:
        r["video_source"] = r["video_source"].replace("segmentaion", "segmentation")
        if os.environ.get("QP_DEPTH_DEDUP") == "1" and r.get("depth"):
            r["depth"] = dedup_depth(r["depth"])
        if not use_v2:
            r["target"].setdefault("rel", None); r["prior_obj"].setdefault("rel", None)
            if r.get("target2"):
                r["target2"].setdefault("rel", None)
        if os.environ.get("QP_SIDE_FIX") == "1" and r.get("q", {}).get("kind") == "size":
            ql_ = r["question"].lower()
            tp_ = ((r.get("target") or {}).get("phrase") or "").strip().lower()
            m_ = re.match(r"^(?:the\s+)?(longer|shorter|long|short|longest|shortest)\s+(?:side|edge)$", tp_)
            mo_ = re.search(r"\b(?:longer|shorter|long|short|longest|shortest)\s+(?:side|edge)\s+of\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z \-]*?)\s*(?:\bin\b|,|\?|$)", ql_)
            if m_ and mo_ and not re.search(SCENE, mo_.group(1)):
                r["target"] = dict(r["target"], phrase=mo_.group(1).strip(), raw=mo_.group(1).strip())
                r["q"] = dict(r["q"], dim="length" if m_.group(1).startswith("long") else "width")
                r["side_fix"] = "target"
            elif re.search(r"\bwidth\s*\(\s*(?:the\s+)?long(?:er|est)?\s+side\s*\)", ql_) and r["q"].get("dim") == "width":
                r["q"] = dict(r["q"], dim="length"); r["side_fix"] = "dim"
            elif tp_ in ("rim", "the rim") and r["q"].get("dim") in ("diameter", "width", "radius"):
                mr_ = re.search(r"\brim of (?:the\s+|a\s+)?([a-z][a-z \-]*?)\s*(?:\bin\b|,|\?|$)", ql_)
                if mr_ and not re.search(SCENE, mr_.group(1)):
                    r["target"] = dict(r["target"], phrase=mr_.group(1).strip(), raw=mr_.group(1).strip())
                    r["q"] = dict(r["q"], part=None); r["side_fix"] = "rim"
        if os.environ.get("QP_FLOOR") == "1" and r.get("q", {}).get("kind") == "other":
            mf_ = re.search(r"\bdistance (?:between|of|from) (?:the )?(?:(?:bottom|lowest point|base)(?: corner)? of (?:the )?)?([a-z][a-z \-]*?) "
                            r"(?:and|to|from|above) the (?:floor|ground)(?: at (\d+(?:\.\d+)?)\s*s)?\b", r["question"].lower())
            if mf_ and mf_.group(2) and not re.search(SCENE, mf_.group(1)):
                r["q"] = dict(r["q"], kind="hfloor", t=float(mf_.group(2)) if mf_.group(2) else None, target=mf_.group(1).strip())
                r["target"] = dict(r.get("target") or {}, phrase=mf_.group(1).strip(), raw=mf_.group(1).strip(), where=(r.get("target") or {}).get("where") or [])
                r["target2"] = None; r["target2_dkeys"] = []; r["floor_fix"] = True
        if os.environ.get("QP_PARSE2") == "1":
            q_, ql_ = r.get("q") or {}, r["question"].lower()
            ob_ = r"(?:the\s+)?([a-z][a-z \-]*?)"
            m1_ = re.search(r"\btotal distance (?:that )?" + ob_ + r" (?:walked|traveled|travelled|moved|ran|flew|swam|covered)\b", ql_)
            m2_ = re.search(r"\btotal length of " + ob_ + r"(?:['’]s)? (?:flight |travel |motion )?path\b", ql_)
            m3_ = re.search(r"\bhorizontal distance of " + ob_ + r" during\b", ql_)
            m4_ = re.search(r"\baverage of " + ob_ + r" at \d", ql_)
            new_ = None
            if q_.get("kind") in ("path", "other") and (m1_ or m2_):
                new_ = ("path", (m1_ or m2_).group(1))
            elif q_.get("kind") == "dist2" and not q_.get("target") and m3_:
                new_ = ("disp", m3_.group(1))
            elif (q_.get("kind") == "other" and q_.get("t") is not None and r.get("target2") and (r.get("target") or {}).get("phrase")
                  and re.search(r"\bdistance between\b", ql_) and not re.search(r"\b(?:floor|ground)\b", ql_)):
                new_ = ("dist2", None)
            m5_ = re.search(r"\b(?:the\s+)?([a-z][a-z \-]*?)(?:['’]s)? moving distance between (\d+(?:\.\d+)?)\s*s?\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*s\b", ql_)
            if new_ is None and q_.get("kind") == "dist2" and not ((r.get("target2") or {}).get("phrase")) and m5_:
                q_ = dict(q_, t1=float(m5_.group(2)), t2=float(m5_.group(3)), t=None)
                r["target2"] = None; r["target2_dkeys"] = []
                new_ = ("path", m5_.group(1))
            if new_ is not None and (new_[1] is None or not re.search(SCENE, new_[1])):
                r["q"] = dict(q_, kind=new_[0], **({"target": new_[1].strip()} if new_[1] else {}))
                if new_[1]:
                    r["target"] = dict(r.get("target") or {}, phrase=new_[1].strip(), raw=new_[1].strip(), where=(r.get("target") or {}).get("where") or [])
                r["parse2_fix"] = new_[0]
        byv.setdefault(r["video_id"], []).append(r)
    vjs = {}
    for v in byv:
        p2 = f"{R}/geo/out/tracks_{split}_v2/{v}.json"
        p1 = f"{R}/geo/out/tracks_{split}/{v}.json"
        p3 = f"{R}/geo/out/tracks_{split}_v3/{v}.json"
        if TRACK_VER == "v3":
            if os.path.exists(p3):
                vjs[v] = json.load(open(p3))
            continue
        if TRACK_VER not in ("v2", "v3"):
            px = f"{R}/geo/out/tracks_{split}_{TRACK_VER}/{v}.json"
            if os.path.exists(px):
                vjs[v] = json.load(open(px))
                continue
        if use_v2 and os.path.exists(p2):
            vjs[v] = json.load(open(p2))
        elif os.path.exists(p1):
            vjs[v] = json.load(open(p1))
    if os.environ.get("QP_TSHIFT", "") == "1":
        for v, rs in byv.items():
            if v not in vjs or not vjs[v].get("fidx"):
                continue
            dur = (max(vjs[v]["fidx"]) + 1) / float(vjs[v]["fps"])
            ts = [float(r["q"][k]) for r in rs for k in ("t", "t1", "t2") if r["q"].get(k) is not None]
            if not ts or max(ts) <= dur + 0.5 or min(ts) - dur < 0 or max(ts) - dur > dur:
                continue
            for r in rs:
                for k in ("t", "t1", "t2"):
                    if r["q"].get(k) is not None:
                        r["q"][k] = float(r["q"][k]) - dur
            print('', file=sys.stderr)
    if os.environ.get("QP_TOFF"):
        src_, off_ = os.environ["QP_TOFF"].split(":")
        for v, rs in byv.items():
            if not str(v).startswith(src_):
                continue
            for r in rs:
                for k in ("t", "t1", "t2"):
                    if r["q"].get(k) is not None:
                        r["q"][k] = max(0.0, float(r["q"][k]) + float(off_))
    return byv, vjs


def main():
    global FIX, T_OWL, TRACK_VER, MOGE_OPTS, MOGE_CORR, HALF_V, HALF_A
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--fix", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tracks", default="v2")
    ap.add_argument("--moge", action="store_true")
    ap.add_argument("--moge_opts", default="reject,fallback,tdepth,known")
    ap.add_argument("--half_v", type=float, default=0.25)
    ap.add_argument("--half_a", type=float, default=0.4)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--winged_len", default="long")
    ap.add_argument("--agg3d", type=float, default=None)
    ap.add_argument("--static_frac", type=float, default=0.25)
    ap.add_argument("--parse_tag", default="")
    ap.add_argument("--focal_mix", default="")
    ap.add_argument("--moge_corr", default="")
    a = ap.parse_args()
    TRACK_VER = a.tracks
    FIX = set(a.fix)
    HALF_V, HALF_A = a.half_v, a.half_a
    global MEASURE_SCENE, WINGED_LEN_SHORT, FOCAL_MIX, PARSE_TAG, AGG3D_P, STATIC_FRAC
    FOCAL_MIX = a.focal_mix
    PARSE_TAG = a.parse_tag
    AGG3D_P = a.agg3d
    STATIC_FRAC = a.static_frac
    MEASURE_SCENE = a.scene
    WINGED_LEN_SHORT = a.winged_len == "short"
    MOGE_OPTS = set(x for x in a.moge_opts.split(",") if x)
    if a.moge_corr:
        MOGE_CORR = json.load(open(a.moge_corr))
    if a.moge:
        print("MoGe uses:", sorted(MOGE_OPTS), "| focal correction per source:", MOGE_CORR)
        import glob
        for p in glob.glob(f"{R}/geo/out/moge_{a.split}/*.npz"):
            z = np.load(p)
            MOGE[os.path.basename(p)[:-4]] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
        print("MoGe videos:", len(MOGE))
    if os.environ.get("QP_TDEPTH_SRC") == "da3":
        import glob
        for p in glob.glob(f"{R}/geo/out/da3_{a.split}/*.npz"):
            z = np.load(p)
            DA3D[os.path.basename(p)[:-4]] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
        print("DA3 target-depth videos:", len(DA3D))
    if os.environ.get("QP_SCENE_RULER") == "1":
        fn_ = f"{R}/geo/out/scene_ruler_{a.split}.json"
        if os.path.exists(fn_):
            SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
        print("scene rulers:", len(SCENE_RULER), sorted(SCENE_RULER)[:20])
    M1.SCALE_DEPTH = True
    if os.environ.get("QP_SD_MODE"):
        M1.track_depth_fn = track_depth_fn_v3
    cfg = "e" if "e" in FIX else ("d" if "d" in FIX else "v1")
    if a.cfg:
        cfg = a.cfg
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        T_OWL = json.load(open(tp))["T_owl"]
    byv, vjs = load_split_tracks(a.split, cfg)
    fmed = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
    print("fix", a.fix or "-", "| T_owl", T_OWL, "| fallback focal f/W:", {s: round(f, 3) for s, f in fmed.items()})
    allf = float(np.median(list(fmed.values()))) if fmed else 1.0
    rows = []
    for v, rs in byv.items():
        if v not in vjs:
            for r in rs:
                rows.append(dict(id=r["id"], kind=r["q"]["kind"], ok=False, why="no_track_file"))
            continue
        W = vjs[v]["W"]
        fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * W for r in rs}
        o, V = measure_video(rs, vjs[v], fdd, cfg)
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
            x.update(video_id=v, cat=r["cat"], source=r["video_source"], cam_mag=V.cam_mag / V.W,
                     tracks="v2" if vjs[v].get("version") == "v2" else "v1")
            rows.append(x)
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/geo/out/meas_{a.split}_v2{a.tag}.csv", index=False)
    if MOGE_STATS:
        ms = pd.DataFrame(MOGE_STATS).drop_duplicates(["video_id", "f_cal"])
        ms.to_csv(f"{R}/geo/out/moge_vs_calib_{a.split}{a.tag}.csv", index=False)
        lr = np.log(ms.f_cal / ms.f_moge)
        print('')
        for s_, g_ in ms.assign(lr=lr).groupby("src"):
            print('')
    print("measured share", round(df.ok.mean(), 3), "n_ok", int(df.ok.sum()))
    if PRIOR_DRIFT > 0:
        dl = sorted(set(DRIFT_LOG))
        print(f"prior drift guard {PRIOR_DRIFT}: {len(dl)} prior tracks use the drift speed:", dl[:40])
    if os.environ.get("QP_SD_MODE"):
        sl = sorted(set(SD_LOG))
        print(f"QP_SD_MODE={os.environ['QP_SD_MODE']}: {len(sl)} tracks with one depth entry; raw >1.15 {sum(1 for x in sl if x[1] >= 1.15)}, "
              f"fitted >1.15 {sum(1 for x in sl if x[2] >= 1.15)}:", [x for x in sl if (x[1] >= 1.15) != (x[2] >= 1.15)][:40])
    print(df.why.value_counts().head(20).to_dict())
    print(df["mode"].value_counts().to_dict())


if __name__ == "__main__":
    main()
