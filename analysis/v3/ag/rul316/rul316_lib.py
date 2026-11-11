import json
import numpy as np


class Ser:

    def __init__(self, states, polys=None, W=None, H=None):
        ok = [j for j, s in enumerate(states) if s]
        self.j = np.array(ok, int)
        self.c = np.array([states[j]["c"] for j in ok], float).reshape(-1, 2)
        self.box = np.array([states[j]["box"] for j in ok], float).reshape(-1, 4)
        self.rr = np.array([states[j]["rr"][:2] for j in ok], float).reshape(-1, 2)
        self.border = np.array([states[j].get("border", 0) for j in ok], int)
        self.area = np.array([states[j].get("area", 0) for j in ok], float)
        self.poly = None if polys is None else [polys[j] for j in ok]
        self.W, self.H = W, H

    @property
    def n(self):
        return len(self.j)

    def at(self, k):
        if self.n == 0 or k < self.j[0] or k > self.j[-1]:
            return None
        return np.array([np.interp(k, self.j, self.c[:, 0]), np.interp(k, self.j, self.c[:, 1])])


def ser_qtrack(T, oid):
    seq = T["objs"].get(str(oid)) or []
    return Ser([(s or {}).get("st") for s in seq], None, T.get("W"), T.get("H"))


def ser_inst(T, inst):
    return Ser(inst["st"], inst.get("poly"), T.get("W"), T.get("H"))


def ext_px(S, dim, inner=True):
    if S.n == 0:
        return None, 0
    m = S.border == 0 if inner else np.ones(S.n, bool)
    if m.sum() < max(3, 0.3 * S.n):
        m = np.ones(S.n, bool)
    rr, box = S.rr[m], S.box[m]
    if dim in ("diameter",):
        e = rr.mean(1)
    elif dim in ("width", "thickness", "short"):
        e = rr[:, 1]
    elif dim in ("height",):
        e = box[:, 3] - box[:, 1]
    else:
        e = rr[:, 0]
    return float(np.median(e)), int(m.sum())


def speed_at(S, t, fps, step=1):
    k = int(round(t * fps))
    a, b = S.at(k), S.at(k + step)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(b - a)) * fps / step


def avg_vel(S, t1, t2, fps):
    a, b = S.at(int(round(t1 * fps))), S.at(int(round(t2 * fps)))
    if a is None or b is None or t2 <= t1:
        return None
    return float(np.linalg.norm(b - a)) / (t2 - t1)


def path_speed(S, fps, stride=None):
    if S.n < 3:
        return None
    st = stride or max(1, int(round(fps / 4)))
    ks = list(range(int(S.j[0]), int(S.j[-1]) + 1, st))
    if ks[-1] != int(S.j[-1]):
        ks.append(int(S.j[-1]))
    P = np.array([S.at(k) for k in ks])
    dur = (ks[-1] - ks[0]) / fps
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) / dur if dur > 0 else None


def disp(S):
    if S.n < 2:
        return None
    return float(np.linalg.norm(S.c[-1] - S.c[0]))


def accel_at(S, t, fps, h=None):
    h = h or max(1, int(round(0.2 * fps)))
    k = int(round(t * fps))
    a, b, c = S.at(k - h), S.at(k), S.at(k + h)
    if a is None or b is None or c is None:
        return None
    return float(np.linalg.norm(c - 2 * b + a)) * fps * fps / (h * h)


def dist_at(S1, S2, when):
    common = sorted(set(S1.j.tolist()) & set(S2.j.tolist()))
    if not common:
        return None
    k = common[0] if when in ("start", None) else common[-1]
    return float(np.linalg.norm(S1.at(k) - S2.at(k)))


def period_px(gray, y0, y1, lo, hi):
    import cv2
    gx = np.abs(cv2.Sobel(gray[int(y0):int(y1)].astype(np.float32), cv2.CV_32F, 1, 0, ksize=5))
    p = gx.mean(0)
    p = p - cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), max(5.0, hi / 3.0)).ravel()
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / ac[0]
    lo, hi = int(lo), min(int(hi), len(ac) - 1)
    k = lo + int(np.argmax(ac[lo:hi]))
    loc = [i for i in range(max(lo, 1), hi) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] >= 0.8 * ac[k]]
    if loc:
        k = loc[0]
    if 0 < k < len(ac) - 1:
        d = ac[k - 1] - 2 * ac[k] + ac[k + 1]
        kf = k + (0.5 * (ac[k - 1] - ac[k + 1]) / d if d != 0 else 0.0)
    else:
        kf = float(k)
    return float(kf), float(ac[k])


def fuse(g, cs, kind, cat="S2"):
    if not (g and g > 0 and cs and cs > 0):
        return None, "no pair"
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../compliance/rules")))
    import refuse
    if abs(np.log(g / cs)) <= refuse.BAND:
        w = refuse.agree_w(kind, cat)
        return float(g ** w * cs ** (1 - w)), f"agree w={w}"
    return None, "outside band"


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def instances(T, phrase, min_frac=0.8):
    cand = [m for m in T["inst"] if m["phrase"] == phrase and sum(1 for s in m["st"] if s) >= min_frac * T["N"]]
    cand.sort(key=lambda m: -m["score"])
    keep = []
    for m in cand:
        dup = False
        for k in keep:
            com = [j for j in range(T["N"]) if m["st"][j] and k["st"][j]]
            if com and np.mean([box_iou(m["st"][j]["box"], k["st"][j]["box"]) > 0.5 for j in com]) >= 0.5:
                dup = True; break
        if not dup:
            keep.append(m)
    return keep


def middle_of(ms):
    if len(ms) < 3 or len(ms) % 2 == 0:
        return None
    com = [j for j in range(len(ms[0]["st"])) if all(m["st"][j] for m in ms)]
    if not com:
        return None
    j = com[0]
    return sorted(ms, key=lambda m: m["st"][j]["c"][0])[len(ms) // 2]


def tip_series(T, m):
    st = []
    for s, pl in zip(m["st"], m["poly"]):
        if not s or not pl:
            st.append(None); continue
        P = np.array(max(pl, key=len), float)
        w = P[P[:, 1] >= P[:, 1].max() - 3].mean(0)
        k = int(np.argmax(np.linalg.norm(P - w, axis=1)))
        st.append(dict(s, c=P[k].tolist()))
    return Ser(st, None, T.get("W"), T.get("H"))


def speed_c(S, t, fps, h=None):
    h = h or max(1, int(round(0.1 * fps)))
    k = int(round(t * fps))
    a, b = S.at(k - h), S.at(k + h)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(b - a)) * fps / (2 * h)


def lane_px(frames_bgr):
    import cv2
    from scipy.signal import find_peaks
    out = []
    for f in frames_bgr:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        prof = ((hsv[..., 1] < 40) & (hsv[..., 2] > 170)).sum(1).astype(float)
        pk, _ = find_peaks(prof, distance=30, prominence=5)
        pk = [p for p in pk if prof[p] < 0.5 * f.shape[1]]
        if len(pk) >= 3:
            out.append(float(np.median(np.diff(pk))))
    return (float(np.median(out)), len(out)) if out else (None, 0)
