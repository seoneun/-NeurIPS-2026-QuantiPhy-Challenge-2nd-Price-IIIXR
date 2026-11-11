#!/usr/bin/env python
"""Lane ruler for top-down road videos whose prior is 'lane width = X m' but whose road has a painted centre line only
(no lane edge lines; geo/scene_ruler.py finds no equally spaced lines there): one lane = the distance from the centre
line to the edge of the asphalt, perpendicular to the road. Then the rows' lengths / average speeds are re-measured in
pixels and scaled by X / lane_px (the annotators' 2D-pixel convention, as on internet_0001).

Lane measurement (full-resolution frames, no labels):
  road axis = dominant orientation of painted marks (top-hat + Hough, as scene_ruler) -> rotate the road to vertical ->
  centre line = strongest vertical mark profile peak whose flanks are low-saturation asphalt, refit per frame (x=a+b*y)
  -> per frame, per 24-px block of rows along the road, per side:  e1 = first column (from the line outward) whose
  row-median Lab distance from the local asphalt colour exceeds 30 (end of the grey asphalt);  e2 = first column beyond
  e1 that matches the verge colour (Lab distance < 30 from the median of 8-22 px further out);  edge = (e1+e2)/2 (the
  middle of the dark boundary strip between asphalt and verge).  lane_px = mean over sides of the median over
  (frames x blocks) of |edge - line|.
Object pixels:
  length = along-road extent of the non-asphalt run (Lab distance > 30 from the asphalt next to it) in the inner 60% of
  the track box, full-res frames, median over fully visible frames;  avg speed = slope of the along-road centroid
  (camera-drift corrected) vs time (frame index / the video file's fps) over fully visible frames.  Track set per row:
  the first of TRACK_DIRS where the target's SAM2 object is its own (a track shared by two named targets belongs to the
  best-scored phrase); if none, the target's own detection boxes that avoid the shared track (RANSAC const-velocity).
  The reported length = the track box extent along the road (480p box, +1 px inclusive, scaled to full-res); the
  full-res non-asphalt run is kept in the detail json as a cross-check.
usage: tools/run_limited.sh --io python analysis/v3/ag/lane/lane_edge.py [--split test] [--check]
writes cache/tmp/ag_lane_patch_{test,val}.json, analysis/v3/ag/lane/lane_edge_{split}.json, debug jpgs in analysis/v3/ag/lane/dbg
"""
import os, sys, json, math, re, argparse
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OUT = f"{R}/analysis/v3/ag/lane"
DBG = f"{OUT}/dbg"
os.makedirs(DBG, exist_ok=True)
TA = TV = 30.0                      # Lab distance (OpenCV 8-bit Lab) thresholds: asphalt end / verge match
TRACK_DIRS = ["v2sc", "v2full", "v2hr1080", "v2repro", "v3", "v3nt", ""]   # preference order; per row the first set
# where the target has its own (unshared) SAM2 track wins, else detection fallback in the first set
LANE_RX = r"\blane width\s*=\s*([0-9.]+)\s*m\b"


# ----------------------------------------------------------------------------------------------- video / marks
class Vid:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.cache = {}

    def frame(self, k):
        k = int(min(max(k, 0), self.N - 1))
        if k not in self.cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, fr = self.cap.read()
            if not ok:
                return None
            if len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = fr
        return self.cache[k]


def video_path(split, vid):
    for p in (f"{R}/data/{split}/{vid}.mp4", f"{R}/data/{split}/validation_videos/{vid}.mp4", f"{R}/data/val/{vid}.mp4"):
        if os.path.exists(p):
            return p
    return None


def marking_mask(g):
    k = max(9, int(round(0.012 * max(g.shape))) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(np.float32)
    med = float(np.median(th)); mad = float(np.median(np.abs(th - med))) * 1.4826
    return (th > max(14.0, med + 5 * mad)).astype(np.uint8)


def line_mask(g, S):
    k = max(5, int(round(13 * S / 1080.0)) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(np.float32)
    med = float(np.median(th)); mad = float(np.median(np.abs(th - med))) * 1.4826
    return (th > max(14.0, med + 5 * mad)).astype(np.uint8)


def segments(m):
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 360, threshold=12, minLineLength=max(8, int(0.008 * max(m.shape))), maxLineGap=3)
    return np.zeros((0, 4)) if L is None else L.reshape(-1, 4).astype(float)


def dominant_theta(segs):
    hist = np.zeros(360)
    for x1, y1, x2, y2 in segs:
        ang = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180) % 180
        hist[int(ang * 2) % 360] += math.hypot(x2 - x1, y2 - y1)
    if hist.sum() == 0:
        return None
    hs = sum(np.roll(hist, k) for k in range(-3, 4))
    i = int(np.argmax(hs)); idx = np.arange(i - 3, i + 4); w = hist[idx % 360]
    return float((np.sum(idx * w) / max(w.sum(), 1e-9)) / 2.0) % 180


def rotate_bound(img, ang, border=cv2.BORDER_CONSTANT):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=border)


def first_cross(xs, vals, thr, above=True):
    """first position along xs (ordered outward) where vals crosses thr (above / below), linear sub-pixel interpolation"""
    ok = vals > thr if above else vals < thr
    i = int(np.argmax(ok)) if ok.any() else -1
    if i < 0:
        return None
    if i == 0:
        return float(xs[0])
    v0, v1 = vals[i - 1], vals[i]
    f = (thr - v0) / (v1 - v0) if v1 != v0 else 0.5
    return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))


# ----------------------------------------------------------------------------------------------- lane ruler
def lane_edge(V, vid, nfr=12, debug=True):
    S = min(V.W, V.H); k = S / 1080.0
    idx = np.linspace(0, V.N - 1, nfr).astype(int)
    frs = [f for f in (V.frame(i) for i in idx) if f is not None]
    segs = [segments(marking_mask(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frs]
    th = dominant_theta(np.concatenate(segs))
    if th is None:
        return dict(ok=False, why="no_marks")
    th_hough = th
    def flanked_peaks(rots, prof):
        """profile peaks whose flanks (5-14 px either side) are low-saturation asphalt, strongest first"""
        Hr, Wr = rots[0].shape[:2]
        pk, _ = find_peaks(prof, prominence=0.02 * prof.max(), distance=max(4, int(8 * k)))
        pk = sorted(pk, key=lambda p: -prof[p])[:25]
        hsv = [cv2.cvtColor(r, cv2.COLOR_BGR2HSV)[Hr // 4:3 * Hr // 4, :, 1] for r in rots]
        out = []
        for p in pk:
            fl = np.r_[np.arange(int(p - 14 * k), int(p - 5 * k)), np.arange(int(p + 5 * k), int(p + 14 * k))]
            fl = fl[(fl >= 0) & (fl < Wr)]
            if len(fl) and float(np.median(np.concatenate([h[:, fl].ravel() for h in hsv]))) < 60:
                out.append(int(p))
        return out
    best = None
    for ph in sorted({round(th - 90.0, 3), round(90.0 - th, 3)}):
        rots = [rotate_bound(f, ph) for f in frs]
        prof = gaussian_filter1d(sum(line_mask(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), S).sum(0) for r in rots).astype(float), 3.0)
        fp = flanked_peaks(rots, prof)
        if fp and (best is None or prof[fp[0]] > best[0]):
            best = (prof[fp[0]], ph, rots, fp[0])
    if best is None:
        return dict(ok=False, why="no_centre_line", theta=th)
    _, ph, rots, c0 = best
    c0 = float(c0)
    valid = rotate_bound(np.ones((V.H, V.W), np.uint8), ph)
    Hr, Wr = rots[0].shape[:2]
    B = max(8, int(round(24 * k))); Wmax = int(0.25 * S)
    res = {s: dict(d1=[], d2=[], dm=[]) for s in ("L", "R")}
    lines = []
    for r in rots:
        # the painted line per row: brightest column within +-20 px of the profile peak, kept when it stands out from
        # the row (dash rows); robust line fit x = a + b*y with iterative trimming (3 px)
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = max(0, int(c0 - 20 * k)), min(Wr, int(c0 + 20 * k) + 1)
        win = cv2.GaussianBlur(g[:, lo:hi], (1, 5), 0)
        am = np.argmax(win, 1); pkv = win[np.arange(Hr), am]; med = np.median(win, 1)
        okr = (pkv - med > 35) & (valid[:, lo:hi].min(1) > 0)
        ys = np.nonzero(okr)[0].astype(float)
        xs = np.empty(len(ys))
        for j, y in enumerate(ys.astype(int)):                 # sub-pixel: centroid of the 5 columns around the max
            i0 = am[y]; c_ = np.arange(max(0, i0 - 2), min(hi - lo, i0 + 3)); w_ = np.maximum(win[y, c_] - med[y], 0)
            xs[j] = lo + float((c_ * w_).sum() / max(w_.sum(), 1e-6))
        a_, b_ = c0, 0.0
        if len(ys) > 50:
            keep = np.ones(len(ys), bool)
            for _ in range(4):
                b_, a_ = np.polyfit(ys[keep], xs[keep], 1)
                keep = np.abs(xs - (a_ + b_ * ys)) < 3.0
                if keep.sum() < 50:
                    break
        lines.append((a_, b_))
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB).astype(np.float32)
        for y0 in range(0, Hr - B, B):
            yc = y0 + B / 2; c = a_ + b_ * yc
            xlo, xhi = int(c - Wmax), int(c + Wmax)
            if xlo < 0 or xhi > Wr or valid[y0:y0 + B, xlo:xhi].min() == 0:
                continue
            blk = lab[y0:y0 + B, xlo:xhi]; xx = np.arange(xlo, xhi)
            refm = (np.abs(xx - c) >= 5 * k) & (np.abs(xx - c) <= 14 * k)
            ref = np.median(blk[:, refm].reshape(-1, 3), 0)
            Dm = gaussian_filter1d(np.median(np.linalg.norm(blk - ref, axis=2), 0), 0.7)
            for side, nm in ((-1, "L"), (1, "R")):
                o = np.argsort(side * (xx - c)); o = o[side * (xx[o] - c) >= 5 * k]
                e1 = first_cross(xx[o], Dm[o], TA, above=True)
                if e1 is None:
                    continue
                vm = (side * (xx - e1) >= 8 * k) & (side * (xx - e1) <= 22 * k)
                if vm.sum() < 4:
                    continue
                vref = np.median(blk[:, vm].reshape(-1, 3), 0)
                Dv = gaussian_filter1d(np.median(np.linalg.norm(blk - vref, axis=2), 0), 0.7)
                o2 = o[(side * (xx[o] - e1) > 0) & (side * (xx[o] - e1) <= 14 * k)]
                e2 = first_cross(xx[o2], Dv[o2], TV, above=False) if len(o2) else None
                d1 = abs(e1 - c); res[nm]["d1"].append(d1)
                if e2 is not None:
                    d2 = abs(e2 - c); res[nm]["d2"].append(d2); res[nm]["dm"].append((d1 + d2) / 2)
    bmed = float(np.median([l_[1] for l_ in lines]))
    Mr = cv2.getRotationMatrix2D((0, 0), ph, 1.0)[:, :2]
    d_o = Mr.T @ np.array([bmed, 1.0]); th = math.degrees(math.atan2(d_o[1], d_o[0])) % 180
    out = dict(ok=True, theta=th, theta_hough=th_hough, tilt_b=bmed, rot=ph, centre=c0, W=V.W, H=V.H, k=k, nfr=len(rots))
    for nm in ("L", "R"):
        for q in ("d1", "d2", "dm"):
            v = np.array(res[nm][q])
            out[f"{q}_{nm}"] = float(np.median(v)) if len(v) else float("nan")
            out[f"{q}_{nm}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25])) / np.median(v)) if len(v) else float("nan")
            out[f"{q}_{nm}_n"] = int(len(v))
    for q in ("d1", "d2", "dm"):
        out[f"lane_{q}"] = float(np.nanmean([out[f"{q}_L"], out[f"{q}_R"]]))
    out["lane_px"] = out["lane_dm"]
    out["lr_ratio"] = float(out["dm_L"] / out["dm_R"])
    # per-frame lane (stability across frames)
    out["ok"] = bool(np.isfinite(out["lane_px"]) and 0.8 <= out["lr_ratio"] <= 1.25 and min(out["dm_L_n"], out["dm_R_n"]) >= 30)
    if debug:
        r = rots[len(rots) // 2].copy(); a_, b_ = lines[len(rots) // 2]
        for off, col in ((0, (0, 0, 255)), (-out["d1_L"], (0, 255, 255)), (out["d1_R"], (0, 255, 255)),
                         (-out["d2_L"], (255, 0, 255)), (out["d2_R"], (255, 0, 255)), (-out["dm_L"], (0, 255, 0)), (out["dm_R"], (0, 255, 0))):
            if np.isfinite(off):
                p1 = (int(round(a_ + off)), 0); p2 = (int(round(a_ + b_ * (Hr - 1) + off)), Hr - 1)
                cv2.line(r, p1, p2, col, 1)
        x0 = int(max(0, c0 - 1.6 * out["lane_px"])); x1 = int(min(Wr, c0 + 1.6 * out["lane_px"]))
        crop = r[Hr // 2 - int(1.5 * out["lane_px"]):Hr // 2 + int(1.5 * out["lane_px"]), x0:x1]
        cv2.imwrite(f"{DBG}/{vid}_edge_crop.png", cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
        sc = 1000 / max(r.shape[:2])
        cv2.imwrite(f"{DBG}/{vid}_edge.jpg", cv2.resize(r, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# ----------------------------------------------------------------------------------------------- tracks
def load_tracks(split, vid):
    for d in TRACK_DIRS:
        fn = f"{R}/geo/out/tracks_{split}{'_' + d if d else ''}/{vid}.json"
        if os.path.exists(fn):
            return json.load(open(fn)), fn
    return None, None


def norm_phrase(p):
    return re.sub(r"[^a-z]", "", (p or "").lower())


def query_tracks(vj):
    """query key -> (query dict, per-fidx list of st dicts or None)"""
    out = {}
    for q in vj["queries"]:
        if "objs" in vj:
            fr = vj["objs"].get(str(q.get("obj"))) if q.get("obj") is not None else None
        else:
            fr = q.get("frames")
        sts = [(x or {}).get("st") for x in fr] if fr else [None] * len(vj["fidx"])
        out[q["key"]] = (q, sts)
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def same_track(s1, s2):
    com = [(a, b) for a, b in zip(s1, s2) if a and b]
    return len(com) >= 5 and np.mean([iou(a["box"], b["box"]) > 0.7 for a, b in com]) >= 0.8


def det_track(vj, phrase, avoid, u, fps):
    """detection boxes of every query with this phrase that do not overlap the shared track; RANSAC const-velocity"""
    pts = {}
    for q in vj["queries"]:
        if norm_phrase(q.get("phrase")) != norm_phrase(phrase) or "target" not in (q.get("roles") or []):
            continue
        for i, (b, s) in enumerate(zip(q["det_boxes"], q["det_scores"])):
            if b is None or s <= 0:
                continue
            fi = vj["det_frames"][i]
            ob = avoid[fi] if fi < len(avoid) else None
            if ob is not None and iou(b, ob["box"]) > 0.1:
                continue
            if fi not in pts or s > pts[fi][1]:
                pts[fi] = (b, s)
    if len(pts) < 4:
        return None
    fis = sorted(pts); t = np.array([vj["fidx"][f] / fps for f in fis])
    C = np.array([[(pts[f][0][0] + pts[f][0][2]) / 2, (pts[f][0][1] + pts[f][0][3]) / 2] for f in fis])
    along = C @ u; cross = C @ np.array([-u[1], u[0]])
    best = None
    for i in range(len(fis)):
        for j in range(i + 1, len(fis)):
            v = (along[j] - along[i]) / (t[j] - t[i]); p0 = along[i] - v * t[i]
            inl = (np.abs(along - (v * t + p0)) < 6) & (np.abs(cross - np.median(cross[[i, j]])) < 6)
            if best is None or inl.sum() > best.sum():
                best = inl
    if best.sum() < 4 or t[best].max() - t[best].min() < 1.0:
        return None
    sts = [None] * len(vj["fidx"])
    for f, ok in zip(fis, best):
        if ok:
            b = pts[f][0]
            sts[f] = dict(box=list(b), c=[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], border=0, det=True)
    return sts


def resolve(vj, target):
    """the per-fidx states for the row's target phrase (shared-track fallback to detections)"""
    QT = query_tracks(vj)
    key = next((k for k, (q, _) in QT.items() if norm_phrase(q.get("phrase")) == norm_phrase(target) and "target" in (q.get("roles") or [])), None)
    if key is None:
        return None, "no_query"
    q, sts = QT[key]
    group = [k for k, (q2, s2) in QT.items() if k != key and "target" in (q2.get("roles") or [])
             and norm_phrase(q2.get("phrase")) != norm_phrase(q.get("phrase"))
             and ((q2.get("obj") is not None and q2.get("obj") == q.get("obj")) or same_track(sts, s2))]
    if not group:
        return sts, "track"
    sc_me = max(QT[k][0].get("best_score", 0) for k in QT if norm_phrase(QT[k][0].get("phrase")) == norm_phrase(q.get("phrase")))
    owner = max(group, key=lambda k: QT[k][0].get("best_score", 0))
    if sc_me >= QT[owner][0].get("best_score", 0):
        return sts, "track_shared_owner"
    return ("DET", QT[owner][1]), "det_fallback(shared with %s)" % owner


def cam_shift(vj):
    cam = np.array(vj.get("cam") or [], float)
    if len(cam) != len(vj["fidx"]):
        return np.zeros((len(vj["fidx"]), 2))
    return np.cumsum(cam[:, :2], 0)


def speed_px(sts, vj, fps, u, Wt, Ht, cs):
    rows = []
    for i, st in enumerate(sts):
        if not st or st.get("border", 0):
            continue
        x1, y1, x2, y2 = st["box"]
        if x1 <= 2 or y1 <= 2 or x2 >= Wt - 3 or y2 >= Ht - 3:
            continue
        c = np.array(st["c"], float) - cs[i]
        rows.append((vj["fidx"][i] / fps, float(c @ u)))
    if len(rows) < 4:
        return None
    t, p = np.array(rows).T
    if t.max() - t.min() < 1.0:
        return None
    sl = np.polyfit(t, p, 1)[0]
    ends = (p[-1] - p[0]) / (t[-1] - t[0])
    return dict(v_fit=float(abs(sl)), v_ends=float(abs(ends)), n=len(t), span=float(t.max() - t.min()))


def body_length(V, sts, vj, sc, axis, nmax=14):
    """along-road extent (full-res px) of the non-asphalt run inside the inner 60% of the box"""
    Wt, Ht = vj["W"], vj["H"]
    cand = [i for i, st in enumerate(sts) if st and not st.get("border", 0)
            and st["box"][0] > 2 and st["box"][1] > 2 and st["box"][2] < Wt - 3 and st["box"][3] < Ht - 3]
    if len(cand) > nmax:
        cand = [cand[j] for j in np.linspace(0, len(cand) - 1, nmax).astype(int)]
    Ls, bx = [], []
    for i in cand:
        fr = V.frame(vj["fidx"][i])
        if fr is None:
            continue
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
        x1, y1, x2, y2 = st_box = sts[i]["box"]
        if sts[i].get("det"):
            X1, Y1, X2, Y2 = x1 * sc, y1 * sc, x2 * sc, y2 * sc
        else:
            X1, Y1, X2, Y2 = x1 * sc, y1 * sc, (x2 + 1) * sc, (y2 + 1) * sc
        if axis == 1:                                   # road horizontal: along = x
            lab = lab.transpose(1, 0, 2); X1, Y1, X2, Y2 = Y1, X1, Y2, X2
        w, h = X2 - X1, Y2 - Y1
        c0, c1 = int(round(X1 + 0.2 * w)), int(round(X2 - 0.2 * w))
        r0, r1 = int(max(0, Y1 - 0.6 * h)), int(min(lab.shape[0], Y2 + 0.6 * h))
        if c1 - c0 < 3 or r1 - r0 < 10:
            continue
        win = lab[r0:r1, c0:c1]
        yy = np.arange(r0, r1)
        outside = (yy < Y1 - 0.1 * h) | (yy > Y2 + 0.1 * h)
        if outside.sum() < 4:
            continue
        ref = np.median(win[outside].reshape(-1, 3), 0)
        frac = gaussian_filter1d((np.linalg.norm(win - ref, axis=2) > TA).mean(1), 0.7)
        inw = (yy >= Y1 - 0.3 * h) & (yy <= Y2 + 0.3 * h)          # bridged: first..last object row near the box
        on = np.where((frac > 0.35) & inw)[0]
        if len(on) < 3:
            continue
        lo, hi = int(on[0]), int(on[-1])
        if lo == 0 or hi == len(frac) - 1:
            continue
        if yy[lo] <= Y1 - 0.3 * h + 1 or yy[hi] >= Y2 + 0.3 * h - 1:
            continue
        f_lo = lo - (frac[lo] - 0.35) / max(frac[lo] - frac[lo - 1], 1e-6)
        f_hi = hi + (frac[hi] - 0.35) / max(frac[hi] - frac[hi + 1], 1e-6)
        Ls.append(float(f_hi - f_lo))
        bx.append(float((y2 - y1 + (0 if sts[i].get("det") else 1)) if axis == 0 else (x2 - x1 + (0 if sts[i].get("det") else 1))))
    if not Ls:
        return None
    return dict(L=float(np.median(Ls)), n=len(Ls), spread=float(np.std(Ls) / np.median(Ls)), all=[round(x, 1) for x in Ls],
                box=float(np.median(bx)), box_all=[round(x, 1) for x in bx])


# ----------------------------------------------------------------------------------------------- rows
def load_parse(split):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = j
    return P


def pick_track(split, vid, tgt, u, fps):
    """(vj, track file, per-fidx states, how) for the target: first track set with an own track, else det fallback"""
    first = None
    for d in TRACK_DIRS:
        fn = f"{R}/geo/out/tracks_{split}{'_' + d if d else ''}/{vid}.json"
        if not os.path.exists(fn):
            continue
        vj = json.load(open(fn))
        if "objs" not in vj:
            continue
        if first is None:
            first = (vj, fn)
        sts, how = resolve(vj, tgt)
        if sts is not None and not isinstance(sts, tuple) and any(sts):
            return vj, fn, sts, how
    if first is None:
        return None, None, None, "no_tracks"
    vj, fn = first
    sts, how = resolve(vj, tgt)
    if isinstance(sts, tuple):
        sts = det_track(vj, tgt, sts[1], u, fps)
    return vj, fn, sts, how


def measure_rows(split, rows, lane_m, lane_px_full, vid, V, theta):
    u = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
    if u[1] < 0 or (abs(u[1]) < 1e-9 and u[0] < 0):
        u = -u
    axis = 0 if abs(u[1]) >= abs(u[0]) else 1
    m_per_full = lane_m / lane_px_full
    out, used = {}, set()
    for rid, p in rows:
        q = p["q"]; tgt = q.get("target")
        vj0, _ = load_tracks(split, vid)
        fps = V.fps if V.fps > 1 else float(vj0["fps"])   # time base = the video file's own frame rate (internet_0001:
        # the 24-fps track metadata gives 33.6 m/s, the file's 20.6 fps gives 28.8 against the label 29.3)
        vj, tfn, sts, how = pick_track(split, vid, tgt, u, fps)
        if vj is None or sts is None:
            out[rid] = dict(skip=how or "no_track"); continue
        used.add(tfn)
        sc = V.W / float(vj["W"]); cs = cam_shift(vj)
        rec = dict(how=how, tracks=os.path.basename(os.path.dirname(tfn)), target=tgt, kind=q["kind"], fps=fps)
        if q["kind"] == "size" and (q.get("dim") in ("length", None)):
            b = body_length(V, sts, vj, sc, axis)
            if b is None:
                out[rid] = dict(skip="no_length", **rec); continue
            rec.update(px_box=b["box"], px_box_full=b["box"] * sc, value=b["box"] * sc * m_per_full, px_full=b["L"],
                       value_fullres=b["L"] * m_per_full, n=b["n"], spread=b["spread"], all=b["all"], box_all=b["box_all"])
        elif q["kind"] == "avgspeed" and q.get("t1") is None and q.get("t") is None:
            s = speed_px(sts, vj, fps, u, vj["W"], vj["H"], cs)
            if s is None:
                out[rid] = dict(skip="no_speed", **rec); continue
            rec.update(s)
            rec.update(value=s["v_fit"] * sc * m_per_full, value_ends=s["v_ends"] * sc * m_per_full)
            s0 = speed_px(sts, vj, fps, u, vj["W"], vj["H"], np.zeros_like(cs))
            rec["value_nocam"] = s0["v_fit"] * sc * m_per_full if s0 else None
        else:
            out[rid] = dict(skip="kind", **rec); continue
        out[rid] = rec
    return out, sorted(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--check", type=int, default=0, help="old painted-lane check (superseded by lane_check.py)")
    a = ap.parse_args()
    for split in ("test", "val"):
        P = load_parse(split)
        srj = f"{R}/geo/out/scene_ruler_{split}.json"
        SR = json.load(open(srj)) if os.path.exists(srj) else {}
        byvid = {}
        for rid, p in P.items():
            m = re.search(LANE_RX, p.get("prior_text", ""), re.I)
            if not m or p.get("cat") != "S2":
                continue
            v = p["video_id"]
            if (SR.get(v) or {}).get("ok"):
                continue                                  # painted lanes found: scene_ruler handles it
            byvid.setdefault(v, (float(m.group(1)), []))[1].append((rid, p))
        patch, detail = {}, {}
        for v, (lane_m, rows) in sorted(byvid.items()):
            V = Vid(video_path(split, v))
            le = lane_edge(V, v)
            detail[v] = dict(lane=le)
            print(split, v, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in le.items()}), flush=True)
            if not le.get("ok"):
                continue
            res, tfn = measure_rows(split, rows, lane_m, le["lane_px"], v, V, le["theta"])
            detail[v]["rows"] = res; detail[v]["tracks"] = tfn
            for rid, r in res.items():
                print("  ", rid, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
                if "value" in r and np.isfinite(r["value"]):
                    if r["kind"] == "size":
                        what = "length = box %.1f px along road" % r["px_box_full"]
                    else:
                        what = "avg speed = track slope %.1f px/s" % (r["value"] * le["lane_px"] / lane_m)
                    patch[str(rid)] = dict(value=float(r["value"]), why="lane_edge: %s x %.2f m / %.1f px lane (centre dash to asphalt edge, full-res; %s, %s)"
                                           % (what, lane_m, le["lane_px"], r["tracks"].replace("tracks_%s_" % split, ""), r["how"]))
        json.dump(patch, open(f"{R}/cache/tmp/ag_lane_patch_{split}.json", "w"), indent=1)
        json.dump(detail, open(f"{OUT}/lane_edge_{split}.json", "w"), indent=1, default=float)
        print(split, "patch rows", len(patch), flush=True)
    if a.check:
        check_painted()


def check_painted():
    """label-free checks on internet_0001 (test + val) where painted lane lines exist: (1) the same object-pixel
    machinery with the scene_ruler painted-lane px (compare to the internet_0001 labels only as evaluation);
    (2) the outermost lane (dashed line -> continuous edge line, first non-asphalt crossing as in lane_edge) vs the
    painted spacing"""
    for split in ("test", "val"):
        SR = json.load(open(f"{R}/geo/out/scene_ruler_{split}.json"))
        P = load_parse(split)
        v = "internet_0001"
        if v not in SR:
            continue
        V = Vid(video_path(split, v))
        sr = SR[v]
        lane_full = sr["px"] * V.W / sr["W"]
        rows = [(rid, p) for rid, p in P.items() if p["video_id"] == v]
        res, tfn = measure_rows(split, rows, 3.66, lane_full, v, V, sr["theta"])
        print("CHECK", split, v, "painted lane px(full) %.1f" % lane_full, tfn)
        for rid, r in res.items():
            print("   ", rid, P[rid]["question"], json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}))
        # outermost lane: per block from each dashed line outward, first non-asphalt crossing
        fr = V.frame(V.N // 2)
        th = sr["theta"]; ph = th - 90.0
        rot = rotate_bound(fr, ph); Hr, Wr = rot.shape[:2]
        lab = cv2.cvtColor(rot, cv2.COLOR_BGR2LAB).astype(np.float32)
        m = marking_mask(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY))
        prof = gaussian_filter1d(m.sum(0).astype(float), 1.0)
        pk, _ = find_peaks(prof, prominence=0.05 * prof.max(), distance=8)
        # continuous lines: mark coverage along the road > 0.8; dashed: 0.2-0.7
        cov = np.array([(m[:, max(0, p - 3):p + 4].max(1) > 0).mean() for p in pk])
        print("    peaks", [(int(p), round(float(c), 2)) for p, c in zip(pk, cov)])
        dash = [p for p, c in zip(pk, cov) if 0.15 < c < 0.75]; cont = [p for p, c in zip(pk, cov) if c >= 0.8]
        sp = np.median(np.diff(sorted(dash))) if len(dash) > 1 else float("nan")
        k = min(V.W, V.H) / 1080.0
        for e in cont:
            d0 = min(dash, key=lambda p: abs(p - e)) if dash else None
            if d0 is None:
                continue
            side = 1 if e > d0 else -1
            B = int(24 * k); ds = []
            for y0 in range(0, Hr - B, B):
                xs = np.arange(int(d0 + side * 5 * k), int(d0 + side * 1.5 * sp), side)
                if xs.min() < 0 or xs.max() >= Wr:
                    continue
                blk = lab[y0:y0 + B][:, xs]
                refm = np.abs(xs - d0) <= 14 * k
                ref = np.median(blk[:, refm].reshape(-1, 3), 0)
                Dm = gaussian_filter1d(np.median(np.linalg.norm(blk - ref, axis=2), 0), 0.7)
                e1 = first_cross(xs, Dm, TA)
                if e1 is not None:
                    ds.append(abs(e1 - d0))
            print("    %s continuous line at %d, nearest dashed %d: painted centre-centre %.1f, first-crossing median %.1f (n=%d), dashed spacing %.1f (rotated full-res px)"
                  % (split, e, d0, abs(e - d0), np.median(ds) if ds else float("nan"), len(ds), sp))


if __name__ == "__main__":
    main()
