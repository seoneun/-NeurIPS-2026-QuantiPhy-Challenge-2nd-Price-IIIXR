#!/usr/bin/env python
import os, sys, json, re, math, argparse
import numpy as np, cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--vids", default="")
ap.add_argument("--nframes", type=int, default=8)
ap.add_argument("--maxdim", type=int, default=1920)
ap.add_argument("--debug", type=int, default=1)
ap.add_argument("--out", default="")
a = ap.parse_args()
DBG = f"{R}/cache/tmp/scene_ruler"
os.makedirs(DBG, exist_ok=True)


KINDS = [("lane", r"\blane width\b", r"\blanes?\b"),
         ("tennis_singles", r"\bsingles court width\b", r"court"),
         ("pickleball_width", r"\bwidth of the pickleball court\b|\bpickleball court width\b", r"court"),
         ("basketball_key", r"\b(?:inner )?court width\s*=\s*4\.8[0-9]?\s*m", r"court"),
         ("soccer_circle", r"\bcent(?:er|re) circle radius\b", r"circle")]


def scene_videos(split):
    sys.path.insert(0, f"{R}/caw")
    import analyze_caw as A
    df = A.load_test() if split == "test" else A.load_val()
    out = {}
    for v, p in zip(df.video_id.astype(str), df.ground_truth_prior.astype(str)):
        for k, rx, prx in KINDS:
            if re.search(rx, p, re.I):
                out.setdefault(v, (k, prx))
    return out


def frames(path, n, maxdim):
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, fr = cap.read()
        if not ok:
            continue
        H0, W0 = fr.shape[:2]
        s = min(1.0, maxdim / max(H0, W0))
        if s < 1:
            fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(fr)
    cap.release()
    return out


def marking_mask(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    k = max(9, int(round(0.012 * max(g.shape))) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(np.float32)
    med = float(np.median(th)); mad = float(np.median(np.abs(th - med))) * 1.4826
    t = max(14.0, med + 5 * mad)
    return (th > t).astype(np.uint8), th


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
    i = int(np.argmax(hs))
    idx = np.arange(i - 3, i + 4)
    w = hist[idx % 360]
    return float((np.sum(idx * w) / max(w.sum(), 1e-9)) / 2.0) % 180


def cross_profile(segs_list, theta, size, tol=3.0):
    th = math.radians(theta)
    prof = np.zeros(size)
    off = size / 2
    for segs in segs_list:
        for x1, y1, x2, y2 in segs:
            ang = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180) % 180
            d = abs((ang - theta + 90) % 180 - 90)
            if d > tol:
                continue
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            yc = -xm * math.sin(th) + ym * math.cos(th) + off
            if 0 <= yc < size - 1:
                f = yc - int(yc)
                ln = math.hypot(x2 - x1, y2 - y1)
                prof[int(yc)] += ln * (1 - f); prof[int(yc) + 1] += ln * f
    return prof, off


def best_progression(pk, wt, min_s, tol=0.08, cands=None):
    best = None
    P = list(pk)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            s = P[j] - P[i]
            if s < min_s:
                continue
            mem = [i, j]
            while True:
                nxt = P[mem[-1]] + s
                c = [k for k in range(mem[-1] + 1, len(P)) if abs(P[k] - nxt) <= tol * s]
                if not c:
                    break
                mem.append(min(c, key=lambda k: abs(P[k] - nxt)))
            while True:
                prv = P[mem[0]] - s
                c = [k for k in range(0, mem[0]) if abs(P[k] - prv) <= tol * s]
                if not c:
                    break
                mem.insert(0, min(c, key=lambda k: abs(P[k] - prv)))
            pos = np.array([P[k] for k in mem], float)
            sp = float(np.median(np.diff(pos)))
            strong = [P[k] for k in range(len(P)) if wt[k] >= 0.25 * max(wt[m] for m in mem)]
            if any(pos[q] + 0.25 * sp < x < pos[q + 1] - 0.25 * sp for q in range(len(pos) - 1) for x in strong):
                continue
            score = (len(mem), -sp)
            if cands is not None:
                cands.append((score, sp, pos))
            if best is None or score > best[0]:
                best = (score, sp, pos)
    return best


def merge_close(pk, p, min_gap):
    out_x, out_w = [], []
    for x in pk:
        if out_x and x - out_x[-1][-1] < min_gap:
            out_x[-1].append(x)
        else:
            out_x.append([x])
    xs = np.array([np.average(g, weights=[p[i] for i in g]) for g in out_x])
    ws = np.array([sum(p[i] for i in g) for g in out_x])
    return xs, ws


def centre_line(fr_list, th, xs, ws):
    f = fr_list[0]
    H, W = f.shape[:2]
    t_ = math.radians(th)
    S = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    c = (-xx * math.sin(t_) + yy * math.cos(t_))
    best = None
    for x, w in zip(xs, ws):
        fl = (np.abs(c - x) >= 4) & (np.abs(c - x) <= 14)
        if fl.sum() < 50:
            continue
        s_ = float(np.median(S[fl]))
        if s_ < 70 and (best is None or w > best[0]):
            best = (w, x)
    return None if best is None else best[1]


def road_edge_lane(fr_list, th, c0):
    t_ = math.radians(th)
    halves = []
    for f in fr_list[::max(1, len(fr_list) // 4)]:
        H, W = f.shape[:2]
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        c = np.round(-xx * math.sin(t_) + yy * math.cos(t_)).astype(int)
        cmin = c.min()
        b = (c - cmin).ravel()
        S, V = hsv[..., 1].ravel(), hsv[..., 2].ravel()
        near = (np.abs(c.ravel() - c0) >= 4) & (np.abs(c.ravel() - c0) <= 14)
        if near.sum() < 50:
            continue
        s0, v0 = float(np.median(S[near])), float(np.median(V[near]))
        asph = ((S < s0 + 25) & (np.abs(V - v0) < 45)).astype(np.float32)
        n = np.bincount(b, minlength=c.max() - cmin + 1).astype(np.float32)
        fa = np.bincount(b, weights=asph, minlength=len(n)) / np.maximum(n, 1)
        fa = gaussian_filter1d(fa, 1.0)
        i0 = int(round(c0 - cmin))
        side = []
        for step in (-1, 1):
            k = i0 + step * 4
            while 0 <= k < len(fa) and n[k] > 20 and fa[k] >= 0.5:
                k += step
            side.append(abs(k - i0))
        if min(side) >= 8 and max(side) / min(side) <= 1.25:
            halves.append(float(np.mean(side)))
    if len(halves) < 2:
        return None
    return float(np.median(halves)), float(np.std(halves) / np.median(halves))


def lane_ruler(vid, split):
    path = next((p for p in (f"{R}/data/{split}/{vid}.mp4", f"{R}/data/{split}/validation_videos/{vid}.mp4") if os.path.exists(p)), None)
    fr = frames(path, a.nframes, a.maxdim) if path else []
    if not fr:
        return None
    H, W = fr[0].shape[:2]
    segs = [segments(marking_mask(f)[0]) for f in fr]
    allseg = np.concatenate(segs) if segs else np.zeros((0, 4))
    th = dominant_theta(allseg)
    if th is None:
        return dict(ok=False, why="no_lines")
    size = int(math.hypot(W, H)) + 4
    prof, off = cross_profile(segs, th, size)
    p = gaussian_filter1d(prof, 1.0)
    pk0, props = find_peaks(p, prominence=0.04 * p.max(), distance=4)
    pk, wt = merge_close(pk0, p, max(4.0, 0.006 * max(W, H)))
    cands = []
    bp = best_progression(pk, wt, min_s=max(10.0, 0.008 * max(W, H)), cands=cands)
    if bp is None or bp[0][0] < 3:
        c0 = centre_line(fr, th, [float(x) - off for x in pk], wt)
        if c0 is not None:
            re_ = road_edge_lane(fr, th, c0)
            if re_ is not None:
                sp_, spread_ = re_
                res = dict(ok=spread_ <= 0.08, kind="lane", mode="edge", px=sp_, W=W, H=H, frac_w=sp_ / W, n=2, npk=int(len(pk)),
                           spread=spread_, theta=float(th), lines=[c0 - sp_, c0, c0 + sp_],
                           lines_w=[(c0 - sp_) / W, c0 / W, (c0 + sp_) / W], bands_w=[[(c0 - sp_) / W, (c0 + sp_) / W]])
                if a.debug:
                    im = fr[len(fr) // 2].copy(); t_ = math.radians(th)
                    for c in res["lines"]:
                        if abs(math.cos(t_)) >= 0.5:
                            pts = [(x, (c + x * math.sin(t_)) / math.cos(t_)) for x in (0, W - 1)]
                        else:
                            pts = [((y * math.cos(t_) - c) / math.sin(t_), y) for y in (0, H - 1)]
                        cv2.line(im, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]), (0, 0, 255), 1)
                    sc = 900 / max(im.shape[:2])
                    cv2.imwrite(f"{DBG}/{vid}_lanes.jpg", cv2.resize(im, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 80])
                return res
        return dict(ok=False, why="no_progression", theta=th, npk=int(len(pk)))
    (cnt, _), sp, pos = bp
    d = np.diff(pos)
    bands = [(float(pos[0]), float(pos[-1]))]
    for sc_, sp_, pos_ in sorted(cands, key=lambda c: c[0], reverse=True):
        if sc_[0] < 3 or abs(sp_ - sp) > 0.08 * sp:
            continue
        lo_, hi_ = float(pos_[0]), float(pos_[-1])
        if all(hi_ < b[0] - 0.25 * sp or lo_ > b[1] + 0.25 * sp for b in bands):
            bands.append((lo_, hi_))
    res = dict(ok=True, kind="lane", px=sp, W=W, H=H, frac_w=sp / W, n=int(cnt), npk=int(len(pk)), spread=float(np.std(d) / sp),
               theta=float(th), lines=[float(x - off) for x in pos], lines_w=[float((x - off) / W) for x in pos],
               bands_w=[[float((b[0] - off) / W), float((b[1] - off) / W)] for b in bands])
    if a.debug:
        im = fr[len(fr) // 2].copy()
        t_ = math.radians(th)
        for yc in pos:
            c = yc - off
            pts = []
            for x in (0, W - 1):
                if abs(math.cos(t_)) > 1e-6:
                    pts.append((x, (c + x * math.sin(t_)) / math.cos(t_)))
            if abs(math.cos(t_)) < 0.5:
                pts = [((y * math.cos(t_) - c) / math.sin(t_), y) for y in (0, H - 1)]
            cv2.line(im, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]), (0, 0, 255), 1)
        for x1, y1, x2, y2 in segs[len(fr) // 2]:
            cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        sc = 900 / max(im.shape[:2])
        cv2.imwrite(f"{DBG}/{vid}_lanes.jpg", cv2.resize(im, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return res


def court_lines(fr_list, tol=2.0):
    segs = [segments(marking_mask(f)[0]) for f in fr_list]
    allseg = np.concatenate(segs) if segs else np.zeros((0, 4))
    th = dominant_theta(allseg)
    if th is None:
        return None
    H, W = fr_list[0].shape[:2]
    size = int(math.hypot(W, H)) + 4
    out = {}
    for ax, t_ in (("long", th), ("short", (th + 90) % 180)):
        prof, off = cross_profile(segs, t_, size, tol=tol)
        p = gaussian_filter1d(prof, 1.0)
        pk, _ = find_peaks(p, prominence=0.04 * max(p.max(), 1e-9), distance=3)
        out[ax] = (t_, pk.astype(float) - off, p[pk])
    return out


def tennis_singles(L):
    th, x, w = L["long"]
    best = None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            D = x[j] - x[i]
            if D < 40:
                continue
            exp = [x[i] + D * 1.37 / 10.97, x[j] - D * 1.37 / 10.97]
            tol = max(2.0, 0.015 * D)
            m = [int(np.argmin(np.abs(x - e))) for e in exp]
            if any(abs(x[k] - e) > tol for k, e in zip(m, exp)):
                continue
            c = int(np.argmin(np.abs(x - (x[i] + D / 2))))
            nc = int(abs(x[c] - (x[i] + D / 2)) <= tol)
            err = sum(abs(x[k] - e) for k, e in zip(m, exp)) / D
            sc = (2 + nc, float(w[i] + w[j] + w[m[0]] + w[m[1]]), -err)
            if best is None or sc > best[0]:
                best = (sc, x[m[1]] - x[m[0]], D, [x[i], x[m[0]], x[m[1]], x[j]])
    if best is None:
        return None
    return dict(px=float(best[1]), check=float(best[1] / best[2]), check_ref=8.23 / 10.97, lines=[float(v) for v in best[3]], theta=float(th))


def pickleball_width(L):
    th, x, w = L["long"]
    best = None
    for i in range(len(x)):
        for j in range(i + 2, len(x)):
            D = x[j] - x[i]
            if D < 40:
                continue
            c = int(np.argmin(np.abs(x - (x[i] + D / 2))))
            if abs(x[c] - (x[i] + D / 2)) > max(2.0, 0.015 * D) or c in (i, j):
                continue
            inner = [k for k in range(i + 1, j) if k != c and w[k] >= 0.3 * min(w[i], w[j])]
            if inner:
                continue
            sc = (float(min(w[i], w[j])), D)
            if best is None or sc > best[0]:
                best = (sc, D, [x[i], x[c], x[j]])
    if best is None:
        return None
    th2, y, wy = L["short"]
    pat = [13.41, 4.57, 4.26, 8.84]
    rat = [abs(y[q] - y[p_]) / best[1] for p_ in range(len(y)) for q in range(p_ + 1, len(y))]
    hit = [r_ for r_ in rat for z in pat if abs(math.log(r_ / (z / 6.10))) <= math.log(1.03)]
    return dict(px=float(best[1]), check=1.0 if hit else float("nan"), check_ref=1.0, n_pattern=len(hit),
                lines=[float(v) for v in best[2]], theta=float(th))


def basketball_key(L):
    th, x, w = L["long"]
    best = None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            S = x[j] - x[i]
            if S < 80:
                continue
            cen = (x[i] + x[j]) / 2
            for a_ in range(i + 1, j):
                for b_ in range(a_ + 1, j):
                    k_ = x[b_] - x[a_]
                    r_ = k_ / S
                    if not (0.30 <= r_ <= 0.345) or abs((x[a_] + x[b_]) / 2 - cen) > 0.02 * S:
                        continue
                    sc = (float(min(w[i], w[j])), -abs(r_ - 0.32))
                    if best is None or sc > best[0]:
                        best = (sc, k_, S, [x[i], x[a_], x[b_], x[j]])
    if best is None:
        return None
    return dict(px=float(best[1]), check=float(best[1] / best[2]), check_ref=4.88 / 15.24, lines=[float(v) for v in best[3]], theta=float(th))


def soccer_circle(fr_list):
    rs = []
    for f in fr_list:
        m, _ = marking_mask(f)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        H, W = m.shape
        cs = cv2.HoughCircles(cv2.GaussianBlur(m * 255, (5, 5), 1.5), cv2.HOUGH_GRADIENT, dp=1, minDist=W / 8, param1=100, param2=18,
                              minRadius=int(0.02 * max(W, H)), maxRadius=int(0.2 * max(W, H)))
        if cs is None:
            continue
        best = None
        for cx, cy, r in cs[0]:
            ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
            px_, py_ = cx + r * np.cos(ang), cy + r * np.sin(ang)
            inside = (px_ >= 0) & (px_ < W) & (py_ >= 0) & (py_ < H)
            if inside.mean() < 0.9:
                continue
            sup = float(m[py_[inside].astype(int), px_[inside].astype(int)].mean())
            if best is None or sup > best[0]:
                best = (sup, float(r), float(cx), float(cy))
        if best and best[0] >= 0.6:
            rs.append(best)
    if not rs:
        return None
    r = float(np.median([b[1] for b in rs]))
    return dict(px=r, check=1.0, check_ref=1.0, support=float(np.median([b[0] for b in rs])), n_frames=len(rs),
                px_spread=float(np.std([b[1] for b in rs]) / r), centre=[float(np.median([b[2] for b in rs])), float(np.median([b[3] for b in rs]))])


def court_ruler(vid, split, kind):
    path = next((p for p in (f"{R}/data/{split}/{vid}.mp4", f"{R}/data/{split}/validation_videos/{vid}.mp4") if os.path.exists(p)), None)
    fr = frames(path, a.nframes, a.maxdim) if path else []
    if not fr:
        return None
    H, W = fr[0].shape[:2]
    if kind == "soccer_circle":
        r = soccer_circle(fr)
    else:
        per = []
        for f in fr:
            L = court_lines([f])
            if L is None:
                continue
            x = {"tennis_singles": tennis_singles, "pickleball_width": pickleball_width, "basketball_key": basketball_key}[kind](L)
            if x is not None:
                per.append(x)
        if not per:
            return dict(ok=False, why="no_pattern", kind=kind)
        px = float(np.median([p["px"] for p in per]))
        r = dict(per[int(np.argmin([abs(p["px"] - px) for p in per]))])
        r.update(px=px, n_frames=len(per), px_spread=float(np.std([p["px"] for p in per]) / px),
                 check=float(np.nanmedian([p["check"] for p in per])))
    if r is None:
        return dict(ok=False, why="no_pattern", kind=kind)
    chk = r["check"] if np.isfinite(r["check"]) else 0.0
    ok = r.get("n_frames", 1) >= max(2, a.nframes // 3) and abs(math.log(max(chk, 1e-9) / r["check_ref"])) <= math.log(1.12)
    r.update(ok=bool(ok), kind=kind, W=W, H=H, frac_w=r["px"] / W)
    if a.debug:
        im = fr[len(fr) // 2].copy()
        if kind == "soccer_circle":
            cv2.circle(im, (int(r["centre"][0]), int(r["centre"][1])), int(r["px"]), (0, 0, 255), 2)
        else:
            t_ = math.radians(r["theta"])
            for c in r["lines"]:
                if abs(math.cos(t_)) >= 0.5:
                    pts = [(x, (c + x * math.sin(t_)) / math.cos(t_)) for x in (0, W - 1)]
                else:
                    pts = [((y * math.cos(t_) - c) / math.sin(t_), y) for y in (0, H - 1)]
                cv2.line(im, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]), (0, 0, 255), 2)
        sc = 900 / max(im.shape[:2])
        cv2.imwrite(f"{DBG}/{vid}_{kind}.jpg", cv2.resize(im, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return r


if __name__ == "__main__":
    sv = scene_videos(a.split)
    vids = a.vids.split(",") if a.vids else sorted(sv)
    out = {}
    for v in vids:
        kind, prx = sv.get(v, ("lane", r"\blanes?\b"))
        r = lane_ruler(v, a.split) if kind == "lane" else court_ruler(v, a.split, kind)
        if r is not None:
            r["prior_re"] = prx
        out[v] = r
        print(v, kind, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in (r or {}).items() if k not in ("lines", "lines_w", "bands_w")}), flush=True)
    fn = a.out or f"{R}/geo/out/scene_ruler_{a.split}.json"
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(out)
    json.dump(old, open(fn, "w"), indent=1)
