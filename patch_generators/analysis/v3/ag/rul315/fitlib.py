"""rul315: robust free-fall pixel scale from the vertical component of a track (label-free).

A body in free flight has y(t) = y0 + vy0 t + a/2 t^2 in the image (y down) with a = g / mpp when its depth is constant
(planar motion), whatever its horizontal motion (projectile: only the vertical component carries g). The pipeline's
freefall() needs a changepoint segment >= 0.15 s with residual <= 1.5 px; it misses clean drops that are short (a 60 fps
droplet: 0.08 s pieces after a duplicated frame), whose centroid is biased by border clipping / motion blur (a falling
sword entering the frame), or whose video has a dropped-frame cadence (internet re-encodes, sawtooth timing).

Here, per signal (box bottom = leading edge of a downward fall, box top, centroid), the longest window of consecutive
frames is searched whose robust quadratic (iterative 3-MAD clipping, <= 25 % outliers) is clean:
  R^2 >= R2MIN on the kept points, t-stat of the curvature >= TMIN, a_y > 0, vertical travel >= max(1.5 sizes, 30 px),
  second differences positive in >= 60 % of steps (monotone acceleration sign), lateral acceleration <= 0.35 a_y,
  halves consistent (n >= 12: both half-window curvatures within [0.6, 1.6] a), object size stable (p90/p10 <= 1.35).
A scale is accepted when >= 2 signals pass and agree within 25 % (median taken), or one signal passes with R^2 >= 0.999
and >= 8 kept points.
"""
import numpy as np

R2MIN = 0.995
TMIN = 8.0


def robust_quad(t, y, clip=3.0, max_out=0.25):
    n = len(t)
    tm = t.mean()
    tt = t - tm
    keep = np.ones(n, bool)
    c = None
    for _ in range(6):
        if keep.sum() < 5:
            return None
        c = np.polyfit(tt[keep], y[keep], 2)
        r = y - np.polyval(c, tt)
        s = max(1.4826 * float(np.median(np.abs(r[keep]))), 0.5)
        new = np.abs(r) <= clip * s
        if new.sum() < 5:
            return None
        if np.array_equal(new, keep):
            break
        keep = new
    if (~keep).sum() > max_out * n:
        return None
    c = np.polyfit(tt[keep], y[keep], 2)
    r = y[keep] - np.polyval(c, tt[keep])
    sst = float(np.sum((y[keep] - y[keep].mean()) ** 2))
    r2 = 1 - float(r @ r) / max(sst, 1e-9)
    dof = max(int(keep.sum()) - 3, 1)
    s_r = float(np.sqrt(r @ r / dof))
    X = np.stack([tt[keep] ** 2, tt[keep], np.ones(keep.sum())], 1)
    try:
        se = s_r * float(np.sqrt(np.linalg.inv(X.T @ X)[0, 0]))
    except np.linalg.LinAlgError:
        se = np.inf
    return dict(c=c, tm=tm, keep=keep, r2=r2, s_r=s_r, a=2 * c[0], tstat=c[0] / max(se, 1e-12))


def window_ok(t, y, x, sz, dt, size_px):
    f = robust_quad(t, y)
    if f is None:
        return None, "nofit"
    k = f["keep"]
    if f["a"] <= 0:
        return None, "a<=0"
    if f["tstat"] < TMIN:
        return None, f"tstat({f['tstat']:.1f})"
    if f["r2"] < R2MIN:
        return None, f"r2({f['r2']:.4f})"
    yf = np.polyval(f["c"], t[k] - f["tm"])
    travel = float(yf.max() - yf.min())
    if travel < max(1.5 * size_px, 30.0):
        return None, f"travel({travel:.0f})"
    tk, yk = t[k], y[k]
    d2 = []
    for i in range(1, len(tk) - 1):
        h1, h2 = tk[i] - tk[i - 1], tk[i + 1] - tk[i]
        if abs(h1 - h2) < 0.25 * min(h1, h2):
            d2.append((yk[i + 1] - yk[i]) / h2 - (yk[i] - yk[i - 1]) / h1)
    if len(d2) >= 3 and np.mean(np.array(d2) > 0) < 0.6:
        return None, "acc_sign"
    ax = 2 * np.polyfit(tk - f["tm"], x[k], 2)[0]
    if abs(ax) > 0.35 * f["a"]:
        return None, f"lateral({ax:.0f}/{f['a']:.0f})"
    n = int(k.sum())
    if n >= 12:
        h = n // 2
        a1 = 2 * np.polyfit(tk[:h] - tk[:h].mean(), yk[:h], 2)[0]
        a2 = 2 * np.polyfit(tk[h:] - tk[h:].mean(), yk[h:], 2)[0]
        if not (0.6 * f["a"] <= a1 <= 1.6 * f["a"] and 0.6 * f["a"] <= a2 <= 1.6 * f["a"]):
            return None, f"halves({a1:.0f},{a2:.0f} vs {f['a']:.0f})"
    if sz is not None:
        s = sz[k]
        s = s[np.isfinite(s) & (s > 0)]
        if len(s) >= 4 and np.percentile(s, 90) / np.percentile(s, 10) > 1.35:
            return None, "size_unstable"
    f.update(n=n, travel=travel, ax=ax, t0=float(t[0]), t1=float(t[-1]))
    return f, "ok"


def best_window(t, y, x, sz, dt, size_px, min_pts=6):
    """longest window of consecutive frames (no gap > 1.6 dt) with a clean parabola"""
    n = len(t)
    if n < min_pts:
        return None, "few"
    # split at time gaps
    cuts = [0] + [i + 1 for i in range(n - 1) if t[i + 1] - t[i] > 1.6 * dt] + [n]
    best, why_last = None, "none"
    for a, b in zip(cuts[:-1], cuts[1:]):
        for L in range(b - a, min_pts - 1, -1):
            if best is not None and (t[a + L - 1] - t[a]) < (best["t1"] - best["t0"]) - 1e-9:
                break
            found = None
            for i in range(a, b - L + 1):
                j = i + L
                f, w = window_ok(t[i:j], y[i:j], x[i:j], None if sz is None else sz[i:j], dt, size_px)
                if f is not None and (found is None or f["r2"] > found["r2"]):
                    f["i"], f["j"] = i, j
                    found = f
                elif f is None:
                    why_last = w
            if found is not None:
                if best is None or (found["t1"] - found["t0"]) > (best["t1"] - best["t0"]) + 1e-9:
                    best = found
                break
    return (best, "ok") if best is not None else (None, why_last)


def track_signals(tr, H):
    """centroid / bottom edge / top edge series of a measure_v3 Track (clipped edges dropped)"""
    okc = list(tr.okc)
    t = np.asarray(tr.t, float)
    uv = np.asarray(tr.uv, float)
    box = np.array([tr.st[j]["box"] for j in okc], float)
    rr0 = np.array([tr.st[j]["rr"][0] for j in okc], float)
    sig = {"c": (t, uv[:, 1], uv[:, 0], rr0)}
    mb = box[:, 3] < H - 1.5
    mt = box[:, 1] > 1.5
    xc = (box[:, 0] + box[:, 2]) / 2
    sig["bot"] = (t[mb], box[mb, 3], xc[mb], rr0[mb])
    sig["top"] = (t[mt], box[mt, 1], xc[mt], rr0[mt])
    return sig


def fall_scale(tr, H, fps):
    """-> (a_px, info) or (None, info)"""
    dt = 1.0 / fps
    size_px = float(getattr(tr, "size_px", 0.0) or 0.0)
    res = {}
    for k, (t, y, x, sz) in track_signals(tr, H).items():
        f, w = best_window(t, y, x, sz, dt, size_px)
        res[k] = (f, w)
    ok = {k: f for k, (f, w) in res.items() if f is not None}
    info = {k: (round(f["a"], 1), f["n"], round(f["r2"], 5), round(f["t0"], 3), round(f["t1"], 3)) if f is not None else w
            for k, (f, w) in res.items()}
    if not ok:
        return None, info
    av = np.array([f["a"] for f in ok.values()])
    if len(ok) >= 2:
        # the signals that agree with the median within 25 %
        med = float(np.median(av))
        agree = [k for k, f in ok.items() if abs(f["a"] / med - 1) <= 0.25]
        if len(agree) >= 2:
            a = float(np.median([ok[k]["a"] for k in agree]))
            info["used"] = agree
            info["dur"] = max(ok[k]["t1"] - ok[k]["t0"] for k in agree)
            return a, info
        info["why"] = "signals_disagree"
        return None, info
    k, f = next(iter(ok.items()))
    if f["r2"] >= 0.999 and f["n"] >= 8:
        info["used"] = [k]
        info["dur"] = f["t1"] - f["t0"]
        return float(f["a"]), info
    info["why"] = "single_signal_weak"
    return None, info
