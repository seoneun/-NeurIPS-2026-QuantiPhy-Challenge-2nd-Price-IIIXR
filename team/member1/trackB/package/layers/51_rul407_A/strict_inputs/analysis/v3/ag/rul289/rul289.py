import json, os, sys, math
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/caw")
import measure_v3 as MV

SUB = sys.argv[1] if len(sys.argv) > 1 else "pred_t093"
RAWN = os.environ.get("FFACC_RAW", "geo_v4ntrj5sQ2srngsr_d_vw")
SI = {"m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001}
STRONG = ("2d_ratio", "3d_calib")
CX_TOL, DROP_FRAC, DROP_MIN = 3.0, 0.6, 3.0
MIN_N, R2_MIN, AX_FRAC, V0_MAX = 6, 0.99, 0.35, 2.0
GEO_TOL = float(os.environ.get("FFACC_GEO_TOL", "1.3"))
END_MARGIN = float(os.environ.get("FFACC_END_MARGIN", "0.5"))
OUT = os.environ.get("FFACC_OUT", "ag_rul289")


def load_vj(split, v):
    for d in (f"tracks_{split}_v2sc", f"tracks_{split}_v2", f"tracks_{split}"):
        p = f"{R}/geo/out/{d}/{v}.json"
        if os.path.exists(p):
            return json.load(open(p)), d
    return None, None


def walk(fr, cx, bot, s):
    ref = cx[s]; prev = s; dprev = None; out = None; mem = [s]; n = len(fr)

    def ok(p, k, dp):
        if fr[k] - fr[p] <= 0:
            return False, None
        d = (bot[k] - bot[p]) / (fr[k] - fr[p])
        if abs(cx[k] - ref) > CX_TOL or bot[k] < bot[p]:
            return False, d
        if dp is not None and dp >= DROP_MIN and d < DROP_FRAC * dp:
            return False, d
        return True, d

    k = s + 1
    while k < n:
        good, d = ok(prev, k, dprev)
        if good:
            mem.append(k); prev = k; dprev = d; k += 1; continue
        if out is None and k + 1 < n:
            good2, d2 = ok(prev, k + 1, dprev)
            if good2:
                out = k; mem.append(k + 1); prev = k + 1; dprev = d2; k += 2; continue
        return mem, k, out
    return mem, None, out


def rfit(x, y, deg=2):
    c = np.polyfit(x, y, deg)
    r = y - np.polyval(c, x)
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) <= 3 * max(mad, 0.5)
    if keep.sum() >= deg + 3 and keep.sum() < len(x):
        c = np.polyfit(x[keep], y[keep], deg)
    else:
        keep = np.ones(len(x), bool)
    rr = y[keep] - np.polyval(c, x[keep])
    sst = float(np.sum((y[keep] - y[keep].mean()) ** 2))
    r2 = 1 - float(np.sum(rr ** 2)) / max(sst, 1e-9)
    return c, r2, keep


def segment(fr, cx, bot, a_lo, a_hi):
    n = len(fr)
    js = [j for j in range(n) if a_lo - 0.5 <= fr[j] <= a_hi + 0.5]
    if not js:
        return None, "no_frames_in_window"
    j0, j1 = js[0], js[-1]
    best = None
    for s in range(j0, -1, -1):
        mem, contact, out = walk(fr, cx, bot, s)
        if mem[-1] >= j1 and all((j in mem) or j == out for j in range(j0, j1 + 1)):
            best = (s, mem, contact, out)
        else:
            break
    if best is None:
        return None, "window_not_free"
    s, mem, contact, out = best
    rest = [j for j in mem if j <= j0 and bot[j] - bot[s] <= 1.0]
    if len(rest) >= 3:
        mem = [j for j in mem if j >= rest[-1]]
    return dict(mem=mem, contact=contact, out=out), "ok"


def g_from_prior(pr, prior_text, target):
    pt = str(prior_text).lower()
    if pr and (pr.get("gravity") or "gravit" in pt):
        v = float(pr.get("value_si") or 0)
        return (v, "prior_g") if 9.0 <= v <= 10.5 else (None, f"other_gravity({v})")
    if pr and pr.get("kind") == "accel" and 9.0 <= float(pr.get("value_si") or 0) <= 10.5 \
            and str(pr.get("obj", "")).strip().lower() == str(target or "").strip().lower():
        return float(pr["value_si"]), "prior_fall"
    if "gravit" in pt:
        return None, "gravity_unparsed"
    return 9.8, "default"


