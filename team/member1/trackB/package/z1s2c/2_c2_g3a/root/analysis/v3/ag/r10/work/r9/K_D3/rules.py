import math
import numpy as np

def _f(x):
    try:
        x = float(x)
        return abs(x) if np.isfinite(x) and x != 0 else None
    except Exception:
        return None

def _d3(r):
    return str(r.get("inference_type", ""))[:1] == "D" and str(r.get("video_type", ""))[1:2] == "3"

def _q(r):
    return ((r.get("parse") or {}).get("q") or {})

def _geo(r):
    g = r.get("geo") or {}
    if not bool(g.get("meas_valid")):
        return None
    return _f(g.get("geo_value"))

def _mode(r):
    return str((r.get("geo") or {}).get("meas_mode"))

def _u1(r, thr, w):
    if not _d3(r) or _q(r).get("kind") != "size":
        return None
    if "tdepth=prior" not in _mode(r):
        return None
    g, c, cur = _geo(r), _f(r.get("s1_cs")), _f(r.get("cur"))
    if g is None or c is None or cur is None or abs(math.log(cur / c)) < 0.01:
        return None
    if abs(math.log(g / c)) > math.log(thr):
        return None
    return (g ** w * c ** (1 - w), "tdp_agree")

def U1_V1(r): return _u1(r, 1.25, 0.0)
def U1_V2(r): return _u1(r, 1.5, 0.0)
def U1_V3(r): return _u1(r, 1.25, 0.25)

def _u2(r, var):
    if not _d3(r) or _q(r).get("kind") != "disp":
        return None
    g, c, cur = _geo(r), _f(r.get("s1_cs")), _f(r.get("cur"))
    if g is None or c is None or cur is None or not _mode(r).startswith(("3d", "vlm_3d")):
        return None
    if abs(math.log(cur / c)) > 0.01 or abs(math.log(g / c)) <= math.log(1.25):
        return None
    if var == 1:
        return (g, "disp_geo")
    if var == 3:
        s = [x for x in r.get("caw_full") or [] if x > 0]
        if len(s) < 4 or math.log(np.percentile(s, 75) / np.percentile(s, 25)) <= 0.3:
            return None
    return (math.sqrt(g * c), "disp_gm")

def U2_V1(r): return _u2(r, 1)
def U2_V2(r): return _u2(r, 2)
def U2_V3(r): return _u2(r, 3)

_USI = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "km": 0.001}


def _u3(r, use_geo, use_caw):
    if not _d3(r):
        return None
    p = r.get("parse") or {}
    q = p.get("q") or {}
    if q.get("kind") != "dist2":
        return None
    k1, k2 = p.get("target_dkeys") or [], p.get("target2_dkeys") or []
    if not k1 or not k2 or set(k1) & set(k2):
        return None
    u = _USI.get(q.get("unit"))
    dA, dB = _dep(p, k1, q.get("t")), _dep(p, k2, q.get("t"))
    if u is None or dA is None or dB is None:
        return None
    lb = abs(dA - dB) * u
    c, cur = _f(r.get("s1_cs")), _f(r.get("cur"))
    if lb <= 0 or c is None or cur is None:
        return None
    if abs(c / lb - 1) > 0.02 or abs(cur / lb - 1) > 0.02:
        return None
    g = _geo(r)
    if g is not None:
        return (g, "ddecho_geo") if (use_geo and g >= lb) else None
    if use_caw:
        s = [x for x in r.get("caw_full") or [] if x > 0 and abs(x / lb - 1) > 0.02]
        if len(s) >= 4 and float(np.median(s)) >= lb:
            return (float(np.median(s)), "ddecho_caw")
    return None

def U3_V1(r): return _u3(r, True, False)
def U3_V2(r): return _u3(r, False, True)
def U3_V3(r): return _u3(r, True, True)

ALL = {"U1_V1": U1_V1, "U1_V2": U1_V2, "U1_V3": U1_V3, "U2_V1": U2_V1, "U2_V2": U2_V2, "U2_V3": U2_V3,
       "U3_V1": U3_V1, "U3_V2": U3_V2, "U3_V3": U3_V3}

ACCEPTED = [("ag_r9_K_tdp_agree", U1_V1)]
