import os, re, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(os.path.dirname(HERE), "cache")


def _fin(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if np.isfinite(v) and v > 0 else None


NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")


def _rigid(r, srcs, calib3d):
    vid = str(r.get("video_id") or ""); src = vid.split("_")[0]
    if src not in srcs: return None
    q = ((r.get("parse") or {}).get("q") or {})
    if q.get("kind") not in ("speed", "accel", "disp"): return None
    if NONRIGID.search(str(r.get("question") or "").lower()): return None
    s0 = str(r.get("s1_src") or "")
    if s0 != "agree_gm": return None
    g, cs, c = _fin(r.get("s1_g")), _fin(r.get("s1_cs")), _fin(r.get("cur"))
    if g is None or cs is None or c is None: return None
    if not (0.5 <= g / cs <= 2.0): return None
    if abs(math.log(c / math.sqrt(g * cs))) > 1e-6: return None
    if calib3d and str(r.get("cat", "")).endswith("3"):
        md = str((r.get("geo") or {}).get("meas_mode") or "")
        if not md.startswith("3d_calib"): return None
    if abs(math.log(g / c)) <= 1e-12: return None
    return g, f"U1 RIGID: own {src} {q.get('kind')} row, rigid target, own geo {g:.4g} and own CaW {cs:.4g} agree (agree_gm {c:.4g}) -> own geo"


SIM = ("simulation",)
def U1_V1(r): return _rigid(r, SIM, False)
def U1_V2(r): return _rigid(r, SIM + ("internet",), False)
def U1_V3(r): return _rigid(r, SIM, True)


VARIANTS = {"U1_V1": U1_V1, "U1_V2": U1_V2, "U1_V3": U1_V3}
ORDER = ["U1_V1"]


UNIT_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
def _q(r): return ((r.get("parse") or {}).get("q")) or {}
def _kind(r):
    k = _q(r).get("kind")
    return str(k if k else (r.get("geo") or {}).get("meas_kind"))


def U2_V1(r):
    if _kind(r) != "size": return None
    q = str(r.get("question") or "").lower()
    if "diameter" not in q or not re.search(r"\bball\b", q): return None
    u = UNIT_M.get(str(_q(r).get("unit"))); cs = _fin(r.get("s1_cs"))
    fm = float(np.median(r["caw_full"])) if r.get("caw_full") else None
    c = _fin(r.get("cur"))
    if not (u and cs and fm and c): return None
    s = 0.040 / u; tol = math.log(1.10)
    if abs(math.log(cs / s)) <= tol and abs(math.log(fm / s)) <= tol and abs(math.log(c / s)) > 1e-12:
        return s, f"U2 PINGPONG_ALL: regulation 40 mm (own cs {cs}, own full median {fm})"
    return None


def U3_V1(r):
    if _q(r).get("kind") != "size": return None
    g_ = r.get("geo") or {}
    if "tdepth=prior" not in str(g_.get("meas_mode")): return None
    g = _fin(g_.get("geo_value")) if bool(g_.get("meas_valid")) else None
    c, cur = _fin(r.get("s1_cs")), _fin(r.get("cur"))
    if g is None or c is None or cur is None or abs(math.log(cur / c)) < 0.01: return None
    if abs(math.log(g / c)) > math.log(1.25): return None
    return c, f"U3 TDP_ALL: geo borrowed the prior depth (tdepth=prior), own geo {g:.4g} within 1.25x of own CaW -> own CaW {c:.4g}"


def U4_V1(r):
    if "rul202" not in str(r.get("s1_src")): return None
    k = _q(r).get("kind")
    if k not in ("speed", "avgspeed", "accel", "disp"): return None
    g, cur = _fin(r.get("s1_g")), _fin(r.get("cur"))
    if not (g and cur) or abs(math.log(cur / g)) < 1e-9: return None
    if str((r.get("geo") or {}).get("merge_src")) != "sruler": return None
    return g, f"U4 S2EDGE_ALL: {k} scaled by a scene ruler; target border-extent ratio does not enter motion -> own geo {g:.4g} (was {cur:.4g})"


VARIANTS.update({"U2_V1": U2_V1, "U3_V1": U3_V1, "U4_V1": U4_V1})
ORDER = ["U1_V1", "U2_V1", "U3_V1", "U4_V1"]


def _bracket(r, sizeonly, valid):
    k = _q(r).get("kind")
    if sizeonly and k != "size": return None
    g, cs, c = _fin(r.get("s1_g")), _fin(r.get("s1_cs")), _fin(r.get("cur"))
    if g is None or cs is None or c is None: return None
    if valid and not bool((r.get("geo") or {}).get("meas_valid")): return None
    if abs(math.log(g / cs)) > math.log(1.10): return None
    lo, hi = min(g, cs) * 0.97, max(g, cs) / 0.97
    if lo <= c <= hi: return None
    v = math.sqrt(g * cs)
    return v, f"U5 BRACKET: own geo {g:.4g} and own CaW {cs:.4g} agree within 10%, answer {c:.4g} outside [{lo:.4g}, {hi:.4g}] -> gm {v:.4g}"


def U5_V1(r): return _bracket(r, True, False)
def U5_V2(r): return _bracket(r, False, False)
def U5_V3(r): return _bracket(r, False, True)


VARIANTS.update({"U5_V1": U5_V1, "U5_V2": U5_V2, "U5_V3": U5_V3})
ORDER = ["U1_V1", "U5_V1"]
