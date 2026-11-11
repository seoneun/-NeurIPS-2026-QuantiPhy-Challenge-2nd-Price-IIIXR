import math, re
import numpy as np

UNIT_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
PINGPONG_M = 0.040
TOL = math.log(1.10)


def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) and x > 0 else None
    except Exception:
        return None


def _q(r):
    return ((r.get("parse") or {}).get("q")) or {}


def _kind(r):
    k = _q(r).get("kind")
    return str(k if k else (r.get("geo") or {}).get("meas_kind"))


def L1_pingpong(r):
    if r.get("cat") != "S3" or _kind(r) != "size":
        return None
    q = r["question"].lower()
    if "diameter" not in q or not re.search(r"\bball\b", q):
        return None
    u = UNIT_M.get(str(_q(r).get("unit")))
    cs = _f(r.get("s1_cs"))
    fm = float(np.median(r["caw_full"])) if r.get("caw_full") else None
    if not (u and cs and fm and _f(r.get("cur"))):
        return None
    s = PINGPONG_M / u
    if abs(math.log(cs / s)) <= TOL and abs(math.log(fm / s)) <= TOL:
        return s, f"L1_pingpong regulation 40 mm (cs {cs}, full med {fm})"
    return None


def L2b_mogefov_speed(r):
    if r.get("cat") != "S3" or r.get("video_source") != "lab":
        return None
    if str((r.get("geo") or {}).get("meas_mode")) != "3d_mogefov" or _kind(r) != "speed":
        return None
    g, cs, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur) or abs(math.log(g / cs)) <= math.log(2):
        return None
    if not abs(math.log(cur / g)) < abs(math.log(cur / cs)):
        return None
    return math.sqrt(g * cs), f"L2b_mogefov_speed gm(g {g:.3g}, cs {cs:.3g})"


ACCEPTED = [("ag_r9_L1_pingpong", L1_pingpong), ("ag_r9_L2b_mogefov_speed", L2b_mogefov_speed)]
