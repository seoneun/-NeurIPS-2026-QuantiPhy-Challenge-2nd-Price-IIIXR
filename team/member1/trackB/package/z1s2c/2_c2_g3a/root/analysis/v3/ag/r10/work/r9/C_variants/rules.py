import math, statistics

def _seg(r):
    return str(r.get("video_source", "")).startswith("segment")

def _kind(r):
    p = r.get("parse") or {}
    return (p.get("q") or {}).get("kind")

def _cawmed(r):
    x = [v for v in (r.get("caw_full") or []) if v and v > 0]
    return statistics.median(x) if len(x) >= 4 else None

def _u1(r, thr, blend):
    if not _seg(r) or str(r["video_type"])[3:4] not in ("C", "S"): return None
    if str(r["inference_type"])[:1] != "S" or _kind(r) != "size": return None
    c = _cawmed(r); cur = r["cur"]
    if c is None or not cur or cur <= 0: return None
    if abs(math.log(cur / c)) <= math.log(thr): return None
    v = math.sqrt(cur * c) if blend else c
    return v, f"segmented real-footage clip, size prior + size question: own pixel ratio {cur:.3g} vs own CaW {c:.3g} >{thr}x apart -> {'log-mean' if blend else 'own CaW median'}"

def u1_V1a(r): return _u1(r, 2.0, False)
def u1_V1b(r): return _u1(r, 1.5, False)
def u1_V1c(r): return _u1(r, 2.0, True)

def _u2(r, caw_only):
    if not _seg(r) or str(r["video_type"])[3:4] != "X": return None
    g = r.get("geo") or {}
    try:
        gv = float(g.get("meas_value"))
    except Exception:
        return None
    if not (gv > 0) or not bool(g.get("meas_valid")) or g.get("meas_mode") not in ("2d_ratio", "3d_calib"): return None
    if caw_only and "caw" not in str(r.get("s1_src", "")): return None
    cur = r["cur"]
    if abs(math.log(cur / gv)) <= math.log(1.25): return None
    return gv, "background-removed simulated clip: own strong geo preferred over a VLM-led answer"

def u2_V2a(r): return _u2(r, False)
def u2_V2b(r): return _u2(r, True)

ACCEPTED = [("segbg_size_caw", u1_V1b)]
