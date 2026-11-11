import os, sys, json, math, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_DF = os.environ["QP_R"] + "/analysis/v3/ag/r10/work/r9/D_fusion/rules.py"
_s = importlib.util.spec_from_file_location("r17_dfusion", _DF); DF = importlib.util.module_from_spec(_s); _s.loader.exec_module(DF)
R16C = os.environ["QP_R"] + "/analysis/v3/ag/r16/OLDROUNDS/cache"

def _cache(unit, r, cd=None):
    cd = cd or os.environ.get("R17_CACHE", os.path.join(os.path.dirname(HERE), "cache"))
    p = f"{cd}/{unit}_{r['split']}_{int(r['id'])}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def _f(x):
    try:
        x = float(x); return x if math.isfinite(x) and x > 0 else None
    except Exception:
        return None

def plausible(r, v):
    b = DF.class_bound(r["question"], r["prior"], DF._kind(r), 2)
    if b is None: return True
    x = v * DF._usi(r["question"])
    return b[0] <= x <= b[1]

def U1_occ(r):
    c = _cache("occ", r)
    if not c: return None
    g = float(c["g_rep"]); cs = _f(r.get("s1_cs"))
    if cs and 0.5 <= g / cs <= 2: v, how = math.sqrt(g * cs), "agree_gm"
    elif c["mode"] == "3d_calib" or not cs: v, how = g, "dis_geo"
    else: v, how = cs, "dis_caw"
    if not plausible(r, v): return None
    return v, f"r17_occ: {c.get('why', '')} -> {how} {v:.4g}"

def U2_short(r):
    c = _cache("r16_rul148", r, cd=os.environ.get("R16_CACHE", R16C))
    if not c: return None
    v = float(c["value"])
    if not plausible(r, v): return None
    return v, "r17_short: " + str(c.get("why", ""))[:200]

def U3_tw(r):
    c = _cache("tw", r)
    if not c: return None
    v = float(c["value"])
    if not plausible(r, v): return None
    return v, f"r17_tw: {c.get('why', '')} ({c['info']})"

def _apx(r, sym):
    c = _cache("apx", r)
    if not c: return None
    h = float(c["h"]); cur = _f(r.get("cur"))
    if not cur: return None
    if sym:
        if abs(math.log(cur / h)) <= math.log(1 / 0.97): return None
    elif not cur < 0.97 * h: return None
    if not plausible(r, h): return None
    return h, f"r17_apx: {c.get('why', '')}; cur {cur:.2f}"
def U4_apx_V1(r): return _apx(r, False)
def U4_apx_V2(r): return _apx(r, True)

VARIANTS = {"U1_occ": U1_occ, "U2_short": U2_short, "U3_tw": U3_tw, "U4_apx_V1": U4_apx_V1, "U4_apx_V2": U4_apx_V2}
ORDER = ["U1_occ", "U2_short", "U3_tw", "U4_apx_V2"]

def U5_rest(r):
    c = _cache("rest", r)
    if not c: return None
    cur = _f(r.get("cur"))
    if not cur: return None
    src = str(r.get("s1_src", ""))
    if "grav" in src or "rul289" in src or abs(cur - 9.8) < 1e-6: return None
    alb, est = float(c["alb"]), float(c["est"])
    if cur >= alb: return None
    cs = _f(r.get("s1_cs"))
    if cs and max(est, cs) / min(est, cs) <= 2: v, how = max(math.sqrt(est * cs), alb), "gm"
    else: v, how = est, "geo"
    if not plausible(r, v): return None
    return v, f"r17_rest: {c.get('why', '')} -> {v:.3g} ({how}); cur {cur:.3g} < a_lb"

def U6_kch(r):
    c = _cache("kch", r)
    if not c: return None
    cur = _f(r.get("cur")); v = float(c["value"])
    if not cur or abs(math.log(cur / v)) <= math.log(1.08): return None
    if not plausible(r, v): return None
    return v, f"r17_kch: {c.get('why', '')}; cur {cur:.4g}"

def U7_gflb(r):
    c = _cache("gflb", r)
    if not c: return None
    cur = _f(r.get("cur")); v = float(c["vlb"])
    if not cur or not v > cur * 1.05: return None
    if not plausible(r, v): return None
    return v, f"r17_gflb: {c.get('why', '')}; cur {cur:.4g} below the floor"

VARIANTS.update({"U5_rest": U5_rest, "U6_kch": U6_kch, "U7_gflb": U7_gflb})
ORDER = ["U1_occ", "U2_short", "U3_tw", "U4_apx_V2", "U5_rest", "U6_kch", "U7_gflb"]

def U8_curb(r):
    c = _cache("curb", r)
    if not c: return None
    import re as _re
    if not _re.search(r"\b(width|wide)\b.*\b(lane|road)\b", r["question"], _re.I): return None
    cur = _f(r.get("cur")); v = float(c["value"])
    if not cur or abs(math.log(v / cur)) <= math.log(1.25): return None
    if not plausible(r, v): return None
    return v, f"r17_curb: {c.get('why', '')}; cur {cur:.4g}"

VARIANTS.update({"U8_curb": U8_curb})
ORDER = ["U1_occ", "U2_short", "U3_tw", "U4_apx_V2", "U5_rest", "U6_kch", "U7_gflb", "U8_curb"]
