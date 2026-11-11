import os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))

def _cache(r):
    p = f"{HERE}/cache/mot_{r['split']}_{r['id']}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def _apply(r, keff):
    c = _cache(r)
    if c is None or c.get("g_box") is None or c.get("g_pt") is None: return None
    g = r.get("s1_g")
    if g is None or not g > 0 or abs(math.log(c["g_box"] / g)) >= 0.01: return None
    k = c["g_pt"] / c["g_box"]
    ke = keff(k, c)
    if ke is None: return None
    cur = r["cur"]
    if abs(math.log(cur / g)) < 0.01:
        w = 1.0
    else:
        cs = r.get("s1_cs")
        if cs is None or not cs > 0 or abs(math.log(g / cs)) < 0.05: return None
        w = math.log(cur / cs) / math.log(g / cs)
        if not (0.05 <= w <= 1.01): return None
        w = min(w, 1.0)
    return cur * ke ** w, f"motion k={k:.3f} keff={ke:.3f} w={w:.2f} J={c.get('J')}"

def V1_jumpy(r):
    return _apply(r, lambda k, c: k if (c.get("J") is not None and c["J"] >= 2.0 and abs(math.log(k)) >= 0.1) else None)

def V2_gm(r):
    return _apply(r, lambda k, c: math.sqrt(k) if abs(math.log(k)) >= 0.05 else None)

def V3_full(r):
    return _apply(r, lambda k, c: k if abs(math.log(k)) >= 0.05 else None)

VARIANTS = [("ag_r10_MOTION_V1_jumpy", V1_jumpy), ("ag_r10_MOTION_V2_gm", V2_gm), ("ag_r10_MOTION_V3_full", V3_full)]
ACCEPTED = []
