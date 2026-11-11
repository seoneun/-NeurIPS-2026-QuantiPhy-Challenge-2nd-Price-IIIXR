import os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
_C = {}

def meas(r):
    k = (r["split"], int(r["id"]))
    if k not in _C:
        p = f"{HERE}/cache/meas_{k[0]}_{k[1]}.json"
        _C[k] = json.load(open(p)) if os.path.exists(p) else None
    m = _C[k]
    return m if m and m.get("ok") and m.get("value") else None

def make_v2(cls):
    def fn(r):
        if r.get("s1_g") is not None: return None
        m = meas(r)
        if m is None or m.get("cls") != cls: return None
        a, g = float(r["cur"]), float(m["value"])
        if a > 0 and g > 0 and abs(math.log(g / a)) <= math.log(2):
            return math.sqrt(a * g), f"gm_{cls}: own re-measured geo {g:.4g} (mode {m.get('mode')}) within 2x -> gm"
        return None
    fn.__name__ = f"GM_{cls}_V2"
    return fn

def make_v3(cls):
    def fn(r):
        if r.get("s1_g") is not None: return None
        m = meas(r)
        if m is None or m.get("cls") != cls: return None
        return float(m["value"]), f"gm_{cls}: own re-measured geo (replace)"
    fn.__name__ = f"GM_{cls}_V3"
    return fn

GM_T_V2 = make_v2("T"); GM_T_V3 = make_v3("T"); GM_G_V2 = make_v2("G"); GM_G_V3 = make_v3("G")
VARIANTS = [("ag_r10_GM_T_V2", GM_T_V2), ("ag_r10_GM_T_V3", GM_T_V3), ("ag_r10_GM_G_V2", GM_G_V2), ("ag_r10_GM_G_V3", GM_G_V3)]
ACCEPTED = [("ag_r10_GM_T_V2", GM_T_V2)]
