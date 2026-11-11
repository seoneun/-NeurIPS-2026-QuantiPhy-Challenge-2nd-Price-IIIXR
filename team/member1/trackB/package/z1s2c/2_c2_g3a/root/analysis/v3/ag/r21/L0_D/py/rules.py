import os, json, math
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("R21_CACHE", f"{HERE}/cache")
_C = {}


def _cache(unit, r):
    k = (CACHE, unit, r["split"], int(r["id"]))
    if k not in _C:
        p = f"{CACHE}/{unit}_{r['split']}_{int(r['id'])}.json"
        _C[k] = json.load(open(p)) if os.path.exists(p) else None
    return _C[k]


def _f(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) and x > 0 else None


def L0D_labf(r):
    c = _cache("labf", r)
    if not c: return None
    return c["meas"], c.get("why", "")


def L0D_ngpropf(r):
    c = _cache("ngpropf", r)
    if not c: return None
    new = c["meas"]; cs = _f(r.get("s1_cs")); k = c.get("kind")
    if cs is None or k in ("dist2", "avgspeed") or not (0.5 <= new / cs <= 2.0):
        v, how = new, "geo"
    else:
        v, how = math.sqrt(new * cs), "gm with own CaW %.4g" % cs
    return v, c.get("why", "") + " -> " + how


def L0D_texper(r):
    c = _cache("texper", r)
    if not c: return None
    g = c["meas"]; cs = _f(r.get("s1_cs"))
    if cs is not None and abs(math.log10(g / cs)) <= math.log10(2):
        return math.sqrt(g * cs), c.get("why", "") + "; gm with own CaW %.4g" % cs
    return g, c.get("why", "") + "; geo (CaW outside 2x)"


ORDER = [("L0D_labf", L0D_labf), ("L0D_ngpropf", L0D_ngpropf), ("L0D_texper", L0D_texper)]
ACCEPTED = [L0D_labf, L0D_ngpropf, L0D_texper]
