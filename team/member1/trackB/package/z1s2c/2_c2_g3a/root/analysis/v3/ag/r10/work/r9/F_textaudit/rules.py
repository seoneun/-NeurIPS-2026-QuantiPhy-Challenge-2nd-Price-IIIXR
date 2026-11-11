import math, re

_CLS = [
    (r"\b(car|sedan|suv|taxi)\b", r"\b(toy|cart|model|car door|tire|wheel|window)\b", {"length": 4.5, "width": 1.8, "height": 1.5}),
    (r"\bbus\b", r"\b(toy|model)\b", {"length": 12.0, "width": 2.55, "height": 3.2}),
    (r"\b(person|man|woman|human|pedestrian)\b", r"\b(toy|model|snow)\b", {"height": 1.7}),
    (r"\bdoor\b", r"\b(car|train|bus|cabinet|fridge|window)\b", {"height": 2.0, "width": 0.9}),
    (r"\bbasketball\b", r"\b(hoop|board|backboard|court|rim)\b", {"diameter": 0.24}),
    (r"\b(soccer ?ball|football)\b", r"\b(field|goal|pitch)\b", {"diameter": 0.22}),
    (r"\bvolleyball\b", r"\b(court|net)\b", {"diameter": 0.21}),
    (r"\bbowling ball\b", None, {"diameter": 0.218}),
    (r"\bbaseball\b", r"\b(bat|field)\b", {"diameter": 0.074}),
    (r"\btennis ball\b", r"\btable\b", {"diameter": 0.067}),
    (r"\bbilliard ball\b", None, {"diameter": 0.057}),
    (r"\bgolf ball\b", None, {"diameter": 0.043}),
    (r"\b(ping ?pong|table tennis) ball\b", None, {"diameter": 0.040}),
    (r"\bcredit card\b", None, {"length": 0.0856, "width": 0.054}),
    (r"\b(bicycle|bike) wheel\b", None, {"diameter": 0.68}),
    (r"\blane\b", None, {"width": 3.5}),
    (r"\bpizza\b", r"\b(cutter|box|slice)\b", {"diameter": 0.30}),
    (r"\bkeyboard\b", r"\b(piano|key)\b", {"length": 0.44, "width": 0.14}),
    (r"\bpen\b", None, {"length": 0.14}),
    (r"\bpencil\b", r"\b(case|bag)\b", {"length": 0.19}),
    (r"\bwine glass\b", None, {"diameter": 0.08, "height": 0.20}),
    (r"\b(smart ?phone|phone|cellphone)\b", r"\b(booth|telephone)\b", {"length": 0.15, "width": 0.072}),
    (r"\beagle\b", None, {"length": 0.85, "wingspan": 2.0}),
    (r"\bhorse\b", r"\b(toy|model|rocking)\b", {"height": 1.6, "length": 2.4}),
    (r"\bforklift\b", r"\b(toy|model)\b", {"height": 2.1, "length": 2.8, "width": 1.2}),
    (r"\btraffic cone\b", None, {"height": 0.7}),
    (r"\bchess ?board\b", r"\bchinese\b", {"side": 0.45, "length": 0.45, "width": 0.45}),
    (r"\b(stair step|first step|step)\b", None, {"height": 0.17}),
]
_UF = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def _head(t):
    t = (t or "").lower()
    t = re.split(r"\b(?:of|on|in|at|under|above|near|next to|placed|that|which|where|with)\b", t)[0]
    return t.strip()


def _typ(row):
    p = row.get("parse") or {}
    q = p.get("q") or {}
    if q.get("kind") != "size":
        return None
    unit = (q.get("unit") or "m").lower()
    if unit not in _UF:
        return None
    dim = q.get("dim")
    h = _head(q.get("target"))
    if not h:
        return None
    for pat, exc, dims in _CLS:
        if re.search(pat, h) and not (exc and re.search(exc, h)):
            if dim in dims:
                return dims[dim] * _UF[unit]
            return None
    return None


def _med(x):
    x = sorted(v for v in (x or []) if v and v > 0 and math.isfinite(v))
    if not x:
        return None
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def _fin(v):
    try:
        v = float(v)
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None




def _cls_common(row, mode):
    typ = _typ(row)
    if typ is None:
        return None
    s = row["cur"]
    if not s or s <= 0:
        return None
    if abs(math.log(s / typ)) <= math.log(2.0):
        return None
    lo, hi = typ / 1.5, typ * 1.5
    if mode == "caw":
        c = _med(row.get("caw_full"))
        if c and lo <= c <= hi:
            return c, f"cls_caw typ={typ:g}"
        return None
    cs = _cands(row)
    best = min(cs.items(), key=lambda kv: abs(math.log(kv[1] / typ)), default=None)
    if best and lo <= best[1] <= hi:
        return best[1], f"cls_near:{best[0]} typ={typ:g}"
    if mode == "clamp":
        return (typ * 2.0 if s > typ else typ / 2.0), f"cls_clamp typ={typ:g}"
    return None


def cls_caw(row): return _cls_common(row, "caw")
def cls_near(row): return _cls_common(row, "near")
def cls_clamp(row): return _cls_common(row, "clamp")


_LK = {"size", "dist2", "disp", "path", "maxheight", "other"}


def _echo_common(row, mode):
    p = row.get("parse") or {}
    q = p.get("q") or {}
    pr = p.get("priors") or []
    if not pr or pr[0].get("kind") != "size" or not _fin(pr[0].get("value_si")):
        return None
    if q.get("kind") not in _LK:
        return None
    unit = (q.get("unit") or "m").lower()
    if unit not in _UF:
        return None
    pv = pr[0]["value_si"] * _UF[unit]
    s = row["cur"]
    if not s or s <= 0:
        return None
    mults = (1.0, 2.0, 0.5) if mode == "x2" else (1.0,)
    hit = [m for m in mults if abs(math.log(s / (pv * m))) < 0.005]
    if not hit:
        return None
    same = (q.get("kind") == "size" and q.get("dim") == pr[0].get("dim")
            and (q.get("target") or "").strip().lower() == (pr[0].get("obj") or "").strip().lower())
    if same:
        return None
    g = _fin((row.get("geo") or {}).get("geo_value"))
    if g and abs(math.log(g / s)) > 1e-6:
        return g, f"echo_geo prior={pv:g}x{hit[0]:g}"
    if mode == "geo":
        return None
    echo_vals = [pv * m for m in mults]
    rest = [v for v in (row.get("caw_full") or []) if v > 0 and all(abs(math.log(v / e)) > 0.02 for e in echo_vals)]
    if len(rest) >= 4:
        return _med(rest), f"echo_caw_nonecho prior={pv:g}x{hit[0]:g} n={len(rest)}"
    return None


def echo_geo(row): return _echo_common(row, "geo")
def echo_geo_caw(row): return _echo_common(row, "geo_caw")
def echo_x2(row): return _echo_common(row, "x2")


ALL = {"cls_caw": cls_caw, "cls_near": cls_near, "cls_clamp": cls_clamp,
       "echo_geo": echo_geo, "echo_geo_caw": echo_geo_caw, "echo_x2": echo_x2}



ACCEPTED = [("ag_r9_F_cls_caw", cls_caw), ("ag_r9_F_echo_geo_caw", echo_geo_caw)]


