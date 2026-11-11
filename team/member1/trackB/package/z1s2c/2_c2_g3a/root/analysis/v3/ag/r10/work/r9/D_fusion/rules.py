import math, re
import numpy as np

THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
STRONG = ("2d_ratio", "3d_calib")
CAWONLY = ("caw_nogeo", "dis_caw_routed", "caw_c1")


def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) and x > 0 else None
    except Exception:
        return None


def _kind(r):
    p = r.get("parse") or {}
    k = (p.get("q") or {}).get("kind") if p else None
    if not k:
        k = str((r.get("geo") or {}).get("meas_kind"))
    return str(k)


def _mode(r):
    return str((r.get("geo") or {}).get("meas_mode"))


def _med(a):
    return float(np.median(a)) if len(a) else None


def _lsd(a):
    a = [x for x in a if x and x > 0]
    if len(a) < 4:
        return None
    return math.log(np.percentile(a, 75) / np.percentile(a, 25)) / 1.349


def _used(r):
    cs = _f(r.get("s1_cs")); cg = _f(r.get("caw_crop_greedy"))
    if r.get("caw_crop") and cs and cg and abs(math.log(cs / cg)) < 0.05:
        return r["caw_crop"]
    return r.get("caw_full") or []


def _dfc(r):
    mf, mc = _med(r.get("caw_full") or []), _med(r.get("caw_crop") or [])
    return math.log(mf / mc) if mf and mc else 0.0


_KC = {}




def _agree_rel(r, s0, size_only):
    g, cs, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur):
        return None
    if abs(math.log(cur / math.sqrt(g * cs))) > 1e-6 or abs(math.log10(g / cs)) > math.log10(2) + 1e-12:
        return None
    if _mode(r) not in STRONG:
        return None
    if size_only and _kind(r) != "size":
        return None
    L = math.log(10)
    sc = _lsd(_used(r))
    sc = (sc or 0.0) / L
    dfc = _dfc(r) / L
    gv, gu = _f((r.get("geo") or {}).get("geo_value")), _f((r.get("geo_u") or {}).get("geo_value"))
    dgu = math.log10(gv / gu) if gv and gu else 0.0
    vc = s0 ** 2 + sc ** 2 + dfc ** 2
    vg = s0 ** 2 + dgu ** 2
    w = vc / (vc + vg)
    if abs(w - 0.5) < 1e-3:
        return None
    return g ** w * cs ** (1 - w), f"agree_rel w_g={w:.2f} (caw sd {sc:.3f} dec, full/crop {dfc:.3f}, geo/geo_u {dgu:.3f})"


def U1a(r): return _agree_rel(r, 0.1, True)
def U1b(r): return _agree_rel(r, 0.1, False)
def U1c(r): return _agree_rel(r, 0.05, True)


def _caw_mra(r, var):
    cs, cur = _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (cs and cur) or abs(math.log(cur / cs)) > 1e-9:
        return None
    if _f(r.get("s1_g")) and str(r.get("s1_src")) not in CAWONLY:
        return None
    s = _lsd(_used(r))
    if s is None:
        return None
    if var in ("b", "c"):
        s = math.sqrt(s ** 2 + _dfc(r) ** 2)
    if var == "c":
        s = max(s, math.log(1.25))
    k = kopt(s)
    if abs(k - 1) < 1e-9:
        return None
    return cs * k, f"caw_mra sig={s:.3f} k={k:.3f}"


def U2a(r): return _caw_mra(r, "a")
def U2b(r): return _caw_mra(r, "b")
def U2c(r): return _caw_mra(r, "c")


B = r"\b"
PERSON = B + r"(?:person|pedestrian|man|woman|walker|people|player|boy|girl|kid|child|lady|guy|swimmer|runner)" + B
BALLS = [(r"tennis ball", 0.05, 0.09), (r"ping[ -]?pong ball", 0.03, 0.05), (r"table tennis ball", 0.03, 0.05), (r"golf ball", 0.035, 0.055),
         (r"billiard ball", 0.045, 0.07), (r"pool ball", 0.045, 0.07), (r"basketball", 0.15, 0.32), (r"soccer ball", 0.12, 0.30),
         (r"football", 0.12, 0.32), (r"volleyball", 0.15, 0.30), (r"baseball", 0.06, 0.09), (r"bowling ball", 0.18, 0.25),
         (r"yoga ball", 0.30, 1.10)]
SIZE3 = [(r"horse", "height|tall", 0.8, 2.7), (r"elephant", "height|tall", 1.4, 4.8), (r"dog", "height|tall", 0.12, 1.2),
         (r"cat", "height|tall", 0.12, 0.7), (r"chair", "height|tall", 0.35, 1.4), (r"sofa|couch", "height|tall", 0.4, 1.3),
         (r"bed", "length|long", 1.4, 2.6), (r"(?:water |wine |plastic |glass )?bottle", "height|tall|length", 0.04, 0.6),
         (r"(?:soda |coke |beer )?can", "height|tall", 0.05, 0.25), (r"(?:cup|mug)", "height|tall", 0.03, 0.35),
         (r"book", "length", 0.08, 0.5), (r"laptop", "width|length", 0.2, 0.5), (r"keyboard", "length", 0.2, 0.65),
         (r"bus", "length|long", 5.0, 25.0), (r"motorcycle|motorbike", "length|long", 1.3, 3.0), (r"(?:bicycle|bike)", "length|long", 1.0, 2.4)]
SPEED3 = [(r"horse", 0.05, 22.0), (r"dog", 0.05, 22.0), (r"cat", 0.02, 15.0), (r"(?:ball|tennis ball|football|basketball|soccer ball)", 0.01, 75.0),
          (r"(?:boat|ship|speedboat|sailboat)", 0.02, 45.0)]


def class_bound(q, prior, kind, level):
    ql = str(q).lower(); pl = str(prior).lower()
    if kind in ("speed", "avgspeed"):
        if re.search(PERSON, ql) and "ball" not in ql:
            if "swim" in ql:
                return 0.05, 2.5
            if "walk" in ql or "pedestrian" in ql:
                return 0.05, 3.0
            return 0.02, 11.0
    if kind == "accel":
        if re.search(B + r"(?:car|truck|bus|van|vehicle|forklift|bike|bicycle|motorcycle)" + B, ql) and "gravit" not in pl:
            return 0.01, 15.0
        if re.search(PERSON, ql):
            return 0.01, 12.0
    if kind == "size":
        if re.search(PERSON, ql) and re.search(B + r"(?:height|tall)" + B, ql) and not re.search(r"sitting|shoulder|hand|arm|leg|foot|head|baby|toddler", ql):
            return 0.5, 2.3
        if re.search(B + r"(?:diameter|size|width)" + B, ql) and not re.search(r"model|toy|cart|rim|hoop|basket|court|planet", ql):
            for pat, lo, hi in BALLS:
                if re.search(B + pat + B, ql):
                    return lo, hi
        if re.search(B + r"(?:tire|tyre|wheel)s?" + B, ql) and "diameter" in ql and not re.search(r"toy|model|steering|cart|robot|kart|rc |drone", ql):
            return 0.25, 2.0
        if re.search(B + r"car" + B, ql) and not re.search(r"truck|bus|van|toy|model|train|cart|race|lift|robot|kart|rc ", ql):
            if re.search(B + r"length|long" + B, ql):
                return 2.0, 7.0
            if re.search(B + r"width|wide" + B, ql):
                return 1.3, 2.8
            if re.search(B + r"height|tall" + B, ql):
                return 0.9, 2.6
        if level >= 2 and not re.search(r"toy|model|miniature|dollhouse|lego", ql):
            if re.search(B + r"door" + B, ql) and re.search(B + r"height|tall" + B, ql) and not re.search(r"car|cabinet|fridge|oven|cage|truck|bus", ql):
                return 1.6, 3.5
            if re.search(B + r"(?:table|desk)" + B, ql) and re.search(B + r"height" + B, ql) and not re.search(r"leg|top|surface thick|lamp", ql):
                return 0.25, 1.4
            if re.search(B + r"(?:iphone|smartphone|phone|cellphone|mobile phone)" + B, ql) and not re.search(r"booth|box|stand", ql):
                if re.search(B + r"width" + B, ql):
                    return 0.045, 0.13
                if re.search(B + r"(?:length|height)" + B, ql):
                    return 0.09, 0.22
    if level >= 2 and kind in ("speed", "avgspeed") and not re.search(r"toy|model|rc |remote", ql):
        if re.search(B + r"car" + B, ql) and not re.search(r"race|f1|formula|train|cart|roller", ql):
            return 0.05, 70.0
        if re.search(B + r"(?:bicycle|bike)" + B, ql) and "motor" not in ql:
            return 0.05, 20.0
    if level >= 2 and not re.search(r"toy|model|miniature|statue|sculpture|cartoon|drawing|picture|poster|robot|rc ", ql):
        if kind == "size":
            for pat, dims, lo, hi in SIZE3:
                if re.search(B + pat + B, ql) and re.search(B + r"(?:" + dims + r")" + B, ql):
                    return lo, hi
        if kind in ("speed", "avgspeed"):
            for pat, lo, hi in SPEED3:
                if re.search(B + pat + B, ql):
                    return lo, hi
    return None


def _usi(q):
    q = q.lower()
    if re.search(B + r"cm(?:/s)?" + B, q):
        return 0.01
    if re.search(B + r"mm(?:/s)?" + B, q):
        return 0.001
    return 1.0


def _phys(r, level, both_gm):
    cur = _f(r.get("cur"))
    if not cur:
        return None
    b = class_bound(r["question"], r["prior"], _kind(r), level)
    if b is None:
        return None
    f = _usi(r["question"])
    inside = lambda x: x is not None and b[0] <= x * f <= b[1]
    if inside(cur):
        return None
    g, cs = _f(r.get("s1_g")), _f(r.get("s1_cs"))
    cand = [x for x in (g, cs) if inside(x)]
    if len(cand) == 1:
        return cand[0], f"phys_bound {b} -> own candidate inside"
    if both_gm and len(cand) == 2:
        return math.sqrt(g * cs), f"phys_bound {b} -> gm of both own candidates"
    return None


def U3a(r): return _phys(r, 2, False)
def U3b(r): return _phys(r, 1, False)
def U3c(r): return _phys(r, 2, True)


def _unstable(r):
    if abs(_dfc(r)) > 0.1 * math.log(10):
        return True
    a = [x for x in (r.get("caw_full") or []) if x and x > 0]
    if len(a) >= 4:
        m = float(np.median(a))
        return float(np.mean([abs(math.log(x / m)) < math.log(2) for x in a])) < 0.75
    return False


def _dis(r, var):
    g, cs, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur) or abs(math.log(cur / cs)) > 1e-9 or abs(math.log10(g / cs)) <= math.log10(2):
        return None
    if not _unstable(r):
        return None
    strong = _mode(r) in STRONG
    if var == "a" and not strong:
        return math.sqrt(g * cs), "dis_rel weak, unstable own CaW -> gm"
    if var == "b":
        return math.sqrt(g * cs), "dis_rel unstable own CaW -> gm"
    if var == "c" and strong:
        return g, "dis_rel strong geo, unstable own CaW -> geo"
    return None


def U4a(r): return _dis(r, "a")
def U4b(r): return _dis(r, "b")
def U4c(r): return _dis(r, "c")


VARIANTS = {"U1a": U1a, "U1b": U1b, "U1c": U1c, "U2a": U2a, "U2b": U2b, "U2c": U2c,
            "U3a": U3a, "U3b": U3b, "U3c": U3c, "U4a": U4a, "U4b": U4b, "U4c": U4c}
ACCEPTED = []




ACCEPTED = [("ag_r9_agree_rel", U1a), ("ag_r9_phys_bound_final", U3b)]
