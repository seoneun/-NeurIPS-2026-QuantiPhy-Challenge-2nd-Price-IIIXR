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




def _d2(r):
    return str(r.get("cat")) == "D2"


def N1a(r): return _caw_mra(r, "a") if _d2(r) else None
def N1b(r): return _caw_mra(r, "b") if _d2(r) else None
def N1c(r): return _caw_mra(r, "c") if _d2(r) else None


def _unveto(r, level, hedge):
    if not _d2(r) or str(r.get("video_source")) != "simulation" or _kind(r) != "size":
        return None
    g, gu, s1, cur = _f((r.get("geo") or {}).get("geo_value")), _f((r.get("geo_u") or {}).get("geo_value")), _f(r.get("s1")), _f(r.get("cur"))
    if not (g and gu and s1 and cur) or _mode(r) != "2d_ratio":
        return None
    if abs(math.log(g / gu)) >= 0.01 or abs(math.log(s1 / g)) > 1e-6 or abs(math.log(cur / s1)) <= math.log(1.1):
        return None
    b = class_bound(r["question"], r["prior"], "size", level)
    if b is None:
        return None
    f = _usi(r["question"])
    if not (b[0] <= g * f <= b[1]):
        return None
    v = math.sqrt(cur * g) if hedge else g
    return v, f"sim_unveto L{level}: strong sim geo {g:.4g} inside general bound {b}; cur {cur:.4g} -> {v:.4g}"


def N2a(r): return _unveto(r, 1, False)
def N2b(r): return _unveto(r, 2, False)
def N2c(r): return _unveto(r, 1, True)


GEOM = ("2d_ratio", "vlm_2d_ratio", "2d_gself")
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _echo(x, text):
    for t in NUM.findall(str(text)):
        try:
            v = abs(float(t))
        except Exception:
            continue
        if v > 0 and abs(math.log(x / v)) < 0.01:
            return True
    return False


def _majority(r, tol, ans):
    if not _d2(r) or _mode(r) not in GEOM:
        return None
    g, cs, q, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("qwen")), _f(r.get("cur"))
    if not (g and cs and q and cur):
        return None
    if _echo(q, r.get("prior")):
        return None
    if abs(math.log(cs / g)) <= math.log(2) or abs(math.log(g / q)) >= math.log(tol):
        return None
    gm = math.sqrt(g * cs)
    if not (abs(math.log(cur / cs)) < math.log(1.05) or abs(math.log(cur / gm)) < math.log(1.05)):
        return None
    if abs(math.log(cur / g)) <= math.log(1.6):
        return None
    v = math.sqrt(g * q) if ans == "gm" else g
    return v, f"indep_majority: geo {g:.4g} ~ qwen {q:.4g} vs CaW {cs:.4g}; cur {cur:.4g} -> {v:.4g}"


def N3a(r): return _majority(r, 1.25, "gm")
def N3b(r): return _majority(r, 1.25, "g")
def N3c(r): return _majority(r, 1.5, "gm")


VARIANTS = {k: globals()[k] for k in ("N1a", "N1b", "N1c", "N2a", "N2b", "N2c", "N3a", "N3b", "N3c")}


def _u4(row, var):
    if not _d2(row) or "rul202" not in str(row.get("s1_src")):
        return None
    p = row.get("parse") or {}; k = (p.get("q") or {}).get("kind")
    if k not in ("speed", "avgspeed", "accel", "disp"):
        return None
    g, cur = _f(row.get("s1_g")), _f(row.get("cur"))
    if not (g and cur) or abs(math.log(cur / g)) < 1e-9:
        return None
    if var in ("a", "c"):
        if str((row.get("geo") or {}).get("merge_src")) != "sruler":
            return None
    else:
        po = str(((p.get("prior_obj") or {}).get("phrase")) or "").strip().lower(); tg = str(((p.get("target") or {}).get("phrase")) or "").strip().lower()
        if not po or not tg or po == tg:
            return None
    v = math.sqrt(cur * g) if var == "c" else g
    return v, f"c95_sruler_D2[{var}]: {k} -> own geo {g:.4g} (was {cur:.4g})"


def N4a(r): return _u4(r, "a")
def N4b(r): return _u4(r, "b")
def N4c(r): return _u4(r, "c")


VARIANTS.update(N4a=N4a, N4b=N4b, N4c=N4c)


ACCEPTED = [("ag_r9_N2a_sim_unveto", N2a), ("ag_r9_N3c_indep_majority", N3c), ("ag_r9_N1c_caw_mra_D2", N1c)]


