import re
import numpy as np

B = chr(92) + "b"
NUM = r"(-?\d+(?:\.\d+)?)"
TIME = r"(?:at|after|before|from|between|during|when|within|over|in)\s*(?:the\s+)?(?:time\s*)?(?:t\s*=\s*)?" + NUM + r"\s*(?:s|sec|seconds)" + B + \
       r"(?:\s*(?:to|and|-)\s*(?:t\s*=\s*)?" + NUM + r"\s*(?:s|sec|seconds)" + B + ")?"
KIND = r"(?:average\s+)?(?:speed|velocity|velocoty|acceleration|acc|displacement|distance\s+travel+ed|distance\s+covered|distance)"
STOP = {"the", "a", "an", "s", "m", "cm", "mm", "km", "in", "meters", "metres", "m/s", "m/s^2", "what", "is", "of", "by", "travelled", "traveled"}


def _clean(s):
    s = str(s).lower()
    s = re.sub(r"^\s*t\s*=\s*[\d.]+\s*s?\s*,?", "", s)
    s = s.split("=")[0]
    s = re.sub(r"what is (the )?", "", s)
    s = re.sub(TIME, " ", s)
    s = re.sub(r",?\s*in\s+(?:m|cm|mm|meters|metres|m/s|cm/s|m/s\^2|cm/s\^2)\b", " ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("'s ", " ").replace("?", " ")
    return s


def obj(s):
    s = _clean(s)
    m = re.search(B + KIND + r"\s+(?:of|by)\s+(.+)", s)
    if m:
        s = m.group(1)
    else:
        s = re.sub(B + KIND + B + ".*", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(w for w in s.split() if w not in STOP)


def qkind(q):
    q = q.lower()
    if "acceler" in q or re.search(B + "acc" + B, q):
        return "accel"
    if "average" in q and ("speed" in q or "veloc" in q):
        return "avgspeed"
    if "displacement" in q or re.search(r"distance\s+(?:travel|cover|moved)", q) or re.search(r"how far", q):
        return "disp"
    if "speed" in q or "veloc" in q:
        return "speed"
    return "other"


def pkind(p):
    p = p.lower()
    if "acc" in p or "gravity" in p:
        return "accel"
    if "speed" in p or "veloc" in p:
        return "speed"
    return "size"


def times(q):
    q = q.lower()
    m = re.search(r"(?:from|between)\s*(?:t\s*=\s*)?" + NUM + r"\s*s?\s*(?:to|and|-)\s*(?:t\s*=\s*)?" + NUM + r"\s*s", q)
    if m:
        return "int", float(m.group(1)), float(m.group(2))
    m = re.search(r"(?:at|after|before)\s*(?:the\s+)?(?:time\s*)?(?:t\s*=\s*)?" + NUM + r"\s*s", q)
    if m:
        return "at", float(m.group(1)), None
    return "none", None, None


def prior_value(p):
    m = re.search(r"=\s*" + NUM + r"\s*([a-z/^²0-9]*)", str(p).lower())
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2)


def to_si(u):
    u = (u or "").lower()
    if u.startswith("cm"):
        return 0.01
    if u.startswith("mm"):
        return 0.001
    if u.startswith("km/h"):
        return 1 / 3.6
    return 1.0


def q_scale(q):
    q = q.lower()
    if re.search(B + r"cm(?:/s)?" + B, q):
        return 0.01
    if re.search(B + r"mm(?:/s)?" + B, q):
        return 0.001
    return 1.0


BEFORE_OK = False


def prior_windowed(p, t_q=None):
    p = p.lower()
    if BEFORE_OK and t_q is not None:
        mb = re.search(r"before\s*([\d.]+)\s*s", p)
        if mb and not re.search(r"after\s*[\d.]+\s*s|from\s*[\d.]+\s*s?\s*to|between\s*[\d.]+", p) and t_q <= float(mb.group(1)) + 1e-9:
            return False
    return bool(re.search(r"from\s*[\d.]+\s*s?\s*to", p) or re.search(r"between\s*[\d.]+", p)
                or re.search(r"(?:after|before)\s*[\d.]+\s*s", p))


TOFF = 0.0


def propagate(prior, q):
    pk, qk = pkind(prior), qkind(q)
    po, qo = obj(prior), obj(q)
    if len(po) < 3 or po != qo:
        return np.nan, "other_object"
    if pk == "size":
        return np.nan, "size_prior"
    if "gravit" in prior.lower() or re.search(r"=\s*9\.8\d*\s*m", prior.lower()):
        return np.nan, "gravity"
    tk, t1, t2 = times(q)
    if prior_windowed(prior, t1 if tk == "at" else (t2 if tk == "int" else None)):
        return np.nan, "windowed"
    v, u = prior_value(prior)
    if v is None:
        return np.nan, "no_value"
    vs = abs(v) * to_si(u)
    out, how = np.nan, "no_rule"
    if pk == "speed":
        if qk in ("speed", "avgspeed"):
            out, how = vs, "v"
        elif qk == "disp" and tk == "int":
            out, how = vs * (t2 - t1), "v*dt"
    else:
        if qk == "accel":
            out, how = vs, "a"
        elif qk == "speed" and tk == "at":
            out, how = vs * max(t1 - TOFF, 0.0), "a*t"
        elif qk == "avgspeed" and tk == "int":
            out, how = vs * (max(t1 - TOFF, 0.0) + max(t2 - TOFF, 0.0)) / 2, "a*tm"
        elif qk == "disp" and tk == "int":
            out, how = vs * (max(t2 - TOFF, 0.0) ** 2 - max(t1 - TOFF, 0.0) ** 2) / 2, "a*dt2/2"
    if not np.isfinite(out) or out <= 0:
        return np.nan, how if not np.isfinite(out) else "zero"
    return out / q_scale(q), how


def propagate_gravity(prior, q):
    p = prior.lower()
    if not ("gravit" in p or re.search(r"=\s*9\.8\d*\s*m", p)):
        return np.nan, "not_gravity"
    tk, t1, _ = times(q)
    if prior_windowed(prior, t1 if tk == "at" else None):
        return np.nan, "windowed"
    if qkind(q) != "speed":
        return np.nan, "no_rule"
    if tk != "at" or t1 is None or t1 <= 0:
        return np.nan, "no_time"
    v, u = prior_value(prior)
    g = abs(v) * to_si(u) if v is not None and abs(v) > 1 else 9.8
    return g * max(t1 - TOFF, 0.0) / q_scale(q), "g*t"


PERSON = B + r"(?:person|pedestrian|man|woman|walker|people|player|boy|girl|kid|child|lady|guy|swimmer|runner)" + B


def class_bound(q, prior, kind):
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
        if EXTRA_BOUNDS:
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
            if EXTRA2 and not re.search(r"toy|model|miniature|dollhouse|lego", ql):
                if re.search(B + r"door" + B, ql) and re.search(B + r"height|tall" + B, ql) and not re.search(r"car|cabinet|fridge|oven|cage|truck|bus", ql):
                    return 1.6, 3.5
                if re.search(B + r"(?:table|desk)" + B, ql) and re.search(B + r"height" + B, ql) and not re.search(r"leg|top|surface thick|lamp", ql):
                    return 0.25, 1.4
                if re.search(B + r"(?:iphone|smartphone|phone|cellphone|mobile phone)" + B, ql) and not re.search(r"booth|box|stand", ql):
                    if re.search(B + r"width" + B, ql):
                        return 0.045, 0.13
                    if re.search(B + r"(?:length|height)" + B, ql):
                        return 0.09, 0.22
    if EXTRA2 and kind in ("speed", "avgspeed") and not re.search(r"toy|model|rc |remote", ql):
        if re.search(B + r"car" + B, ql) and not re.search(r"race|f1|formula|train|cart|roller", ql):
            return 0.05, 70.0
        if re.search(B + r"(?:bicycle|bike)" + B, ql) and "motor" not in ql:
            return 0.05, 20.0
    if EXTRA3 and not re.search(r"toy|model|miniature|statue|sculpture|cartoon|drawing|picture|poster|robot|rc ", ql):
        if kind == "size":
            for pat, dims, lo, hi in SIZE3:
                if re.search(B + pat + B, ql) and re.search(B + r"(?:" + dims + r")" + B, ql):
                    return lo, hi
        if kind in ("speed", "avgspeed"):
            for pat, lo, hi in SPEED3:
                if re.search(B + pat + B, ql):
                    return lo, hi
    return None


EXTRA3 = False
SIZE3 = [(r"horse", "height|tall", 0.8, 2.7), (r"elephant", "height|tall", 1.4, 4.8), (r"dog", "height|tall", 0.12, 1.2),
         (r"cat", "height|tall", 0.12, 0.7), (r"chair", "height|tall", 0.35, 1.4), (r"sofa|couch", "height|tall", 0.4, 1.3),
         (r"bed", "length|long", 1.4, 2.6), (r"(?:water |wine |plastic |glass )?bottle", "height|tall|length", 0.04, 0.6),
         (r"(?:soda |coke |beer )?can", "height|tall", 0.05, 0.25), (r"(?:cup|mug)", "height|tall", 0.03, 0.35),
         (r"book", "length", 0.08, 0.5), (r"laptop", "width|length", 0.2, 0.5), (r"keyboard", "length", 0.2, 0.65),
         (r"bus", "length|long", 5.0, 25.0), (r"motorcycle|motorbike", "length|long", 1.3, 3.0), (r"(?:bicycle|bike)", "length|long", 1.0, 2.4),
         (r"(?:rabbit|bunny|hare)", "length|long|height|tall", 0.08, 1.0), (r"pepperoni", "diameter|size|width", 0.015, 0.10)]
SPEED3 = [(r"horse", 0.05, 22.0), (r"dog", 0.05, 22.0), (r"cat", 0.02, 15.0), (r"(?:ball|tennis ball|football|basketball|soccer ball)", 0.01, 75.0),
          (r"(?:boat|ship|speedboat|sailboat)", 0.02, 45.0)]


EXTRA2 = False


EXTRA_BOUNDS = False
BALLS = [(r"tennis ball", 0.05, 0.09), (r"ping[ -]?pong ball", 0.03, 0.05), (r"table tennis ball", 0.03, 0.05), (r"golf ball", 0.035, 0.055),
         (r"billiard ball", 0.045, 0.07), (r"pool ball", 0.045, 0.07), (r"basketball", 0.15, 0.32), (r"soccer ball", 0.12, 0.30),
         (r"football", 0.12, 0.32), (r"volleyball", 0.15, 0.30), (r"baseball", 0.06, 0.09), (r"bowling ball", 0.18, 0.25),
         (r"yoga ball", 0.30, 1.10)]


def q_si(q):
    return q_scale(q)
