import re

UNIT_SI = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3,
           "mm/s": 1e-3, "cm/s": 1e-2, "m/s": 1.0, "km/h": 1 / 3.6,
           "mm/s^2": 1e-3, "cm/s^2": 1e-2, "m/s^2": 1.0}
UNIT_K = {"mm": 0, "cm": 0, "m": 0, "km": 0, "mm/s": 1, "cm/s": 1, "m/s": 1, "km/h": 1,
          "mm/s^2": 2, "cm/s^2": 2, "m/s^2": 2}

TYPOS = {
    "bycicle": "bicycle", "conrner": "corner", "balck": "black", "ballon the": "ball on the",
    "velolicty": "velocity", "velocoty": "velocity", "diatance": "distance", "soccor": "soccer",
    "diasplacement": "displacement", "centeral": "central", "baskteball": "basketball",
    "lenth": "length", "rebounce": "rebound", "soccerball": "soccer ball", "bill board": "billboard",
    "the the ": "the ", "shoppingbag": "shopping bag", "conveyer": "conveyor", "trashbin": "trash bin",
    "pingpong": "ping pong",
}
NUM = r"(\d*\.?\d+)"


def clean_text(s):
    s = str(s)
    s = re.sub(r"�+(?=s\b)", "'", s)
    s = s.replace("�", "").replace("’", "'").replace("‘", "'")
    s = s.replace("“", "").replace("”", "").replace('"', "")
    s = s.replace("？", "?").replace("²", "^2")
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()
    for a, b in TYPOS.items():
        low = low.replace(a, b)
    return low


UNIT_RE = r"(m/s\^2|m/s2|cm/s\^2|cm/s2|mm/s\^2|m/s|cm/s|mm/s|km/h|meters?|metres?|centimeters?|millimeters?|cm|mm|km|m)"


def norm_unit(u):
    u = u.lower().strip()
    u = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "centimeter": "cm", "centimeters": "cm",
         "millimeter": "mm", "millimeters": "mm", "m/s2": "m/s^2", "cm/s2": "cm/s^2"}.get(u, u)
    return u if u in UNIT_SI else None


def question_unit(q, inference_type):
    m = re.findall(r"\bin\s*\(?\s*" + UNIT_RE + r"(?![a-z/])", q)
    if not m:
        m = re.findall(r",\s*in\s+" + UNIT_RE, q)
    if m:
        u = norm_unit(m[-1])
        if u:
            return u
    return "m" if inference_type[1] == "S" else "m/s"


def parse_times(q):
    t = t1 = t2 = None
    when = None
    s = q
    pats_iv = [r"(?:between|from|in)\s+(?:t\s*=\s*)?" + NUM + r"\s*s?(?:econds?)?\s*(?:and|to|-)\s*(?:t\s*=\s*)?" + NUM + r"\s*s(?:econds?)?\b"]
    for p in pats_iv:
        m = re.search(p, s)
        if m:
            t1, t2 = float(m.group(1)), float(m.group(2))
            s = s[:m.start()] + " " + s[m.end():]
            break
    m = re.search(r"(?:in|during|within)\s+the\s+first\s+" + NUM + r"\s*s(?:econds?)?\b", s)
    if m and t1 is None:
        t1, t2 = 0.0, float(m.group(1))
        s = s[:m.start()] + " " + s[m.end():]
    if t1 is None:
        for p in [r"(?:,\s*)?(?:at|when|after)\s*(?:the\s+)?(?:time\s*)?(?:t\s*=\s*)?" + NUM + r"\s*s(?:econds?)?\b,?",
                  r"(?:,\s*)?(?:at|when)\s*(?:time\s*)?t\s*=\s*" + NUM + r"\s*s?\b,?",
                  r"\(\s*at\s+time\s+" + NUM + r"\s*s\s*\)",
                  r"\bt\s*=\s*" + NUM + r"\s*s?\b,?"]:
            m = re.search(p, s)
            if m:
                t = float(m.group(1))
                s = s[:m.start()] + " " + s[m.end():]
                break
    if re.search(r"\binitial(ly)?\b|\bat the (beginning|start)\b", s) and t is None and t1 is None:
        when = "start"
    if re.search(r"\bfinal(ly)?\b|\bat the end\b", s) and t is None and t1 is None:
        when = "end"
    s = re.sub(r"from the beginning to the end", " ", s)
    return t, t1, t2, when, re.sub(r"\s+", " ", s).strip()


SIZE_DIMS = [("wingspan", "wingspan"), ("wing span", "wingspan"), ("diameter", "diameter"), ("radius", "radius"),
             ("thickness", "thickness"), ("height", "height"), ("tall", "height"), ("length", "length"),
             ("long", "length"), ("width", "width"), ("wide", "width"), ("breadth", "width"), ("size", "size"),
             ("depth", "thickness")]


def _strip_phrase(p):
    p = p.strip(" ?.,;:")
    p = re.sub(r"^(?:the|a|an)\s+", "", p)
    p = re.sub(r"\s*\([^)]*\)", "", p)
    p = re.sub(r"\s+in\s*$", "", p)
    p = re.sub(r"\s+(?:moving|when it|that is|which is).*$", "", p) if len(p.split()) > 4 else p
    return p.strip(" ?.,;:")


def parse_question(q_raw, inference_type="SS"):
    q = clean_text(q_raw)
    unit = question_unit(q, inference_type)
    t, t1, t2, when, s = parse_times(q)
    s = re.sub(r",?\s*\(?\bin\s*\(?\s*" + UNIT_RE + r"(?![a-z/])\)?,?", " ", s)
    s = re.sub(r"^(?:when\s*,?\s*)?", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ?.,")
    body = re.sub(r"^(?:what\s*(?:'s|is|was|are)?|how\s+(?:wide|tall|long|high|big|far|fast)\s+(?:is|are))\s+", "", s)
    howdim = re.match(r"^how\s+(wide|tall|long|high|big)\s+(?:is|are)\s+(.*)$", s)
    r = dict(unit=unit, t=t, t1=t1, t2=t2, when=when, kind="other", dim=None, target=None, target2=None,
             horizontal=bool(re.search(r"horizontal", q)), vertical=bool(re.search(r"vertical", q)), ok=False, q=q)
    if howdim:
        r["kind"], r["dim"] = "size", {"wide": "width", "tall": "height", "long": "length", "high": "height", "big": "size"}[howdim.group(1)]
        tgt = howdim.group(2)
        if "wingspan" in tgt or "wing span" in tgt:
            r["dim"] = "wingspan"
            tgt = re.sub(r"'s\s+wing\s*span.*$", "", tgt)
        r["target"] = _strip_phrase(tgt)
        r["ok"] = bool(r["target"])
        return r
    b = re.sub(r"^(?:the\s+)", "", body)
    mpos = re.match(r"^(?:that\s+|this\s+)?(.+?)'s\s+(.*)$", b)
    quant_txt, obj_txt = b, None
    if mpos and not re.match(r"^(?:height|length|width|diameter|speed|velocity|acceleration|distance|displacement)\b", b):
        obj_txt, quant_txt = mpos.group(1), mpos.group(2)
    lowq = re.sub(r"\s*\((?!the one)[^)]*\)", " ", quant_txt)
    lowq = re.sub(r"\s+", " ", lowq).strip()
    if re.search(r"\bacceleration\b", lowq):
        r["kind"] = "accel"
    elif re.search(r"\bdisplacement\b", lowq):
        r["kind"] = "disp"
    elif re.search(r"distance\s+(?:traveled|travelled|covered|moved)|total distance|distance (?:it|the \w+) (?:traveled|travelled|moved)|path length", lowq):
        r["kind"] = "path"
    elif re.search(r"\b(?:speed|velocity|velocities)\b", lowq):
        avg = re.search(r"\baverage\b|\bmean\b", lowq) or (t1 is not None) or (t is None and when is None)
        r["kind"] = "hvel" if (r["horizontal"] and avg) else ("avgspeed" if avg else "speed")
    elif re.search(r"\borbital (?:diameter|size)|diameter of (?:the |its )?orbit", lowq):
        r["kind"], r["dim"] = "orbitdiam", None
    elif re.search(r"\bmaximum (?:height|rebound)|max(?:imum)? height|highest|rebound height", lowq):
        r["kind"] = "maxheight"
    elif re.search(r"\bdistance\b|\bspacing\b|\bgap\b", lowq):
        r["kind"] = "dist2"
    else:
        for w, d in SIZE_DIMS:
            if re.search(r"\b" + w + r"\b", lowq):
                r["kind"], r["dim"] = "size", d
                break
    if obj_txt is not None:
        r["target"] = _strip_phrase(obj_txt)
    else:
        if r["kind"] == "dist2":
            m = re.search(r"\bdistance\b.*?\bbetween\s+(.+?)\s+and\s+(.+)$", lowq) or \
                re.search(r"\bdistance\b.*?\bfrom\s+(.+?)\s+to\s+(.+)$", lowq) or \
                re.search(r"\bdistance\s+of\s+(.+?)\s+(?:to|from)\s+(.+)$", lowq)
            if m:
                a, bb = _strip_phrase(m.group(1)), _strip_phrase(m.group(2))
                r["target"], r["target2"] = a, bb
            else:
                m = re.search(r"\bbetween\s+(?:the\s+)?(two|2|both)\s+(.+)$", lowq)
                if m:
                    r["target"] = _strip_phrase(m.group(2))
                    r["target2"] = "#second"
        else:
            m = re.search(r"\b(?:of|for|by)\s+(.+)$", lowq)
            if m:
                tgt = m.group(1)
                r["target"] = _strip_phrase(tgt)
    if r["target"]:
        r["target"] = re.sub(r"\s+(?:between|from|during|at)\s*$", "", r["target"]).strip()
    if r["kind"] == "size" and re.search(r"\bwing\s*span\b|\(wing span\)", lowq):
        r["dim"] = "wingspan"
    if r["kind"] == "size" and re.search(r"\bside length\b", lowq):
        r["dim"] = "side"
    GROUND = r"^(?:the\s+)?(?:floor|ground|wall|water|water surface|surface|ceiling|edge|goal|net|table surface|road|line|finish line|start line)$"
    if r["kind"] == "dist2" and r["target2"] and re.match(GROUND, r["target2"]):
        r["kind"] = "other"
    if r["kind"] == "size" and r["target"] and re.search(r"\babove the\b|\bfrom the floor\b|\bto the ground\b|\bbetween\b", r["target"]):
        r["kind"] = "other"
    r["ok"] = r["kind"] != "other" and bool(r["target"])
    return r


def parse_prior_line(line):
    s = clean_text(line)
    out = dict(kind=None, dim=None, value_si=None, t=None, t1=None, t2=None, obj=None, gravity=False, ok=False, raw=s)
    pos = max(s.rfind("="), s.rfind("~"))
    tail = s[pos + 1:] if pos >= 0 else s
    m = re.search(NUM + r"\s*" + UNIT_RE + r"(?![a-z])", tail)
    if not m:
        return out
    val = abs(float(m.group(1)))
    u = norm_unit(m.group(2))
    name = s[:pos] if pos >= 0 else s[:m.start()]
    k = UNIT_K[u]
    if re.search(r"acc", name) and k == 1:
        u, k = u + "^2", 2
    out["value_si"] = val * UNIT_SI[u]
    out["kind"] = {0: "size", 1: "speed", 2: "accel"}[k]
    mt = re.search(r"from\s+" + NUM + r"\s*s?\s*to\s+" + NUM + r"\s*s", name)
    if mt:
        out["t1"], out["t2"] = float(mt.group(1)), float(mt.group(2)); name = name.replace(mt.group(0), " ")
    mt = re.search(r"before\s+" + NUM + r"\s*s", name)
    if mt:
        out["t1"], out["t2"] = 0.0, float(mt.group(1)); name = name.replace(mt.group(0), " ")
    mt = re.search(r"(?:at|after)\s+" + NUM + r"\s*s", name) or re.search(r"t\s*=\s*" + NUM + r"\s*s?\s*,?", name)
    if mt:
        out["t"] = float(mt.group(1)); name = name.replace(mt.group(0), " ")
    name = re.sub(r"\s+", " ", name).strip(" ,")
    out["gravity"] = bool(re.search(r"gravity|gravitational", name)) or (k == 2 and abs(out["value_si"] - 9.8) < 0.15 and not re.search(r"\b(?:car|truck|boat|plane|rocket)\b", name))
    for w, d in SIZE_DIMS:
        if re.search(r"\b" + w + r"\b", name):
            out["dim"] = d if k == 0 else None
            break
    obj = name
    obj = re.sub(r"\b(?:average|total|the lifting|lifting|horizontal|vertical|inner|outer)\b", " ", obj)
    obj = re.sub(r"\b(?:speed|velocity|acceleration|acc|diameter|radius|length|lenth|height|width|size|calibre|wingspan|thickness|gravity_acceleration|gravity|gravitational)\b", " ", obj)
    obj = re.sub(r"^\s*(?:of|the)\s+", " ", obj)
    obj = re.sub(r"\s+", " ", obj).strip()
    obj = re.sub(r"^(?:of\s+)?(?:the\s+)?", "", obj)
    obj = re.sub(r"^(?:of\s+)?(?:the\s+)?", "", obj)
    obj = re.sub(r"\(.*?\)", "", obj).strip(" ,'")
    obj = re.sub(r"'s?\s*model", " model", obj).strip()
    obj = re.sub(r"^one\s+", "", obj)
    if re.fullmatch(r"(?:pedestrian\s*)?walking(?:\s*person)?|pedestrian|walking pedestrian|person walking", obj or ""):
        obj = "person"
    out["obj"] = obj if obj else None
    if re.search(r"ruler|calibre", name):
        out["obj"], out["ok_measure"] = "ruler", False
    if out["gravity"] and (not obj or re.fullmatch(r"(?:gravity\s*)?(?:acc|acceleration)?", obj or "")):
        out["obj"] = None
    out["ok"] = True
    return out


def parse_prior(prior):
    lines = [l for l in str(prior).split("\n") if l.strip()]
    res = [parse_prior_line(l) for l in lines]
    return [r for r in res if r["ok"]] or res[:1]


def parse_depth(depth_info):
    out = []
    if depth_info is None or str(depth_info).strip() in ("", "nan", "None"):
        return out
    for line in str(depth_info).split("\n"):
        l = line.strip()
        if not l:
            continue
        t = None
        mt = re.match(r"^\s*t\s*=\s*" + NUM + r"\s*s?\s*,\s*", l)
        if mt:
            t = float(mt.group(1)); l = l[mt.end():]
        m = re.search(r"distance_?(.+?)\s*=\s*" + NUM, l)
        if not m:
            continue
        key = m.group(1).strip()
        key = re.sub(r"(?:_and)?_camera\s*$", "", key)
        out.append(dict(key=key, t=t, d=float(m.group(2))))
    return out


SYN = {"human": "person", "people": "person", "pedestrian": "person", "man": "person", "woman": "person",
       "boy": "person", "girl": "person", "player": "person", "walker": "person", "lady": "person",
       "bike": "bicycle", "cyclist": "bicycle", "pingpong": "ping", "ping-pong": "ping", "pong": "ping",
       "notebook": "note", "trashbin": "trash", "bin": "trash", "automobile": "car", "vehicle": "car",
       "aeroplane": "plane", "airplane": "plane", "jet": "plane", "kid": "person", "child": "person",
       "astronaut": "person", "swimmer": "person", "boatman": "person", "runner": "person",
       "centeral": "central", "seagull": "bird", "eagle": "eagle", "ship": "boat", "sailboat": "boat",
       "basketball": "basketball", "football": "soccer", "mug": "cup", "glass": "cup"}
COLORS = {"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver", "golden", "gold"}
STOP = {"the", "a", "an", "of", "and", "on", "in", "at", "to", "camera", "distance", "one", "with", "is", "that", "this"}


def _toks(s):
    s = clean_text(s).replace("_", " ").replace("-", " ")
    toks = [w for w in re.findall(r"[a-z]+", s) if w not in STOP]
    out = []
    for w in toks:
        w = SYN.get(w, w)
        if w.endswith("s") and len(w) > 3 and w not in ("glass", "bus", "grass", "chess"):
            w = w[:-1]
        out.append(w)
    return out


def match_depth_keys(phrase, keys):
    best, bs = match_depth_key(phrase, keys)
    if best is None:
        return [], bs
    out = [k for k in keys if abs(match_depth_key(phrase, [k])[1] - bs) < 0.02]
    return out, bs


def match_depth_key(phrase, keys):
    if not phrase or not keys:
        return None, 0.0
    pt = _toks(phrase)
    if not pt:
        return None, 0.0
    head = pt[-1]
    best, bs = None, 0.0
    for k in keys:
        kt = _toks(k)
        if not kt:
            continue
        inter = len(set(pt) & set(kt))
        if inter == 0:
            continue
        sc = inter / len(set(pt) | set(kt))
        if head in kt:
            sc += 0.5
        for a, b2 in (("left", "right"), ("near", "far"), ("big", "small"), ("front", "back"), ("top", "bottom")):
            if (a in pt and b2 in kt) or (b2 in pt and a in kt):
                sc -= 0.6
        pc, kc = set(pt) & COLORS, set(kt) & COLORS
        if pc and kc and not (pc & kc):
            sc -= 1.0
        if sc > bs:
            best, bs = k, sc
    return (best, bs) if bs >= 0.3 else (None, bs)
