import re
from rules import (UNIT_SI, UNIT_K, clean_text, parse_question as parse_question_v1, parse_prior, parse_depth,
                   match_depth_key, match_depth_keys, COLORS, SIZE_DIMS)

PART_V2 = r"^(?:shoulders?|arms?|legs?|heads?|hands?|feet|foot|fingers?|fingertips?|tips?|edges?|corners?|rims?|undersides?|handles?|lids?|necks?|tails?|ears?|knees?|waist|chest|breadth|eyes?|mouth|nose|beak|roof|screen)$"
WHEEL = r"^(?:wheels?|tires?|tyres?)$"


def fix_typos(q):
    q = re.sub(r"(?i)\b(what is|what's|of|at|is)\s+he\s+(?=[a-z])", lambda m: m.group(1) + " the ", q)
    q = re.sub(r"(?i)\b(?:outer|overall)\s+(diameter|length|width|height)\b", lambda m: m.group(1), q)
    return q


def _sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss") and w not in ("bus", "glass", "chess", "grass"):
        return w[:-1]
    return w


REL_PATS = [
    (r"\s+(?:placed\s+|that is\s+|which is\s+)?(?:to the right of|on the right of|right of)\s+(?:the\s+)?(.+)$", "right_of"),
    (r"\s+(?:placed\s+|that is\s+|which is\s+)?(?:to the left of|on the left of|left of)\s+(?:the\s+)?(.+)$", "left_of"),
    (r"\s+(?:placed\s+|that is\s+|which is\s+|lying\s+)?(?:under|underneath|below|beneath)\s+(?:the\s+)?(.+)$", "below"),
    (r"\s+(?:placed\s+|that is\s+|which is\s+|lying\s+)?(?:on top of|above)\s+(?:the\s+)?(.+)$", "above"),
    (r"\s+(?:placed\s+|that is\s+|which is\s+|standing\s+)?(?:next to|beside|besides|near|behind|in front of|before|ahead of|following|closest to|nearest to|nearest|closest)\s+(?:the\s+)?(.+)$", "near"),
]


def split_relation(phrase):
    if not phrase:
        return phrase, None, None
    for pat, rel in REL_PATS:
        m = re.search(pat, phrase)
        if m and m.start() > 0:
            head = phrase[:m.start()].strip(" ,")
            ref = re.sub(r"\s+(?:in|on|at|with|of)\s+.*$", "", m.group(1)).strip(" ,.?")
            ref = re.sub(r"^(?:the|a|an)\s+", "", ref)
            if head and ref and len(ref.split()) <= 4:
                return head, rel, ref
    return phrase, None, None


def fix_object_phrase(p):
    out = dict(phrase=p, rel=None, ref=None, part=False, dim=None, changed=[])
    if not p:
        return out
    s = p.strip()
    m = re.match(r"^(?:the\s+)?models?\s+of\s+(?:the\s+)?([a-z]+)\b(.*)$", s)
    if m:
        s = (m.group(1) + " model" + m.group(2)).strip(); out["changed"].append("model_of")
    m = re.match(r"^(?:the\s+)?(.+?)'s\s+(.+)$", s)
    if m:
        owner, y = m.group(1).strip(), m.group(2).strip()
        y_head = y.split()[-1] if y.split() else y
        if re.match(r"^wings?$", y_head):
            s = owner; out["dim"] = "wingspan"; out["changed"].append("poss_wings")
        elif re.match(r"^body$", y_head):
            s = owner; out["changed"].append("poss_body")
        elif re.match(PART_V2, y_head):
            s = y; out["part"] = True; out["changed"].append("poss_part")
        else:
            s = _sing(y_head) if re.match(WHEEL, y_head) else y
            out["rel"], out["ref"] = "in", owner; out["changed"].append("poss_in")
    else:
        h, rel, ref = split_relation(s)
        if rel:
            s, out["rel"], out["ref"] = h, rel, ref; out["changed"].append("rel_" + rel)
    out["phrase"] = s
    return out


def parse_question(q_raw, inference_type="SS"):
    q_fixed = fix_typos(str(q_raw))
    r = parse_question_v1(q_fixed, inference_type)
    r["fixes"] = []
    if r["kind"] == "dist2" and r.get("target") and r.get("target2") and re.search(r"\b(?:closest|nearest|farthest)$", r["target"]):
        m = re.match(r"^(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+)$", r["target2"])
        if m:
            r["target"], r["target2"] = r["target"] + " to " + m.group(1), m.group(2)
            r["fixes"].append("dist_split")
            if re.match(r"^(?:floor|ground|wall|water|water surface|surface|ceiling|edge|goal|net|table surface|road|line|finish line|start line)$", r["target2"]):
                r["kind"], r["ok"] = "other", False
    if q_fixed != str(q_raw):
        r["fixes"].append("text_fix")
    for key in ("target", "target2"):
        p = r.get(key)
        if not p or p == "#second":
            r[key + "_rel"] = None
            continue
        f = fix_object_phrase(p)
        r[key] = f["phrase"]
        r[key + "_rel"] = [f["rel"], f["ref"]] if f["rel"] else None
        if f["part"] and key == "target":
            r["part"] = True
        if f["dim"] and key == "target" and r["kind"] == "size" and r["dim"] in ("length", "width", "size", None):
            r["dim"] = f["dim"]
        r["fixes"] += [key + ":" + c for c in f["changed"]]
    return r


def heads_equal(a, b):
    ta = re.findall(r"[a-z]+", a or "")
    tb = re.findall(r"[a-z]+", b or "")
    if not ta or not tb:
        return False
    return _sing(ta[-1]) == _sing(tb[-1])
