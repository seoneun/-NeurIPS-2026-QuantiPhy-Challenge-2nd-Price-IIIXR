import re
NUM = r"(\d*\.?\d+)"
S = r"\s*s(?:ec(?:ond)?s?)?\b"


def fix_times(q):
    s = q.lower().replace("’", "'")
    m = re.search(r"\bfrom\s+(?:rom\s+|the\s+)?(?:start|beginning)\s+to\s+" + NUM + S, s) or \
        re.search(r"\b(?:from\s+)?rom\s+start\s+to\s+" + NUM + S, s)
    if m:
        return dict(t1=0.0, t2=float(m.group(1)), t=None, to_end=False, rule="T_start_to")
    m = re.search(r"\bover\s+" + NUM + r"\s*s?\s*(?:to|-)\s*" + NUM + S, s)
    if m:
        return dict(t1=float(m.group(1)), t2=float(m.group(2)), t=None, to_end=False, rule="T_over")
    m = re.search(r"\bbetween\s+" + NUM + S + r"\s+and\s+" + NUM + r"(?![\d.])(?!\s*s)", s)
    if m:
        return dict(t1=float(m.group(1)), t2=float(m.group(2)), t=None, to_end=False, rule="T_between_nounit")
    m = re.search(r"\bfrom\s+" + NUM + S + r"\s+to\s+the\s+end\b", s)
    if m:
        return dict(t1=float(m.group(1)), t2=None, t=None, to_end=True, rule="T_to_end")
    m = re.search(r"\btime" + NUM + S, s)
    if m:
        return dict(t1=None, t2=None, t=float(m.group(1)), to_end=False, rule="T_glued")
    return None


EVENT = re.compile(r"\b(?:velocity|speed)\b.*\bwhen\s+(?:it\s+)?(?:hits?|hitting|lands?|landing|reaches|touches|strikes)\b")


def event_velocity(q):
    return bool(EVENT.search(q.lower()))


def depth_lines(depth_info):
    out = []
    for l in str(depth_info or "").split("\n"):
        l = l.strip()
        if not l:
            continue
        mt = re.match(r"^t\s*=\s*" + NUM + r"\s*s?\s*(\([^)]*\))?\s*,?\s*", l)
        t, note = (float(mt.group(1)), (mt.group(2) or "").strip("() ")) if mt else (None, "")
        md = re.search(r"distance_?(.+?)\s*=\s*" + NUM, l)
        if md:
            out.append((t, re.sub(r"(?:_and)?_camera\s*$", "", md.group(1).strip()), float(md.group(2)), note))
    return out


def event_time(depth_info, words=("hit", "impact", "land", "touch", "reach")):
    for t, k, d, note in depth_lines(depth_info):
        if t is not None and note and any(w in note.lower() for w in words):
            return t
    return None
