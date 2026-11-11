import re, math
import numpy as np, cv2

COLOR_WORDS = ["white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver"]


def sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if (w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w




def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
















SYN_CLASS = {"human": "person", "people": "person", "pedestrian": "person", "man": "person", "woman": "person",
             "boy": "person", "girl": "person", "player": "person", "walker": "person", "lady": "person", "kid": "person",
             "child": "person", "astronaut": "person", "swimmer": "person", "runner": "person", "passer": "person",
             "receiver": "person", "bike": "bicycle", "cyclist": "bicycle", "automobile": "car", "vehicle": "car",
             "aeroplane": "plane", "airplane": "plane", "jet": "plane", "seagull": "bird", "ship": "boat",
             "football": "soccer", "mug": "cup", "glass": "cup", "trashbin": "bin", "trashcan": "bin"}
QUAL_GROUPS = [{"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver", "golden", "gold"},
               {"big", "large", "biggest", "largest", "small", "little", "smallest", "tall", "short"},
               {"left", "right", "middle", "center", "centre"}, {"upper", "top", "lower", "bottom"},
               {"front", "near", "back", "far", "closest", "farthest"}, {"first", "second", "third", "last"}]
_CUTW = r"\b(?:riding|walking on|holding|wearing|with|in|on|at|of|from|near|next|behind|under|above)\b"


def head_noun(p):
    s = (p or "").lower()
    toks = re.findall(r"[a-z]+", s)
    if not toks:
        return ""
    m = re.search(_CUTW, s)
    if m and m.start() > 0:
        pre = re.findall(r"[a-z]+", s[:m.start()])
        if pre:
            toks = pre
    return sing(toks[-1])


def heads_compatible(pa, pb):
    ha, hb = head_noun(pa), head_noun(pb)
    if not ha or not hb:
        return False
    if ha == hb or SYN_CLASS.get(ha, ha) == SYN_CLASS.get(hb, hb):
        return True
    return (len(hb) >= 3 and ha.endswith(hb)) or (len(ha) >= 3 and hb.endswith(ha))


def qualifier_conflict(wa, wb):
    for g in QUAL_GROUPS:
        a, b = wa & g, wb & g
        if a and b and a != b:
            return True
    return False


def contrastive_v2(pa, where_a, pb, where_b):
    where_a = [w for w in (where_a or [])]; where_b = [w for w in (where_b or [])]
    if "#second" in where_a or "#second" in where_b:
        return True
    if not heads_compatible(pa, pb):
        return True
    wa = set(re.findall(r"[a-z]+", (pa or "") + " " + " ".join(w for w in where_a if not w.startswith("#"))))
    wb = set(re.findall(r"[a-z]+", (pb or "") + " " + " ".join(w for w in where_b if not w.startswith("#"))))
    return qualifier_conflict(wa, wb)
