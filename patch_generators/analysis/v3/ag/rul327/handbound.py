"""rul327 (audit-12 rule 5): a generous physical speed bound for a HAND / FINGER target in a close-up manipulation scene,
plugged into the pipeline's own class-bound step (kinematic.class_bound, used by submit_v3 --phys_bounds /
--phys_bounds_final: an answer outside the bound takes the one candidate - geo or CaW - inside it).

Parse fix (target phrase): the pipeline parse keeps verbs and possessives in the target ("bottom hand moves",
"moving speed of the user's (middle) fingertip", "the man's moving hand"); here the question is normalised first -
possessor "X's" and "(middle)" qualifiers and the verbs move/moves/moved/moving are stripped - and the target is the
noun after "speed|velocity of". A failed parse kind ('other') is re-derived: "from/between t1 to/and t2" -> avgspeed,
"at t" -> speed.

Physics: when the only stated prior is a small hand-held object (a length <= 0.3 m: bill, card, mouse) the shot is a
close-up of hands manipulating that object. Hand transport in manipulation follows a bell-shaped (minimum-jerk) speed
profile whose mean is ~0.53 x the peak, and peak speeds of fast natural reaches are ~1-2 m/s: an AVERAGE hand speed
above 1.5 m/s or an instantaneous one above 2.5 m/s does not occur. Throwing / striking / ball scenes are excluded
(a pitcher's hand reaches 20+ m/s). Lower bound 0.02 m/s (the question says the hand moves)."""
import re

B = r"\b"
HAND = r"(?:hand|hands|finger|fingers|fingertip|fingertips|palm|palms|wrist|wrists|thumb|thumbs)"
VERB = r"\b(?:move|moves|moved|moving)\b"
EXCL = re.compile(r"\b(?:ball|throw\w*|threw|pitch\w*|punch\w*|swing\w*|swung|hit\w*|strik\w*|slap\w*|clap\w*|bat|racket|racquet|"
                  r"serv\w*|toss\w*|whip\w*|box\w*|drum\w*|karate|fight\w*|wav\w*|shak\w*)\b")
LEN = re.compile(r"=\s*([0-9]*\.?[0-9]+)\s*(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|in|inch|inches)\b(?!\s*/)")
TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}
AVG_HI, INST_HI, LO = 1.5, 2.5, 0.02
PRIOR_MAX = 0.3


def norm(q):
    """the parse fix: strip possessors, parenthesised qualifiers and the verb 'move' from the question"""
    ql = str(q).lower().replace("\u2019", "'")
    ql = re.sub(r"\b[a-z]+'s\s+", "", ql)              # "the user's (middle) fingertip" -> "the (middle) fingertip"
    ql = re.sub(r"\(([a-z ]+)\)\s*", r"\1 ", ql)        # "(middle) fingertip" -> "middle fingertip"
    ql = re.sub(VERB, " ", ql)                          # "bottom hand moves", "moving hand", "moving speed of"
    return re.sub(r"\s+", " ", ql).strip()


def hand_target(q):
    """the target is a hand / finger: 'speed|velocity of (the) [<=2 modifiers] HAND'"""
    return re.search(r"\b(?:speed|velocity) of (?:the |a |an )?(?:[a-z\-]+ ){0,2}" + HAND + B, norm(q)) is not None


def kind_of(q, kind):
    if kind in ("speed", "avgspeed"):
        return kind
    if kind not in (None, "", "other"):
        return kind
    n = norm(q)
    if not re.search(r"\b(?:speed|velocity)\b", n):
        return kind
    if re.search(r"\b(?:from|between)\s+[0-9.]+\s*s?\s+(?:to|and)\s+[0-9.]+", n) or "average" in n:
        return "avgspeed"
    if re.search(r"\bat\s+(?:time\s+)?[0-9.]+\s*s\b", n):
        return "speed"
    return kind


def prior_len_m(prior):
    m = LEN.search(str(prior).lower())
    if not m:
        return None
    u = m.group(2)
    u = "mm" if u.startswith("milli") else "cm" if u.startswith("centi") else "m" if u.startswith("met") else u
    return float(m.group(1)) * TO_M[u]


def hand_bound(q, prior, kind):
    """(lo, hi) in m/s for a hand/finger speed question in a close-up hand-held-object scene, else None"""
    k = kind_of(q, kind)
    if k not in ("speed", "avgspeed"):
        return None
    if not hand_target(q):
        return None
    ql, pl = str(q).lower(), str(prior).lower()
    if EXCL.search(ql) or EXCL.search(pl):
        return None
    L = prior_len_m(prior)
    if L is None or not (0 < L <= PRIOR_MAX):
        return None
    return (LO, AVG_HI if k == "avgspeed" else INST_HI)


def install(K):
    """wrap kinematic.class_bound: an existing class bound wins; otherwise the hand bound"""
    if getattr(K.class_bound, "_parse3", False):
        return
    orig = K.class_bound

    def cb(q, prior, kind):
        b = orig(q, prior, kind)
        return b if b is not None else hand_bound(q, prior, kind)
    cb._parse3 = True
    K.class_bound = cb
