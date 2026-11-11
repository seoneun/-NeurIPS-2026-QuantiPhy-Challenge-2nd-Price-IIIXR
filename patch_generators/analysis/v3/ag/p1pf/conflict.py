"""Shared colour / type-modifier conflict test for two object phrases (one function for P1, A, E and the rul439 skip).

phrase_conflict(a, b) -> (bool, reason). True when the two phrases name DIFFERENT objects, so one track carrying both
phrases means one grounding is wrong:
  - both carry a colour and the colours are disjoint            ('white car' vs 'yellow car')
  - both carry a type / identity modifier and they are disjoint ('tennis ball' vs 'ping pong ball', 'jupiter model' vs
    'callisto model', 'virus|left' vs 'virus|right')
  - the head nouns differ and are not synonyms                  ('trash bin' vs 'book', 'boat' vs 'person')
Never a conflict (stop-list):
  - synonyms of one head ('person'/'man'/'human'/'pedestrian', 'bike'/'bicycle', 'car'/'vehicle'/'taxi', 'boat'/'speedboat')
  - a part named next to its whole ('boat' vs 'outer end', 'trash bin' vs 'rim', 'bicycle' vs 'bicycle wheel')
  - motion / pose descriptors ('turning yellow car' vs 'yellow car', 'moving bicycle' vs 'bike')
  - one phrase a word-subset of the other with no disjoint modifier ('cells' vs 'red blood cell')
Pure function of the two phrase strings (no labels)."""
import re

COLOURS = {"white", "black", "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "grey", "gray",
           "silver", "gold", "golden", "beige", "cyan", "violet", "maroon", "navy", "turquoise", "magenta"}
TYPEMOD = {"tennis", "soccer", "basket", "basketball", "yoga", "ping", "pong", "golf", "volley", "volleyball", "foot",
           "football", "baseball", "billiard", "bowling", "rubber", "glass", "wooden", "metal", "plastic", "paper", "toy",
           "jupiter", "callisto", "europa", "io", "ganymede", "earth", "moon", "mars", "sun",
           "left", "right", "first", "second", "third", "front", "back", "rear", "near", "far", "upper", "lower",
           "big", "small", "large", "little", "tall", "short", "bigger", "smaller", "larger", "taller", "shorter"}
DESCR = {"the", "a", "an", "of", "model", "moving", "turning", "walking", "rolling", "falling", "running", "flying",
         "standing", "sitting", "parked", "bouncing", "swinging", "sliding", "spinning", "rotating", "orbiting", "dropped",
         "thrown", "launched", "accelerating", "decelerating", "leading", "following", "another", "other", "one", "two"}
SYN = [{"person", "man", "woman", "human", "pedestrian", "people", "boy", "girl", "player", "child", "kid", "lady", "guy",
        "adult", "walker"},
       {"bike", "bicycle"},
       {"car", "vehicle", "taxi", "automobile", "suv", "sedan", "cab"},
       {"boat", "speedboat", "ship", "vessel"},
       {"cell", "cells"},
       {"plane", "airplane", "aeroplane", "aircraft", "jet"},
       {"bin", "can", "trashcan", "dustbin", "garbage"},
       {"ball"}]
PARTS = {"rim", "lid", "handle", "wheel", "wheels", "leg", "legs", "top", "edge", "end", "tip", "window", "door", "cap",
         "screen", "base", "wing", "wings", "tail", "roof", "bottom", "corner", "outer", "inner", "blade", "seat",
         "pedal", "mast", "sail", "propeller", "rotor"}
PREP = {"in", "with", "on", "at", "near", "wearing", "holding", "from", "under", "behind", "beside", "by", "carrying"}


def _norm(w):
    w = w.lower()
    if len(w) > 4 and w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _toks(p):
    p = str(p or "").lower().replace("|", " ").replace("#", " ")
    p = p.replace("ping pong", "ping_pong").replace("ping-pong", "ping_pong")
    return [t for t in re.findall(r"[a-z_]+", p)]


def _head_part(t):
    """tokens before the first preposition ('woman in yellow' -> ['woman'])"""
    for k, w in enumerate(t):
        if w in PREP and k > 0:
            return t[:k]
    return t


def _syn(a, b):
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    return any(a in s and b in s for s in SYN)


def phrase_conflict(a, b):
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return False, "empty"
    if ta == tb:
        return False, "same"
    # parts: a phrase naming a part (rim, outer end, wheel ...) is not judged against its whole
    if (set(ta) & PARTS) or (set(tb) & PARTS):
        return False, "part"
    ca, cb = set(ta) & COLOURS, set(tb) & COLOURS
    if ca and cb and not (ca & cb):
        return True, f"colour {sorted(ca)} vs {sorted(cb)}"
    ma = {("ping_pong" if t == "ping_pong" else t) for t in ta if t in TYPEMOD or t == "ping_pong"}
    mb = {("ping_pong" if t == "ping_pong" else t) for t in tb if t in TYPEMOD or t == "ping_pong"}
    if ma and mb and not (ma & mb):
        return True, f"modifier {sorted(ma)} vs {sorted(mb)}"
    # compound type words fused with the head ('basketball' vs 'yoga ball'): the fused word is a type modifier + head
    ca_ = [t for t in _head_part(ta) if t not in DESCR and t not in COLOURS]
    cb_ = [t for t in _head_part(tb) if t not in DESCR and t not in COLOURS]
    if not ca_ or not cb_:
        return False, "descriptor-only"
    ha, hb = ca_[-1], cb_[-1]
    fused = {"basketball": "ball", "football": "ball", "volleyball": "ball", "baseball": "ball", "softball": "ball",
             "whiteboard": "board", "speedboat": "boat", "trashcan": "bin", "dustbin": "bin"}
    ma2 = set(ma) | ({ha} if ha in fused else set())
    mb2 = set(mb) | ({hb} if hb in fused else set())
    ha2, hb2 = fused.get(ha, ha), fused.get(hb, hb)
    if _syn(ha2, hb2) or _syn(ha, hb):
        if ma2 and mb2 and not (ma2 & mb2):
            return True, f"type {sorted(ma2)} vs {sorted(mb2)}"
        return False, "same head"
    # word-subset ('cells' vs 'red blood cell' handled above by head; 'moving speed' etc.)
    na, nb = {_norm(t) for t in ca_}, {_norm(t) for t in cb_}
    if na <= nb or nb <= na:
        return False, "subset"
    return True, f"head {ha} vs {hb}"


if __name__ == "__main__":
    tests = [("white car", "yellow car"), ("white car", "turning yellow car"), ("white car", "green car"),
             ("boat", "outer end"), ("person", "human"), ("person", "man"), ("bike", "moving bicycle"),
             ("red blood cell", "cells"), ("callisto model", "jupiter model"), ("typewriter", "gun"), ("eagle", "rock"),
             ("tennis ball", "ping pong ball"), ("tennis ball", "white ball"), ("yoga ball", "basketball"),
             ("yoga ball", "soccer ball"), ("trash bin", "rim"), ("trash bin", "book"), ("orange ball", "green ball"),
             ("whiteboard", "white box"), ("book", "slope"), ("boat", "person"), ("virus|right", "virus|left"),
             ("viruses", "virus"), ("bicycle", "bicycle wheel"), ("walking pedestrian", "human"), ("bicycle", "human"),
             ("cars", "black car"), ("speed boat", "speedboat"), ("white airplane", "car"), ("woman in yellow", "person")]
    for a, b in tests:
        print(f"{a!r:22} {b!r:22} -> {phrase_conflict(a, b)}")
