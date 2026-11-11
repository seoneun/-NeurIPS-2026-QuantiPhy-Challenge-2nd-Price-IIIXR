import json
import re

import numpy as np

from common import ROOT

FW = 0.5959
UNIT = {"cm": 100.0, "mm": 1000.0, "m": 1.0}


def key_names(phrase, key):
    ph, k = re.sub(r"[^a-z ]", "", phrase.lower()).strip(), re.sub(r"[^a-z ]", " ", key.lower()).strip()
    if not ph or not k:
        return False
    return ph == k or ph.replace(" ", "").startswith(k.replace(" ", "")) or ph.split()[-1] == k.split()[-1]


def centroid_at(c, fi):
    ks = sorted(c)
    lo = max([k for k in ks if k <= fi], default=None)
    hi = min([k for k in ks if k >= fi], default=None)
    if lo is None or hi is None or hi - lo > 3:
        return None
    if hi == lo:
        return np.array(c[lo], float)
    w = (fi - lo) / (hi - lo)
    return (1 - w) * np.array(c[lo], float) + w * np.array(c[hi], float)


def measure(p, split):
    q = p.get("q") or {}
    t1, t2 = q.get("t1"), q.get("t2")
    if t1 is None or t2 is None or q.get("kind") != "disp":
        return None, "not a timed displacement"
    keys = {}
    for e in p.get("depth") or []:
        if e.get("t") is not None and e.get("key"):
            keys.setdefault(e["key"], {})[round(float(e["t"]), 3)] = float(e["d"])
    both = [k for k, v in keys.items() if round(float(t1), 3) in v and round(float(t2), 3) in v]
    if len(both) != 1:
        return None, f"keys at both times {both}"
    k = both[0]
    path = ROOT / f"geo/out/tracks_{split}_v3ntmv/{p['video_id'].strip()}.json"
    if not path.exists():
        return None, "no track file"
    d = json.load(open(path))
    for qq in d["queries"]:
        if qq.get("obj") is None or not key_names(qq["phrase"], k):
            continue
        seq = d["objs"][str(qq["obj"])]
        c = {d["fidx"][j]: o["st"]["c"] for j, o in enumerate(seq) if isinstance(o, dict) and o.get("st")}
        uv = [centroid_at(c, float(t) * d["fps"]) for t in (t1, t2)]
        if any(x is None for x in uv):
            continue
        W, H = d["W"], d["H"]
        f = FW * W
        P = [keys[k][round(float(t), 3)] * np.array([(x[0] - W / 2) / f, (x[1] - H / 2) / f, 1.0]) for t, x in zip((t1, t2), uv)]
        return float(np.linalg.norm(P[1] - P[0])) * UNIT.get(q.get("unit"), 1.0), f"key {k}, track '{qq['phrase']}'"
    return None, 't170'
