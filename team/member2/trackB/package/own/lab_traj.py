import json
import re

import numpy as np
import pandas as pd

from lab3d import norm


def load_objs(vid):
    d = json.load(open(f"geo/out/tracks_test_v3ntmvu/{vid}.json"))
    seen, objs = set(), []
    for q in d["queries"]:
        oid = str(q.get("obj"))
        if oid in seen or oid not in d["objs"]:
            continue
        seen.add(oid)
        rows = []
        for fr in d["objs"][oid]:
            st = fr.get("st") if isinstance(fr, dict) else None
            rows.append([st["c"][0], st["c"][1], st["box"][3] - st["box"][1]] if st else [np.nan] * 3)
        objs.append((norm(q["phrase"]), pd.DataFrame(np.array(rows, float), columns=["cx", "cy", "bh"])))
    return dict(W=d["W"], H=d["H"], fidx=np.array(d["fidx"]), objs=objs)


def pick(objs, name):
    name = norm(name)
    side = "left" if re.search(r"\bleft\b", name) else "right" if re.search(r"\bright\b", name) else None
    core = re.sub(r"\b(left|right|the|rolling|moving)\b", "", name).strip()
    toks = [t for t in core.split() if len(t) > 2]
    c = [(p, o) for p, o in objs if p == core] or [(p, o) for p, o in objs if toks and all(t in p for t in toks)]
    if not c:
        c = [(p, o) for p, o in objs if any(t in p for t in toks if t not in ("ball",))]
    if not c and "ball" in core:
        c = [(p, o) for p, o in objs if "ball" in p]
    if not c:
        return None
    return c[0] if len(c) == 1 or side is None else None
