import json, os, sys
import numpy as np
R = os.path.abspath(os.environ["QP_ROOT"])
sys.path.insert(0, f"{R}/geo")
from rules import UNIT_SI

TRK = "_v2sc"
MIN_PX = 8.0
FLAT = 3.0
PL_SIZE = (1e-4, 500.0)


def ext_stats(split, vid, key):
    p = f"{R}/geo/out/tracks_{split}{TRK}/{vid}.json"
    if not os.path.exists(p):
        return None
    T = json.load(open(p))
    if not isinstance(T.get("queries"), list) or "objs" not in T:
        return None
    qs = [q for q in T["queries"] if q.get("key") == key]
    if not qs:
        return None
    rows = []
    for x in T["objs"][str(qs[0]["obj"])]:
        s = x.get("st") if isinstance(x, dict) and "st" in x else x
        if not s or s.get("border", 0) != 0:
            continue
        b = s["box"]
        rows.append((b[2] - b[0], b[3] - b[1], s["rr"][0], s["rr"][1]))
    if len(rows) < 3:
        return None
    a = np.array(rows, float)
    return dict(bw=float(np.median(a[:, 0])), bh=float(np.median(a[:, 1])), rr0=float(np.median(a[:, 2])),
                rr1=float(np.median(a[:, 3])), rr1p75=float(np.percentile(a[:, 3], 75)), n=len(a))
