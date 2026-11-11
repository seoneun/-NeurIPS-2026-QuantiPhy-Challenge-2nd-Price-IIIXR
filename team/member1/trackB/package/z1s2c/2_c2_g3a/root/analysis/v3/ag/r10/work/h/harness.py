import math, pickle, os
import numpy as np, pandas as pd
W = os.environ.get("QP_WORK", "")
ROWS = pickle.load(open(f"{W}/h/rows.pkl", "rb"))["rows"]

def base(sp, cur=None):
    if cur is not None: return cur[sp].copy()
    return pd.Series({i: r["s1"] for i, r in ROWS[sp].items()}).sort_index()

def apply(rule, sp, cur=None):
    import copy
    a = base(sp, cur); new = a.copy(); why = {}
    for i, r in ROWS[sp].items():
        rr = copy.deepcopy(r); rr["cur"] = float(a[i])
        v = rule(rr)
        if v is None: continue
        if isinstance(v, tuple): v, w = v
        else: w = ""
        if v is not None and np.isfinite(v) and v > 0 and abs(math.log(v / a[i])) > 1e-12:
            new[i] = float(v); why[i] = w
    return new, why
