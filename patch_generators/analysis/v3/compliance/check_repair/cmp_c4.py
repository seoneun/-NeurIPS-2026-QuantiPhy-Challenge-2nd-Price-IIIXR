#!/usr/bin/env python
"""cmp_c4.py OUT.csv TAG [TAG ...]: same-rule table of the C4 patches (cache/tmp/ag_<tag>C4_patch_{test,val}.json) against the
stored production patch (patches/ag_<tag>_patch_<split>.json, the cache/tmp copy if absent). Columns per tag and split:
n (C4 rows), stored_n, both_n / both_eq (common rows / of them equal, rel 1e-6), unch_n / unch_eq (common rows whose geo input
did not change: ffmv geo_value equal to rj5sQ2 geo_value rel 1e-9 (both missing = equal) and the same meas_mode), C2, C3
(criteria of notes/clean_rerun.md 5.1). The stored values are read only for this comparison."""
import sys, os, json
import numpy as np, pandas as pd
out, tags = sys.argv[1], sys.argv[2:]
def load(p): return {int(k): float(v["value"]) for k, v in json.load(open(p)).items()} if os.path.exists(p) else None
U = {}
for sp in ("test", "val"):
    c = lambda x: x in ("Unnamed: 0", "geo_value", "meas_mode")
    g0 = pd.read_csv(f"preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0, usecols=c)
    g1 = pd.read_csv(f"preds/{sp}_geo_v4ntffmvsrngsr_d_vw_raw.csv", index_col=0, usecols=c).reindex(g0.index)
    a, b = g0.geo_value.astype(float).values, g1.geo_value.astype(float).values
    same = (np.isclose(a, b, rtol=1e-9) | (np.isnan(a) & np.isnan(b))) & (g0.meas_mode.astype(str).values == g1.meas_mode.astype(str).values)
    U[sp] = dict(zip(g0.index.astype(int), same))
rows = []
for t in tags:
    for sp in ("test", "val"):
        n = load(f"cache/tmp/ag_{t}C4_patch_{sp}.json") or {}
        s = load(f"patches/ag_{t}_patch_{sp}.json") or load(f"cache/tmp/ag_{t}_patch_{sp}.json") or {}
        both = sorted(set(n) & set(s)); eq = [k for k in both if np.isclose(n[k], s[k], rtol=1e-6)]
        un = [k for k in both if U[sp].get(k, False)]; ueq = [k for k in un if k in eq]
        rows.append(dict(tag=t, split=sp, n=len(n), stored_n=len(s), both_n=len(both), both_eq=len(eq), unch_n=len(un),
                         unch_eq=len(ueq), C2=len(un) >= 1 and len(ueq) == len(un), C3=len(both) >= 1 and len(eq) == len(both),
                         new_rows=" ".join(map(str, sorted(set(n) - set(s)))), lost_rows=" ".join(map(str, sorted(set(s) - set(n))))))
D = pd.DataFrame(rows); D.to_csv(out, index=False)
pd.set_option("display.width", 250); print(D[D.split == "test"].drop(columns="split").to_string(index=False))
