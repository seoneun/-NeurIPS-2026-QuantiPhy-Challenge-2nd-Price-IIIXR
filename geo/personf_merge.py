#!/usr/bin/env python
import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools")
from qp_common import OUT_COLS
split, new, ref, out = sys.argv[1:5]
A = pd.read_csv(f"{R}/preds/{split}_{new}_raw.csv").set_index("Unnamed: 0")
B = pd.read_csv(f"{R}/preds/{split}_{ref}_raw.csv").set_index("Unnamed: 0").reindex(A.index)
PER = re.compile(r"person|people|\bman\b|woman|girl|boy|human|player|child|kid|lady|guy", re.I)
P = {}
for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
    x = json.loads(ln); P[int(x["id"])] = x
pf = A.meas_mode.astype(str).str.startswith("3d_personf")
ga, gb = pd.to_numeric(A.geo_value, errors="coerce"), pd.to_numeric(B.geo_value, errors="coerce")
with np.errstate(divide="ignore", invalid="ignore"):
    r = ga / gb
ra = pf & B.meas_mode.astype(str).str.startswith("3d_calib") & (B.answer_source == "meas") & (r >= 0.95) & (r <= 1.05)
rc = pd.Series(False, index=A.index)
for i in A.index[pf]:
    x = P.get(int(i)) or {}
    q = x.get("q") or {}
    if q.get("kind") in ("speed", "avgspeed", "hvel", "path", "disp") and any(PER.search(k) for k in (x.get("target_dkeys") or [])):
        rc[i] = True
back = ra | rc
o = A.copy()
o.loc[back] = B.loc[back]
o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
o.to_csv(f"{R}/preds/{split}_{out}_raw.csv")
o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_{out}.csv", index=False)
print('')
