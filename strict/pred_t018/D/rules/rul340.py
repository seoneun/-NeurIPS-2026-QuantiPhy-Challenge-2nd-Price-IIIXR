#!/usr/bin/env python
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
F, CUR, GT = sys.argv[1:4]
TAG = "rul340"
HGT = re.compile(r"\b(?:height|tall)\b", re.I)
RATIO = 1.5
CAWSRC = {"rul263_caw", "caw_nogeo", "dis_caw_routed", "caw_c1"}
P = {int(json.loads(l)["id"]): json.loads(l) for l in open(f"{R}/geo/out/parse_test_v2.jsonl")}
G = pd.read_csv(f"{R}/preds/test_geo_v4nt{GT}srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
D = pd.read_csv(f"{R}/submissions/{F}_detail.csv", index_col=0)
cur = pd.read_csv(f"{R}/submissions/{CUR}_id1.csv").parsed_value
out = {}
for i in D.index:
    r = P.get(int(i))
    if r is None or r["cat"] != "D3" or r["video_source"] != "simulation" or (r.get("q") or {}).get("kind") != "size":
        continue
    mm = str(G.meas_mode.get(i, ""))
    if "fovdefault" not in mm and "mogefov" not in mm:
        continue
    a = float(D.answer[i]); src = str(D.answer_source[i]); g = float(D.g[i]); cs = float(D.cs[i]); c = float(cur[i])
    if not (np.isfinite(g) and g > 0 and np.isfinite(cs) and cs > 0) or g / cs < RATIO:
        continue
    if src in CAWSRC or not np.isclose(c, a, rtol=1e-9):
        continue
    v = cs * (1.05 if HGT.search(r["question"]) else 1.1)
    if abs(v / c - 1) <= 1e-3:
        continue
    out[str(int(i))] = dict(value=float(v))
json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_test.json", "w"), indent=0)
print(f"{TAG}: {len(out)} rows")
