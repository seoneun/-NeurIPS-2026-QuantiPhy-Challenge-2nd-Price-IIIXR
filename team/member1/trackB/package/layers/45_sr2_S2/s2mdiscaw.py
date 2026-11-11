import sys, json
import numpy as np, pandas as pd
DET, CUR, PARSE, OUT = sys.argv[1:5]
MOTION = ("speed", "accel", "avgspeed", "disp", "dist2", "path")
det = pd.read_csv(DET, index_col=0)
P = {r["id"]: r for r in map(json.loads, open(PARSE))}
cur = pd.read_csv(CUR).parsed_value.values
out = {}
for pos, (i, r) in enumerate(det.iterrows()):
    i = int(i); x = P.get(i) or {}
    if x.get("cat") != "S2" or not str(r.video_id).startswith("internet") or str(r.answer_source) != "dis_caw_routed":
        continue
    if (x.get("q") or {}).get("kind") not in MOTION:
        continue
    g, a = float(r.g), float(cur[pos])
    if not (np.isfinite(g) and g > 0):
        continue
    if not np.isclose(a, float(r.answer), rtol=1e-9):
        continue
    out[str(i)] = dict(value=g)
json.dump(out, open(OUT, "w"), indent=0)
