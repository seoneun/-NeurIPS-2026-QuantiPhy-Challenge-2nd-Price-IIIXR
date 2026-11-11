#!/usr/bin/env python
import sys, json, csv
import numpy as np, pandas as pd
DET, GEO, ROWS = sys.argv[1:4]
want = [int(r["id0"]) for r in csv.DictReader(open(ROWS)) if r["tag"] == "macroscaleP"]
D = pd.read_csv(DET).set_index("Unnamed: 0")
G = pd.read_csv(GEO).set_index("Unnamed: 0").reindex(D.index)
PP = {p["id"]: p for p in map(json.loads, open("geo/out/parse_test_v2.jsonl"))}
for i in want:
    p = PP[i]; q = p["q"]
    if not (str(G.video_id[i]).startswith("simulation") and q.get("kind") == "size" and q.get("unit") in ("m", "cm", "mm")):
        continue
    pr = [x for x in p["priors"] if x.get("kind") == "size" and x.get("value_si")]
    if len(pr) != 1 or pr[0]["value_si"] >= 0.01:
        continue
    src, mode = str(D.answer_source[i]), str(G.meas_mode[i])
    ans, g = float(D.answer[i]), float(D.g[i])
    if src.startswith("patch"):
        continue
    if mode in ("2d_ratio", "vlm_2d_ratio") and np.isfinite(g) and g > 0 and src in ("dis_size_gm", "agree_gm", "agree_wgm", "caw_c1", "dis_caw_routed"):
        if not np.isclose(ans, g, rtol=1e-6) and not (np.isfinite(ans) and ans != 0 and abs(g / ans - 1.0) <= 0.005):
            print(f"{i},{round(float(g), 7)!r}")
