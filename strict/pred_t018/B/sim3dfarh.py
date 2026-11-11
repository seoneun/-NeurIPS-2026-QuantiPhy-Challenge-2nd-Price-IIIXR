#!/usr/bin/env python
import sys, json, csv
import numpy as np, pandas as pd
DET, MD, ROWS = sys.argv[1:4]
DMIN = 50.0
want = [int(r["id0"]) for r in csv.DictReader(open(ROWS)) if r["tag"] == "sim3dfarh"]
det = pd.read_csv(DET, index_col=0)
P = {json.loads(l)["id"]: json.loads(l) for l in open("geo/out/parse_test_v2.jsonl")}
M = {v: pd.read_csv(f"{MD}/ag_qtrk_meas_test_{v}_solo_q3.csv").set_index("id") for v in ("v", "m")}
for i in want:
    x = P[i]
    if not str(x["video_id"]).startswith("simulation"): continue
    cat = x["inference_type"][0] + x["video_type"][1]
    if cat != "D3" or (x.get("q") or {}).get("kind") != "size" or (x.get("q") or {}).get("dim") != "height": continue
    tk = x.get("target_dkeys") or []
    dd = [d["d"] for d in x.get("depth") or [] if d["key"] in tk and d.get("d")]
    if not dd or float(np.mean(dd)) < DMIN: continue
    if str(det.answer_source[i]).startswith("patch"): continue
    g, cs = float(det.g[i]), float(det.cs[i])
    if not (g > 0 and cs > 0) or abs(np.log(g / cs)) <= np.log(2): continue
    mode = None
    for v in ("v", "m"):
        r = M[v].loc[i] if i in M[v].index else None
        if r is not None and pd.notna(r.val_si) and r.val_si > 0 and abs(np.log(r.val_si / g)) < 0.01:
            mode = str(r["mode"]); break
    if mode is None or not (mode.startswith("3d_calib") and "rejected" not in mode or mode.startswith("3d_personf")): continue
    if abs(np.log(g / det.answer[i])) > 0.005:
        print(f"{i},{g!r}")
