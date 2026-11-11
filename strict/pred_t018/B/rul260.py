#!/usr/bin/env python
import sys, json, csv
import numpy as np, pandas as pd
DET, ROWS = sys.argv[1:3]
TAG = "rul260"
BLEND = {"rul263_dimgm", "rul263_unstable_gm", "dis_size_gm"}
BAND = np.log10(1.5)
want = [int(r["id0"]) for r in csv.DictReader(open(ROWS)) if r["tag"] == TAG]
det = pd.read_csv(DET, index_col=0)
P = {json.loads(l)["id"]: json.loads(l) for l in open("geo/out/parse_test_v2.jsonl")}
for i in want:
    x = P[i]
    if not str(x["video_id"]).startswith("simulation"): continue
    if str(x.get("video_type", ""))[1:2] != "3" or not str(x.get("inference_type", "")).startswith("D"): continue
    if (x.get("q") or {}).get("kind") != "size": continue
    src = str(det.answer_source[i]).replace("mo_guard:", "")
    if src != "rul263_caw" and src not in BLEND: continue
    a, g = float(det.answer[i]), float(det.g[i]) if pd.notna(det.g[i]) else np.nan
    if not (g > 0 and a > 0): continue
    if src == "rul263_caw" and abs(np.log10(g / a)) > BAND: continue
    print(f"{i},{float(np.sqrt(g * a))!r}")
