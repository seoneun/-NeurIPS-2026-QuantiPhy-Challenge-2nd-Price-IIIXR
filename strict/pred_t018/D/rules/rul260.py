#!/usr/bin/env python
import sys, os, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t064"
FINAL = sys.argv[2] if len(sys.argv) > 2 else "pred_t053"
TAG = "rul260"
BLEND = {"rul263_dimgm", "rul263_unstable_gm", "dis_size_gm"}
BAND = np.log10(1.5)

for split in ("test",):
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    fin = pd.read_csv(f"{R}/submissions/{FINAL}_id1.csv").parsed_value.values if split == "test" else None
    P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")}
    out, n_skip = {}, {}
    for i in det.index:
        x = P.get(i)
        if x is None or not str(x["video_id"]).startswith("simulation"): continue
        if str(x.get("video_type", ""))[1:2] != "3" or not str(x.get("inference_type", "")).startswith("D"): continue
        if (x.get("q") or {}).get("kind") != "size": continue
        src = str(det.answer_source[i]).replace("mo_guard:", "")
        if src != "rul263_caw" and src not in BLEND: continue
        a, g = float(det.answer[i]), float(det.g[i]) if pd.notna(det.g[i]) else np.nan
        if not (g > 0 and a > 0):
            n_skip["no geo"] = n_skip.get("no geo", 0) + 1; continue
        if fin is not None and not np.isclose(fin[i], a, rtol=1e-9):
            n_skip["rule-written"] = n_skip.get("rule-written", 0) + 1; continue
        if src == "rul263_caw" and abs(np.log10(g / a)) > BAND:
            n_skip["CaW route, geo outside 1.5x"] = n_skip.get("CaW route, geo outside 1.5x", 0) + 1; continue
        v = float(np.sqrt(g * a))
        out[str(int(i))] = dict(value=v, why=f"{TAG}: CaW exponent halved on {src}: sqrt(g {g:.4g} x answer {a:.4g}) = {v:.4g}")
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
    print('')
