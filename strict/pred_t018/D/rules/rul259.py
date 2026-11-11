#!/usr/bin/env python
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t060"; GT = sys.argv[2] if len(sys.argv) > 2 else "c17"
MT = "P"; TAG = "rul259"
WG = 0.9; FSPEC = 0.5959
SPH = re.compile(r"\b(?:ball|balls|basketball|sphere|globe|balloon)\b", re.I)
UNIT = {"cm": 100.0, "m": 1.0, "mm": 1000.0}


def agg(x, dim):
    return float(np.percentile(x, 75)) if dim in ("diameter", "radius") else float(np.median(x))


for split in ("test",):
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    G = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")}
    M = {(v, t): pd.read_csv(f"{R}/cache/tmp/ag_lab3d_meas_{split}_{v}_{t}{MT}.csv").set_index("id") for v in "mv" for t in ("ref",)}
    DU = {(v, t): {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/cache/tmp/ag_lab3d_dump_{split}_{v}_{t}{MT}.jsonl")}
          for v in "mv" for t in ("ref", "flab")}
    out = {}
    for i in det.index:
        x = P.get(i)
        if x is None or not str(x["video_id"]).startswith("captured") or det.cat[i] != "D3" or (x.get("q") or {}).get("kind") != "size": continue
        src = str(det.answer_source[i])
        if src not in ("d3lab_gm", "d3lab_inband_gm"): continue
        gv, cs = float(G.geo_value.get(i, np.nan)), float(det.cs[i])
        if not (gv > 0 and cs > 0): continue
        best = (None, np.inf)
        for v in "mv":
            m = M[(v, "ref")]
            if i in m.index and bool(m.ok[i]) and m.value[i] > 0 and abs(np.log(m.value[i] / gv)) < best[1]:
                best = (v, abs(np.log(m.value[i] / gv)))
        if best[0] is None or best[1] > 0.1: continue
        rf, fl = DU[(best[0], "ref")].get(i), DU[(best[0], "flab")].get(i)
        if not rf or not fl or len(fl["e"]) == 0 or len(fl["e"]) != len(fl["D"]): continue
        f = FSPEC * fl["W"]; e = np.array(fl["e"], float); Dd = np.array(fl["D"], float); uv = np.array(fl["uv"], float)
        nr = np.sqrt(1 + ((uv[:, 0] - fl["cx"]) / f) ** 2 + ((uv[:, 1] - fl["cy"]) / f) ** 2)
        s = e * Dd / nr / f
        sph = bool(SPH.search(str((x.get("target") or {}).get("phrase") or "")))
        if sph:
            s = s / (1 - e / (2 * f * nr))
        g = gv * agg(s, fl["dim"]) / rf["val_si"]
        val = g ** WG * cs ** (1 - WG)
        if abs(np.log(val / det.answer[i])) > 0.005:
            out[str(i)] = dict(value=float(val), why=f"{TAG}: D3 lab size ({src}), re-measured at spec f/W {FSPEC} on own track/depth"
                               f"{' + near-surface sphere depth' if sph else ''}: geo {gv:.4g} -> {g:.4g}; CaW {cs:.4g} -> g^{WG} cs^{1 - WG:.1f}; was {det.answer[i]:.4g}")
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
    print(split, len(out), "rows")
