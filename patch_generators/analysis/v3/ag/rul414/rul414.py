#!/usr/bin/env python
"""rul414: a planar (S2/D2) size question about a scene element (meas_why scene_target, answered by CaW alone) whose
element has a VLM-anchored track (v3ntmv) is measured as a pixel ratio to the prior object's own VLM track: the scene box side
on the axis that is not cut by the frame border, times metres per pixel from the prior object's box. The prior's reference
side is the box's long side when its dimension is a height/length (a person seen from above lies along the long side), else
the short side. Only fires when the prior object is a size prior with a track and the scene box is cut on at most one axis.
usage: python rul414.py [write]"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul414"; WRITE = "write" in sys.argv[1:]
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}

def obj_of(tr, phrase):
    ph = str(phrase or "").strip().lower()
    for qk in ("queries", "queries_d"):
        for q in tr.get(qk) or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("obj") is not None:
                return q["obj"]
    return None

def boxes(tr, obj):
    b = [s["st"]["box"] for s in (tr["objs"].get(str(obj)) or []) if s and s.get("st")]
    return np.array(b) if len(b) >= 0.8 * len(tr["fidx"]) else None

def rule(sp):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
    PP = {p["id"]: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    out = {}
    for i, r in A.iterrows():
        p = PP[i]; q = p["q"]
        if not (r["cat"] in ("S2", "D2") and q.get("kind") == "size" and str(r.source) == "caw_nogeo" and str(r.meas_why) == "scene_target" and q.get("unit") in UNIT):
            continue
        pr = [x for x in p["priors"] if x.get("kind") == "size" and x.get("value_si")]
        if len(pr) != 1: continue
        pr = pr[0]
        fn = f"{R}/geo/out/tracks_{sp}_v3ntmv/{r.video}.json"
        if not os.path.exists(fn): continue
        tv = json.load(open(fn)); W, H = tv["W"], tv["H"]
        to = obj_of(tv, (p.get("target") or {}).get("phrase")); po = obj_of(tv, (p.get("prior_obj") or {}).get("phrase"))
        if to is None or po is None or to == po: continue
        tb, pb = boxes(tv, to), boxes(tv, po)
        if tb is None or pb is None: continue
        tw, th = np.median(tb[:, 2] - tb[:, 0]), np.median(tb[:, 3] - tb[:, 1])
        cutx = np.median((tb[:, 0] <= 1) | (tb[:, 2] >= W - 2)) > 0.5; cuty = np.median((tb[:, 1] <= 1) | (tb[:, 3] >= H - 2)) > 0.5
        if cutx and cuty: continue
        ext = th if cutx else tw if cuty else (tw if q.get("dim") in ("width", "length", "diameter") else th)
        pw, ph_ = np.median(pb[:, 2] - pb[:, 0]), np.median(pb[:, 3] - pb[:, 1])
        pref = max(pw, ph_) if pr.get("dim") in ("height", "length", None) else min(pw, ph_)
        if pref < 20 or ext < 20: continue
        v = float(ext * pr["value_si"] / pref) / UNIT[q["unit"]]
        out[i] = (v, f"scene box {ext:.0f} px on the uncut axis x {pr['value_si']:.3g} m / prior box {pref:.0f} px (prior {p.get('prior_obj', {}).get('phrase')}); CaW {r.cs:.3g} replaced")
    return out

if __name__ == "__main__":
    res = {sp: rule(sp) for sp in ("test", "val")}
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0")
    for i, (v, why) in res["test"].items(): print(f"test {i} {A.video[i]} {A['cat'][i]} scored {A.scored[i]} | {A.question[i][:60]} | old {A.answer[i]:.4g} new {v:.4g} | {why}")
    V = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv").set_index("id0")
    for i, (v, why) in res["val"].items(): print(f"val {i} {V.video[i]} y {V.y[i]:.4g} old {V.answer[i]:.4g} new {v:.4g}: MRA {V.mra[i]:.1f} -> {float(FC.row_mra(np.array([v]), np.array([V.y[i]]))[0]):.1f}")
    if WRITE:
        for sp in res:
            json.dump({str(i): dict(value=round(float(v), 6), why=f"{TAG}: {why}") for i, (v, why) in res[sp].items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print("written")
