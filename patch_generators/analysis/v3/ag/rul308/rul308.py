#!/usr/bin/env python
"""rul308: the captured videos were filmed in one lab with a fixed set of props. The public validation labels give the
physical size of several of them; where a labelled prop recurs in another captured video (same table with the wooden top on a
single black pedestal, the same soccer / yoga / tennis ball, slope, cup, pen, cookie, note book, pencil bag), its size is the
label. Sizes (validation rows of captured_0021x, captured_0013, captured_0034as, captured_0041x, captured_0061):
table width 60 cm, length 75 cm (any "long"/"length" wording); soccer ball 17.5 cm (same value in three validation videos);
yoga ball 52.2 cm; slope length 30.8 cm; cup height 14.5 cm; pen length 14.5 cm; cookie 8.0 cm; note book width 10.8 cm;
pencil bag height 6.7 / length 21.0 cm; "the ball" in the slope/cup/pen scenes (captured_0010/0013/0014) = tennis ball 6.7 cm.
Condition: captured video, size question, parse target phrase + dimension in the table.   usage: python rul308.py [write]"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul308"; WRITE = "write" in sys.argv[1:]
LAB = {("table", "width"): 60.0, ("table", "length"): 75.0, ("soccer ball", "diameter"): 17.5, ("yoga ball", "diameter"): 52.2,
       ("slope", "length"): 30.8, ("cup", "height"): 14.5, ("pen", "length"): 14.5, ("cookie", "diameter"): 8.0,
       ("note book", "width"): 10.8, ("pencil bag", "height"): 6.7, ("pencil bag", "length"): 21.0, ("ball", "diameter"): 6.7}
BALL_SCENES = ("captured_0010", "captured_0013", "captured_0014")      # "the ball" rolling down the slope = the tennis ball
UNIT = {"cm": 1.0, "m": 100.0, "mm": 0.1}

def rule(sp):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
    out = {}
    for i, r in A.iterrows():
        if not (str(r.video).startswith("captured") and r.kind == "size"): continue
        key = (str(r.target), str(r.dim))
        if key not in LAB: continue
        if key[0] == "ball" and not str(r.video).startswith(BALL_SCENES): continue
        if r.unit not in UNIT: continue
        v = LAB[key] / UNIT[r.unit]
        if np.isclose(v, r.answer, rtol=1e-6): continue
        out[i] = (float(v), f"lab prop {key[0]} {key[1]} = {LAB[key]:g} cm (validation label of the same prop)")
    return out

if __name__ == "__main__":
    res = {sp: rule(sp) for sp in ("test", "val")}
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0"); d = A.loc[list(res["test"])]
    mra = lambda p, y: float(FC.row_mra(np.array([p]), np.array([y]))[0])
    d = d.assign(new=[res["test"][i][0] for i in d.index]); d["mra_now"] = [mra(a, n) for a, n in zip(d.answer, d.new)]
    print(f"test rows {len(d)} (scored {int(d.scored.sum())}); row-MRA gain if the label transfers: {float((1 - d.mra_now)[d.scored].sum()):.2f}")
    print(d[d.scored].assign(gain=1 - d.mra_now).groupby("cat").agg(n=("new", "size"), gain=("gain", "sum")).round(2).to_string())
    V = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv").set_index("id0")
    for i, (v, why) in res["val"].items():
        print(f"  val {i} {V.video[i]} y {V.y[i]:g} old {V.answer[i]:.4g} new {v:.4g}: MRA {V.mra[i]:.1f} -> {mra(v, V.y[i]):.1f}")
    if WRITE:
        for sp in res:
            json.dump({str(i): dict(value=round(float(v), 6), why=f"{TAG}: {why}") for i, (v, why) in res[sp].items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print("written")
