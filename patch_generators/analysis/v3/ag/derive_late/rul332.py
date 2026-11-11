"""rul332: pendacc applied raise-only.

The labelled lab analogue on the same rig (captured_0027: a(0.2 s) = 10.86 m/s^2,
v(0.2 s) = 2.02 m/s) runs above the ideal-pendulum values. So the ideal |a| is
used only as a floor: answer = max(current answer, pendacc value). That keeps
the rows where the current answer is far below any pendulum acceleration
(1.47, 5.31) and leaves the g-like answers (8.97, 9.09) alone.
Inputs: the pendacc patch and the current build's detail file. No labels.
"""
import json, sys
import pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
base = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzm"
for split, det in (("test", f"{R}/submissions/{base}_detail.csv"), ("val", f"{R}/submissions/{base}_valdetail.csv")):
    src = json.load(open(f"{R}/cache/tmp/ag_pendacc_patch_{split}.json"))
    cur = pd.read_csv(det).set_index("Unnamed: 0")["answer"]
    out = {}
    for k, v in src.items():
        a = float(cur.loc[int(k)])
        if v["value"] > a:
            out[k] = {"value": v["value"], "why": f"rul332: raise-only floor {v['value']:.3f} > current {a:.3f}; " + v["why"]}
    json.dump(out, open(f"{R}/cache/tmp/ag_pendacc_up_patch_{split}.json", "w"), indent=1)
    print(split, {k: round(o["value"], 3) for k, o in out.items()})
