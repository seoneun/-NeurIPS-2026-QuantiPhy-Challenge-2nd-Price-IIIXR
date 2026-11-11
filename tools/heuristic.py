import os, sys, json, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--outdir", default=f"{R}/preds")
a = ap.parse_args()
df = load_split(a.split)
nfr = {}
for v, p in df.groupby("video_id").video_path.first().items():
    c = cv2.VideoCapture(p); nfr[v] = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
df["T"] = [nfr[v] / float(f) for v, f in zip(df.video_id, df.fps)]
tu = [target_unit(q, it) for q, it in zip(df.question, df.inference_type)]
pv = [prior_value(p) for p in df.ground_truth_prior]
def convert(si, k_from, k_to, tau):
    if k_from == k_to: return si
    if tau is None: return si
    if (k_from, k_to) == (0, 1): return si / tau
    if (k_from, k_to) == (1, 0): return si * tau
    if (k_from, k_to) == (2, 0): return 0.5 * si * tau ** 2
    if (k_from, k_to) == (0, 2): return 2 * si / tau ** 2
    if (k_from, k_to) == (1, 2): return si / tau
    if (k_from, k_to) == (2, 1): return si * tau
outs = {}
for name in ["h0_const1", "h1_prior_copy", "h2_prior_dur"]:
    vals = []
    for (si, k), u, T in zip(pv, tu, df["T"]):
        f, kt = UNIT_TABLE[u]
        if name == "h0_const1" or si is None:
            vals.append(1.0); continue
        tau = 1.0 if name == "h1_prior_copy" else T
        vals.append(convert(si, k, kt, tau) / f)
    o = df.copy(); o["parsed_value"] = vals
    path = f"{a.outdir}/{a.split}_{name}.csv"
    o[OUT_COLS].to_csv(path, index=False); outs[name] = path
print("prior parse failures:", sum(si is None for si, k in pv), "of", len(pv))
print("target units:", pd.Series(tu).value_counts().to_dict())
print(json.dumps(outs))
