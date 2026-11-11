#!/usr/bin/env python
import sys
import pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/analysis/v3")
from variant_keys import split_variant
t = pd.read_csv(f"{R}/caw/inputs/test_input.csv")
ids = set(t.video_id.astype(str).str.strip())
fps_of = t.assign(v=t.video_id.astype(str).str.strip()).groupby("v").fps.first().to_dict()
src_of = t.assign(v=t.video_id.astype(str).str.strip()).groupby("v").video_source.first().to_dict()
rows = []
for _, r in t.iterrows():
    b, var = split_variant(r.video_id)
    if var not in ("x", "_segmented"):
        continue
    sib = None
    for cand in (b, b.upper(), b.capitalize()):
        if cand in ids:
            sib = cand; break
    if sib is None:
        continue
    o = r.copy(); o["video_id"] = sib; o["fps"] = fps_of[sib]; o["video_source"] = src_of[sib]
    rows.append(o)
out = pd.DataFrame(rows)
cols = list(t.columns)
out = out[cols].rename(columns={cols[0]: ""})
out.to_csv(f"{R}/caw/sc/inputs/test_sib.csv", index=False)
print('')
