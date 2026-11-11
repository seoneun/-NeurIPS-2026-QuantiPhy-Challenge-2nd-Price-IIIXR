#!/usr/bin/env python
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
A = pd.read_csv(sys.argv[1]); a = A.parsed_value.values
D = pd.read_csv(sys.argv[2], index_col=0)
g = D.g.values.astype(float); cs = D.cs.values.astype(float)
cat = (A.inference_type.str[0] + A.video_type.str[1]).values; vid = A.video_id.astype(str).values
unp = np.isclose(a, D.answer.values, rtol=1e-9) & ~D.answer_source.astype(str).str.startswith("patch").values
both = (g > 0) & (cs > 0)
simplain = np.array([v.startswith("simulation") and not v.endswith("_segmented") for v in vid])
b2 = (cat == "S2") & simplain & unp & both
out = a.copy(); out[b2] = g[b2]
o = A.copy(); o["parsed_value"] = out; o.to_csv(f"{R}/submissions/{sys.argv[3]}_id1.csv", index=False)
print(sys.argv[3], "S2 block", int(b2.sum()))
