import json, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/analysis/v3")
from lab_rows import labelled
L = labelled("sub_xt_Nzz29")
for s in ("test", "val"):
    pt = json.load(open(f"{R}/cache/tmp/ag_c95priorpmovew_patch_{s}.json"))
    Ls = L[L.split == s]
    hit = [int(k) for k in pt if int(k) in Ls.index]
    print(s, "patch", len(pt), "labelled hits", hit, "labelled rows in split", len(Ls))
