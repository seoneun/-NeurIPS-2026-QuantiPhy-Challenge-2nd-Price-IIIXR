import json, re, sys
import pandas as pd, numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")); sys.path.insert(0, R + "/cache/tmp")
from lab_rows import labelled
L = labelled("sub_xt_Nzz27")
P = {}
for sp in ("test", "val"):
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); P[(sp, r["id"])] = r
for i, row in L.iterrows():
    r = P.get((row.split, int(i)))
    if r is None: continue
    q = r.get("q") or {}
    t = (q.get("target") or "").lower()
    if q.get("kind") == "size" and ("ping" in t or "white ball" in t):
        print(row.split, i, r["video_id"], row.cat, t, q.get("dim"), q.get("unit"), "y", row.y, "ans", row.answer if "answer" in row else None, "m", row.m)
D = pd.read_csv(f"{R}/submissions/sub_xt_Nzz27_detail.csv", index_col=0)
print(D.loc[[2722, 2729, 2736], ["video_id", "question", "cat", "answer", "answer_source", "g", "cs"]].to_string())
