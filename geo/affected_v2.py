import json, sys, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/geo")
from measure import contrastive
split = sys.argv[1]
recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
m = pd.read_csv(f"{R}/geo/out/meas_{split}.csv").set_index("id")
byv = {}
for r in recs:
    byv.setdefault(r["video_id"], []).append(r)
reasons = {}
for v, rs in byv.items():
    rsn = set()
    if any(r["v2_changed"] for r in rs):
        rsn.add("A1_parse")
    try:
        vj = json.load(open(f"{R}/geo/out/tracks_{split}/{v}.json"))
    except FileNotFoundError:
        rsn.add("no_v1_track"); reasons[v] = rsn; continue
    for q in vj["queries"]:
        ious = q.get("det_iou") or []
        if q.get("best_score", 0) < 0.3 or (ious and np.median(ious) < 0.5):
            rsn.add("A2_det")
    if any(str(m.why.get(r["id"])) in ("target=prior_object", "target=target2") for r in rs):
        rsn.add("A3_coll")
    qs = [q for q in vj["queries"] if q.get("anchor")]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = qs[i]["anchor"], qs[j]["anchor"]
            if a[0] != b[0]:
                continue
            x, y = a[1], b[1]
            ix = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
            ua = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - ix
            pa = qs[i]["phrase"] + " " + " ".join(qs[i]["where"]); pb = qs[j]["phrase"] + " " + " ".join(qs[j]["where"])
            if ua > 0 and ix / ua > 0.9 and contrastive(pa, pb):
                rsn.add("A3_coll")
    if any(r["q"]["kind"] == "dist2" for r in rs):
        rsn.add("A4_dist")
    reasons[v] = rsn
aff = sorted(v for v, s in reasons.items() if s)
open(f"{R}/geo/out/retrack_{split}_v2.txt", "w").write("\n".join(aff) + "\n")
cnt = {}
for v in aff:
    for s in reasons[v]:
        cnt[s] = cnt.get(s, 0) + 1
nrows = sum(len(byv[v]) for v in aff)
print(json.dumps(dict(split=split, videos=len(byv), affected_videos=len(aff), rows_in_affected=nrows, by_reason=cnt)))
