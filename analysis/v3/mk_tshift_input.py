import json
import os
import re
import sys

import cv2
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/tools")
from qp_common import TEST_VID

d = pd.read_csv(f"{R}/caw/inputs/test_input.csv", index_col=0)

ts, fps = {}, {}
for ln in open(f"{R}/geo/out/parse_test_v2.jsonl"):
    x = json.loads(ln)
    v = str(x["video_id"]).strip()
    fps[v] = float(x["fps"])
    q = x.get("q") or {}
    ts.setdefault(v, []).extend(float(q[k]) for k in ("t", "t1", "t2") if q.get(k) is not None)


def n_frames(v):
    p = os.path.join(TEST_VID, f"{v}.mp4")
    if not os.path.exists(p):
        sys.exit('')
    c = cv2.VideoCapture(p)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n


shift = {}
for v, t in sorted(ts.items()):
    if not t:
        continue
    n = n_frames(v)
    if n <= 0 or not fps.get(v):
        continue
    dur = n / fps[v]
    if max(t) <= dur + 0.5 or min(t) - dur < 0 or max(t) - dur > dur:
        continue
    shift[v] = round(dur, 2)
    print('')


def retime(q, s):
    return re.sub(r"(\d+(?:\.\d+)?)\s*s\b", lambda k: f"{float(k.group(1)) - s:.2f}s", q)


vid = d.video_id.astype(str).str.strip()
z = d[vid.isin(shift)].copy()
z["question"] = [retime(q, shift[v]) for q, v in zip(z.question, vid[z.index])]
z.to_csv(f"{R}/caw/inputs/test_tshift.csv")
print(len(z)); print(z.question.head(10).to_string())
