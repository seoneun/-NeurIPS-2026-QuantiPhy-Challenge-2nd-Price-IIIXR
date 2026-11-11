import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
TAG = "rul197"
PERSON = re.compile(r"\b(pedestrian|person|human|man|woman|people|rider)\b", re.I)


def load_parse(sp):
    P = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        d = json.loads(l)
        ph = []
        t = d.get("target")
        if isinstance(t, dict) and t.get("phrase"): ph.append(str(t["phrase"]))
        q = d.get("q") or {}
        if q.get("target"): ph.append(str(q["target"]))
        P[int(d["id"])] = ph
    return P


def rule(D, P):
    out = {}
    for i, r in D.iterrows():
        if not str(r.answer_source).startswith("rul263"):
            continue
        g, c, a = float(r.g), float(r.cs), float(r.answer)
        if not (np.isfinite(g) and np.isfinite(c) and g > 0 and c > 0 and np.isfinite(a)):
            continue
        if abs(np.log(g / c)) >= np.log(1.1):
            continue
        lo, hi = min(g, c) * 0.97, max(g, c) / 0.97
        if lo <= a <= hi:
            continue
        ph = P.get(int(i), [])
        if any(PERSON.search(p) for p in ph):
            continue
        v = float(np.sqrt(g * c))
        out[str(int(i))] = {"value": v, "why": 't151'}
    return out


D = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv", index_col=0)
out = rule(D, load_parse("test"))
json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_test.json", "w"), indent=1)
print('')
