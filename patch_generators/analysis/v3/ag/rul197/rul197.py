"""rul197: c94d3brkt + verifier filter. Rows whose answer comes from a d3size_* source where geo (g) and CaW (cs)
agree within 10% but the rul263 multipliers pushed the answer outside [min(g,cs)*0.97, max(g,cs)/0.97] -> geometric mean.
FILTER: exclude rows whose parsed target phrase matches \b(pedestrian|person|human|man|woman|people|rider)\b, because
CaW returns a stock 1.7 m person default there, so agreement with it is not independent evidence.
Code condition on inputs only (parse + g/cs/answer); labels only evaluate."""
import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/cache/tmp")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz29"
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


def rule(D, P, excl):
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
            excl.append((int(i), ph)); continue
        v = float(np.sqrt(g * c))
        out[str(int(i))] = {"value": v, "why": f"{TAG}: rul263 moved answer outside agreeing geo/CaW bracket, non-person target ({r.answer_source} {a:.4g}; g {g:.4g} cs {c:.4g})"}
    return out


import analyze_caw as A
from lab_rows import labelled
L = labelled(SUB)
for sp in ("val", "test"):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if sp == 'test' else 'valdetail'}.csv", index_col=0)
    P = load_parse(sp); excl = []
    out = rule(D, P, excl)
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
    print(f"[{sp}] kept {len(out)} rows", D.loc[[int(k) for k in out], "cat"].value_counts().to_dict() if out else {})
    print(f"  excluded (person target) {len(excl)}:", [(i, ph[0] if ph else '') for i, ph in excl])
    Ls = L[L.split == sp]
    sb = sa = 0
    for k, x in out.items():
        i = int(k); y = Ls.loc[i, "y"] if i in Ls.index else np.nan
        mb = ma = np.nan
        if np.isfinite(y):
            mb = A.row_mra(np.array([D.loc[i, "answer"]]), np.array([y]))[0]; ma = A.row_mra(np.array([x["value"]]), np.array([y]))[0]
            sb += mb; sa += ma
        print(f"  {i} {D.loc[i,'video_id']} {D.loc[i,'cat']} | {(P.get(i) or [''])[0]} | {str(D.loc[i,'question'] if 'question' in D.columns else '')[:55]} | {D.loc[i,'answer']:.4g} -> {x['value']:.4g} | y={y} {mb}->{ma}")
    print(f"  labelled sum {sb:.2f} -> {sa:.2f}")
