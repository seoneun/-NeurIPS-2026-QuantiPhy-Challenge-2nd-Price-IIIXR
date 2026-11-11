import json, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
split = sys.argv[1]
BASE = sys.argv[3] if len(sys.argv) > 3 else "caw9b"
base = pd.read_csv(f"{R}/preds/{split}_{BASE}.csv")
idc = base.columns[0]
two = set()
for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
    x = json.loads(ln)
    if x.get("target2") or x["q"].get("kind") == "dist2":
        two.add(int(x["id"]))
ans = {}
for ln in open(f"{R}/caw/sc/out/{split}_crop/g16.jsonl", encoding="utf-8"):
    x = json.loads(ln)
    p = [abs(q) for q in x["parsed"] if q is not None and np.isfinite(q) and q != 0]
    if p and int(x["original_id"]) not in two:
        ans[int(x["original_id"])] = p[0]
import os
p2 = f"{R}/caw/sc/out/{split}_crop_two/g16.jsonl"
if os.path.exists(p2) and (len(sys.argv) < 3 or sys.argv[2] != "single"):
    n2 = 0
    for ln in open(p2, encoding="utf-8"):
        x = json.loads(ln)
        p = [abs(q) for q in x["parsed"] if q is not None and np.isfinite(q) and q != 0]
        if p and int(x["original_id"]) in two:
            ans[int(x["original_id"])] = p[0]; n2 += 1
    print('')
ids = base[idc].astype(int)
new = ids.map(ans)
out = base.copy()
out["parsed_value"] = np.where(new.notna(), new, pd.to_numeric(base.parsed_value, errors="coerce"))
suffix = "_crop" if (len(sys.argv) > 2 and sys.argv[2] == "single") or not os.path.exists(p2) else "_crop2"
out.to_csv(f"{R}/preds/{split}_{BASE}{suffix}.csv", index=False)
chg = new.notna() & ~np.isclose(new.fillna(-1), pd.to_numeric(base.parsed_value, errors="coerce").abs().fillna(-2), rtol=1e-9)
print('')
