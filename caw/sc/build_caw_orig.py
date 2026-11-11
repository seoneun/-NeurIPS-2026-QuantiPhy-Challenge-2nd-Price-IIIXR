import json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
base = pd.read_csv(f"{R}/preds/test_caw9b.csv")
ans = {}
for ln in open(f"{R}/caw/sc/out/test_orig/g16.jsonl", encoding="utf-8"):
    x = json.loads(ln); p = x["parsed"][0]
    ans[int(x["original_id"])] = p
ids = base["Unnamed: 0"].astype(int)
new = ids.map(ans)
assert new.notna().sum() >= 0.99 * len(base)
out = base.copy(); out["parsed_value"] = new.values
out.to_csv(f"{R}/preds/test_caw9b_orig.csv", index=False)
old = pd.to_numeric(base.parsed_value, errors="coerce").abs()
with np.errstate(divide="ignore", invalid="ignore"):
    e = np.abs(np.log10(new.abs() / old))
print('')
