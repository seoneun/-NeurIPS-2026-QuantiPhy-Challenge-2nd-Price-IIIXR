import sys, json, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
d = json.load(open(f"{R}/cache/tmp/ag_rul056_patch_test.json"))
cat = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv").set_index("Unnamed: 0")["cat"].astype(str)
out = {k: v for k, v in d.items() if cat.get(int(k), "")[1:2] == "3"}
json.dump(out, open(f"{R}/cache/tmp/ag_rul057_patch_test.json", "w"), indent=1)
print("[test] rul057:", len(d), "->", len(out))
