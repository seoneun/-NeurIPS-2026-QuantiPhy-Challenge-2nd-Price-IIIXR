"""rul057: rul056 restricted to 3D categories (S3/D3), per its verifier (cycle 38).
In planar videos the rotated-rect tilt of swimming/flying silhouettes is fin geometry, not a lying object's tilt,
so the planar rows are dropped. Inputs: the rul056 patch and the row category from the build detail."""
import json, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for sp, det in (("test", "sub_xt_Nzw_detail.csv"), ("val", "sub_xt_Nzw_valdetail.csv")):
    d = json.load(open(f"{R}/cache/tmp/ag_c38hlie_patch_{sp}.json"))
    cat = pd.read_csv(f"{R}/submissions/{det}").set_index("Unnamed: 0")["cat"].astype(str)
    out = {k: v for k, v in d.items() if cat.get(int(k), "")[1:2] == "3"}
    json.dump(out, open(f"{R}/cache/tmp/ag_c38hlie3d_patch_{sp}.json", "w"), indent=1)
    print(sp, len(d), "->", len(out), sorted(out, key=int))
