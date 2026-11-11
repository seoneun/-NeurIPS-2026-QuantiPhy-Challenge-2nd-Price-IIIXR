"""rul028: c23s1kchord, far/near slope branch only (verifier filter, cycle 23).
The left/right branch (a table cut at the border) is dropped: it is not a slope chord."""
import json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for sp in ("test", "val"):
    d = json.load(open(f"{R}/cache/tmp/ag_c23s1kchord_patch_{sp}.json"))
    out = {k: v for k, v in d.items() if v["why"].startswith("c23s1kchord: far/")}
    json.dump(out, open(f"{R}/cache/tmp/ag_c23s1kchord2_patch_{sp}.json", "w"), indent=1)
    print(sp, sorted(out))
