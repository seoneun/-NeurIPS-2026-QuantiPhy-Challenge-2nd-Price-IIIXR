"""rul042: c26tsw without clip-start instantaneous speeds (verifier filter, cycle 26).
Geo's endpoint derivative at t <= 1/fps is biased low in this template, so the
ruler rescale would move those rows away from the truth. Inputs: patch + parse."""
import json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
P = {}
for l in open(f"{R}/geo/out/parse_test_v2.jsonl"):
    j = json.loads(l); P[int(j["id"])] = j
for sp in ("test", "val"):
    d = json.load(open(f"{R}/cache/tmp/ag_c26tsw_patch_{sp}.json")); out = {}
    for k, v in d.items():
        j = P.get(int(k), {}) if sp == "test" else {}
        q = j.get("q", {}) or {}; fps = float(j.get("fps") or 24)
        if q.get("kind") == "speed" and ((q.get("t") is not None and float(q["t"]) <= 1.0 / fps + 1e-9) or q.get("when") == "start"):
            print("drop", sp, k, q.get("t")); continue
        out[k] = v
    json.dump(out, open(f"{R}/cache/tmp/ag_c26tsw2_patch_{sp}.json", "w"), indent=1); print(sp, sorted(out))
