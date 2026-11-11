"""rul050: c36ffdrag without size rows (both verifiers, cycle 36): the agree_wgm CaW hedge already lowers
the size answers; keep the hfloor / speed / displacement rows. Inputs: patch + parse kind."""
import json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for sp in ("test", "val"):
    P = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = ((j.get("q") or {}).get("kind"))
    d = json.load(open(f"{R}/cache/tmp/ag_c36ffdrag_patch_{sp}.json"))
    out = {k: v for k, v in d.items() if P.get(int(k)) != "size"}
    json.dump(out, open(f"{R}/cache/tmp/ag_c36ffdrag2_patch_{sp}.json", "w"), indent=1)
    print(sp, len(d), "->", len(out), sorted(out, key=int))
