"""rul280 final filter (main session q585.sh, 10-05 02:55, overwrote filter_c4's rul280 test file): drop echo rows whose target and
prior phrases are word-subsets of each other, for disp/path/dist2 questions under a time-less speed prior. val = echo val."""
import json, re, os
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
P = {}
for ln in open("geo/out/parse_test_v2.jsonl"):
    x = json.loads(ln); P[str(x["id"])] = x
d = json.load(open("cache/tmp/ag_echo_patch_test.json"))
drop = []
for k in d:
    x = P.get(k, {}); q = x.get("q") or {}; pr = (x.get("priors") or [{}])[0]
    tw = set(re.findall(r"[a-z]+", ((x.get("target") or {}).get("phrase") or "").lower())); pw = set(re.findall(r"[a-z]+", ((x.get("prior_obj") or {}).get("phrase") or "").lower()))
    if q.get("kind") in ("disp", "path", "dist2") and tw and pw and (tw <= pw or pw <= tw) and pr.get("kind") == "speed" and pr.get("t") is None and pr.get("t1") is None:
        drop.append(k)
o = {k: v for k, v in d.items() if k not in drop}
json.dump(o, open("cache/tmp/ag_echo2_patch_test.json", "w"), indent=0)
json.dump(json.load(open("cache/tmp/ag_echo_patch_val.json")), open("cache/tmp/ag_echo2_patch_val.json", "w"))
print("rul280", len(d), "->", len(o), "dropped", drop)
