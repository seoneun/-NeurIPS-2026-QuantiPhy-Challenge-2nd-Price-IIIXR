"""wing = the wingspan branch of the width rule (main session q575.sh, 10-04 23:04): keep ag_width rows whose why starts with 'wingspan:'."""
import json, os
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
for s in ["test", "val"]:
    d = json.load(open(f"cache/tmp/ag_width_patch_{s}.json"))
    w = {k: v for k, v in d.items() if "wingspan:" in str(v.get("why", ""))[:40]}
    json.dump(w, open(f"cache/tmp/ag_wing_patch_{s}.json", "w"), indent=0)
    print(s, len(d), "->", len(w))
