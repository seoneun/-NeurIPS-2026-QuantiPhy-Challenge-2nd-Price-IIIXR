"""rul284 (main session q624.sh, 10-05 09:15): evtparse main patch minus its FF part."""
import json, os
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
e = json.load(open("cache/tmp/ag_evtparse_patch_test.json")); ff = json.load(open("cache/tmp/ag_evtparse_ff_patch_test.json"))
e2 = {k: v for k, v in e.items() if k not in ff}
json.dump(e2, open("cache/tmp/ag_evtparse2_patch_test.json", "w"), indent=0); json.dump({}, open("cache/tmp/ag_evtparse2_patch_val.json", "w"))
print("rul284", len(e), "->", len(e2), sorted(e2, key=int))
