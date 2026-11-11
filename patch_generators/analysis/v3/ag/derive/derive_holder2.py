"""rul301 (main session q616.sh, 10-05 07:51): holder rows minus those whose new answer is CaW-routed (dis_caw_routed / caw_c1 / caw_nogeo)."""
import json, re, os
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
h = json.load(open("cache/tmp/ag_holder_patch_test.json"))
h2 = {k: v for k, v in h.items() if not re.search(r"\((dis_caw_routed|caw_c1|caw_nogeo)\b", v["why"])}
json.dump(h2, open("cache/tmp/ag_holder2_patch_test.json", "w"), indent=0); json.dump({}, open("cache/tmp/ag_holder2_patch_val.json", "w"))
print("rul301", len(h), "->", len(h2), "dropped", sorted(set(h) - set(h2), key=int))
