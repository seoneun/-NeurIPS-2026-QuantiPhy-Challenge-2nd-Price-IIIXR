"""rul436 (main session q616.sh, 10-05 07:51): tlight rows whose CaW crop and full-frame answers agree within 0.1 decade."""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
crop = pd.to_numeric(pd.read_csv("preds/test_caw9b_orig_crop.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs()
full = pd.to_numeric(pd.read_csv("preds/test_caw9b_orig.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs()
t = json.load(open("cache/tmp/ag_tlight_patch_test.json"))
t2 = {k: v for k, v in t.items() if abs(np.log10(crop[int(k)] / full[int(k)])) <= 0.1}
json.dump(t2, open("cache/tmp/ag_tlight2_patch_test.json", "w"), indent=0); json.dump({}, open("cache/tmp/ag_tlight2_patch_val.json", "w"))
print("rul436", len(t), "->", len(t2), sorted(t2, key=int))
