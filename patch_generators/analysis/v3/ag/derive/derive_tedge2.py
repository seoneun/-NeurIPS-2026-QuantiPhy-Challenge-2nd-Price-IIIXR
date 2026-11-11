"""rul432 (main session q609.sh, 10-05 04:47): tedge rows minus those whose sub_xt_Nze answer came from the scene ruler (sruler_geo)."""
import json, os
import pandas as pd
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
for s, det in (("test", "sub_xt_Nze_detail"), ("val", "sub_xt_Nze_valdetail")):
    D = pd.read_csv(f"submissions/{det}.csv").set_index("Unnamed: 0")
    t = json.load(open(f"cache/tmp/ag_tedge_patch_{s}.json"))
    o = {k: v for k, v in t.items() if str(D.answer_source.get(int(k), "")).replace("mo_guard:", "") != "sruler_geo"}
    json.dump(o, open(f"cache/tmp/ag_tedge2_patch_{s}.json", "w"), indent=0); print("rul432", s, len(t), "->", len(o))
