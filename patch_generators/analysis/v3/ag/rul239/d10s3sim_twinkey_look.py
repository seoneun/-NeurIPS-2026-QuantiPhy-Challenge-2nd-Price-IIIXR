import json,sys,pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
P=[json.loads(l) for l in open(R+"/geo/out/parse_test_v2.jsonl")]
raw=pd.read_csv(R+"/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
det=pd.read_csv(R+"/submissions/sub_xt_Nzz38_detail.csv").set_index("Unnamed: 0")
for p in P:
    if p["video_id"] in ("simulation_0025","simulation_0026","simulation_0026_segmented","simulation_0025_segmented"):
        i=p["id"]; d=det.loc[i]
        print(i,p["video_id"],d["cat"],repr(p["question"]),"| prior:",p["prior_text"],"| tdk",p.get("target_dkeys"),"pdk",p.get("prior_dkeys"))
        print("   ans",d.answer,d.answer_source,"g",d.g,"cs",d.cs,d["mode"], "| why", str(raw.loc[i].meas_why)[:200])
print(raw.loc[[p["id"] for p in P if p["video_id"]=="simulation_0026"][0]].depth_info)
print(raw.loc[[p["id"] for p in P if p["video_id"]=="simulation_0025"][0]].depth_info)
