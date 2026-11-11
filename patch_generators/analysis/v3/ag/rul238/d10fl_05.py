import sys, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp')
from lab_rows import labelled
mem=['simulation_00%s'%x for x in ('25','26','27','31','35','37','38')]
mem=mem+[m+'_segmented' for m in mem]
raw=pd.read_csv(f"{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv",index_col=0)
det=pd.read_csv(f"{R}/submissions/sub_xt_Nzz38_detail.csv",index_col=0)
PP={p['id']:p for p in map(json.loads,open(f"{R}/geo/out/parse_test_v2.jsonl"))}
L=labelled('sub_xt_Nzz38')
for i in raw.index:
    if raw.video_id[i] in mem:
        p=PP[i]
        print(i, raw.video_id[i], raw.fps[i], '|', raw.question[i][:60], '|', raw.ground_truth_prior[i][:50], '|', str(raw.depth_info[i])[:120], '| g %.3g ans %.3g %s'%(det.g[i], det.answer[i], det.answer_source[i]), 'LAB' if i in L.index else '')
# any val rows with these question texts
vr=pd.read_csv(f"{R}/preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv",index_col=0)
m=vr[vr.question.str.contains('person',case=False)&vr.video_id.str.startswith('simulation')&vr.ground_truth_prior.str.contains('1.3')]
print(m[['video_id','question','ground_truth_prior','ground_truth_posterior']].to_string())
