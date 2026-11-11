import pandas as pd, numpy as np, json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
pd.set_option("display.width",320); pd.set_option("display.max_colwidth",120)
d=pd.read_csv(f"{R}/submissions/sub_xt_Nzz24_detail.csv",index_col=0)
raw=pd.read_csv(f"{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv",index_col=0)
print(d.columns.tolist()); print(raw.columns.tolist())
P={}
for l in open(f"{R}/geo/out/parse_test_v2.jsonl"):
    j=json.loads(l); P[j.get('id',j.get('row_id'))]=j
k=list(P)[0]; print('pkeys',list(P[k].keys()), type(k))
for i in [2389,2390,2396,2402,2194]:
    j=P.get(i,P.get(str(i)))
    print(i, json.dumps({kk:j[kk] for kk in ['q','depth','target_dkeys']}))
    print(d.loc[i,['video_id','question','answer','answer_source','g','cs','mode','kind']].to_dict())
    print(raw.loc[i,['depth_info','meas_why','meas_mode','geo_value']].to_dict())
