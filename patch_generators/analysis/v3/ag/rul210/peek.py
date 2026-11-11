import pandas as pd, json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
raw=pd.read_csv(R+'/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
print(list(raw.columns))
r=raw.loc[2508]; print(r.to_dict())
D=pd.read_csv(R+'/submissions/sub_xt_Nzz34_detail.csv',index_col=0)
print(D.columns.tolist()); print(D.loc[[2508,2512,2516]])
for l in open(R+'/geo/out/parse_test_v2.jsonl'):
    j=json.loads(l)
    if j['id']==2508: print(j)
