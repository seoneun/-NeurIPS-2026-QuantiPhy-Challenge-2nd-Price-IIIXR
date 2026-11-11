import sys, json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp'); sys.path.insert(0,R+'/analysis/v3')
import pandas as pd, numpy as np
from lab_rows import labelled
L=labelled('sub_xt_Nzz39')
pt=json.load(open(R+'/cache/tmp/ag_d11tilt_incline_patch_test.json')); pv=json.load(open(R+'/cache/tmp/ag_d11tilt_incline_patch_val.json'))
print('labelled', len(L), L.split.value_counts().to_dict())
hit=[i for i in L.index if (str(i) in pt and (L.split.loc[i]=='test')) or (str(i) in pv and L.split.loc[i]=='val')]
print('labelled changed:', hit)
d=pd.read_csv(R+'/submissions/sub_xt_Nzz39_detail.csv',index_col=0)
for k,v in pt.items():
    r=d.loc[int(k)]; print(k, r.video_id, r.cat, r.question[:90], r.answer, r.answer_source, r.g, r.cs, '->', round(v['value'],4))
# other patches touching these rows
import glob,re
sh=open('/dev/stdin').read() if False else ''
