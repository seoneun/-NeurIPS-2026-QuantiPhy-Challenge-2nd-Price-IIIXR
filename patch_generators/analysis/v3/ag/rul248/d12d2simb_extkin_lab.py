import sys, json, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp')
from lab_rows import labelled
L=labelled('sub_xt_Nzz40')
print(L.shape, L.split.value_counts().to_dict())
P={}
for sp in ('test','val'):
    for l in open(R+f'/geo/out/parse_{sp}_v2.jsonl'):
        j=json.loads(l); P[(sp,j['id'])]=j
for i,r in L.iterrows():
    j=P.get((r.split,i))
    if j is None or not str(j['video_id']).startswith('simulation'): continue
    q=j['q']; pr=j['priors']
    if q.get('t') is None or len(pr)!=1 or pr[0]['kind']!='speed' or pr[0].get('gravity') or pr[0].get('t') is not None: continue
    v=pr[0]['value_si']; t=q['t']; U={'m':1,'cm':100,'mm':1000,'m/s':1,'cm/s':100,'km/h':3.6}.get(q['unit'],np.nan)
    print(r.split,i,j['video_id'],q['kind'],'|',j['question'][:80],'| prior',j['prior_text'][:60],'| y',r.y,'v*t',round(v*t*U,4) if q['kind']!='speed' else '', 'ans',r.get('answer',''), 'm',round(r.m,3))
