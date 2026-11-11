import sys, json, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp')
from lab_rows import labelled
L=labelled('sub_xt_Nzz40')
S=set('0035 0043 0058 0059 0060 0061 0063 0064 0066 0068 0070 0071 0074 0075 0079 0082 0083 0084 0085 0086 0087 0088 0089 0090 0091 0092 0093 0095 0096 0104 0106 0107 0108 0110 0124 0126 0127 0129 0263 0269 0272 0276 0277 0283 0288 0290 0293 0295 0296 0299 0300 0301 0307 0309 0310 0311 0098 0099 0100 0101 0102 0103 0076 0077 0078 0080 0081 0062 0065 0067'.split())
P={}
for sp in ('test','val'):
    for l in open(R+f'/geo/out/parse_{sp}_v2.jsonl'):
        j=json.loads(l); P[(sp,j['id'])]=j
for i,r in L[L.split=='test'].iterrows():
    j=P[('test',i)]; m=re.match(r'simulation_(\d+)',str(j['video_id']))
    if not m or m.group(1) not in S: continue
    q=j['q']
    if q.get('t') is None and q.get('t1') is None: continue
    print(i,j['video_id'],q['kind'],'|',j['question'][:90],'| prior',j['prior_text'][:70],'| y',r.y,'ans %.4g'%r.answer if 'answer' in r else '', 'm',round(r.m,3))
