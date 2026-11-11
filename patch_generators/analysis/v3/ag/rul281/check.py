import json,sys
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")); D=R+'/analysis/v3/ag/rul281'
E=json.load(open(R+'/cache/tmp/ag_echo2_patch_test.json')); N=json.load(open(R+'/cache/tmp/ag_echo2r100_patch_test.json'))
O=json.load(open(D+'/prev_tri87_patch_test.json'))
print('identical_to_tri87_output', N==O, 'val', json.load(open(R+'/cache/tmp/ag_echo2r100_patch_val.json')))
print('removed', sorted(set(E)-set(N),key=int), 'kept', len(N), 'of', len(E), 'values unchanged', all(N[k]==E[k] for k in N))
P={}
for l in open(R+'/geo/out/parse_test_v2.jsonl'):
    j=json.loads(l); P[str(j['id'])]=j
import os
f=R+'/submissions/sub_xt_Nzz43_detail.csv'
if not os.path.exists(f): f=R+'/submissions/sub_xt_Nzz42_detail.csv'
d=pd.read_csv(f,index_col=0); print('detail',f.split('/')[-1])
for k in sorted(set(E)-set(N),key=int)+['1632']:
    i=int(k); p=P[k]
    print(k,p['video_id'],'|',p['question'][:90],'| rul280',round(E[k]['value'],4),'| cur',d.loc[i,'answer'] if i in d.index else None, d.loc[i,'answer_source'] if i in d.index else None)
