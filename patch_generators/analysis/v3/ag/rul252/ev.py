import sys, json, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")); sys.path.insert(0,R+'/cache/tmp'); sys.path.insert(0,R+'/caw')
from lab_rows import labelled; import analyze_caw as A
L=labelled('sub_xt_Nzz40')
for sp in ('test','val'):
    p=json.load(open(f'{R}/cache/tmp/ag_d18veh_wheelcls_patch_{sp}.json'))
    h=L[(L.split==sp)&L.index.astype(str).isin(p)]
    print(sp,'patch',len(p),'labelled hits',len(h))
    if len(h):
        h=h.copy(); h['new']=[p[str(i)]['value'] for i in h.index]; h['m2']=A.row_mra(h.new.values,h.y.values); print(h[['video_id','answer','new','y','m','m2']])
    Q=pd.read_csv(f'{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv',index_col=0).question.astype(str).str.lower()
    s=L[L.split==sp]; q=Q.reindex(s.index).fillna('')
    s=s[q.str.contains('diameter') & q.str.contains('wheel|tire|tyre')]
    print(s.assign(q=q.reindex(s.index).str[:60])[['video_id','cat','answer','y','m','q']].to_string())
