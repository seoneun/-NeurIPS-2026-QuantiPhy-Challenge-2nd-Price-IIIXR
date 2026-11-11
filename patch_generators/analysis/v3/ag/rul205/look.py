import pandas as pd, json, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',70)
D=pd.read_csv(R+'/submissions/sub_xt_Nzz29_detail.csv').rename(columns={'Unnamed: 0':'rid'})
print(D.columns.tolist())
G=pd.read_csv(R+'/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').rename(columns={'Unnamed: 0':'rid'})
P={}
for l in open(R+'/geo/out/parse_test_v2.jsonl'):
    j=json.loads(l); P[j['id']]=j
v='internet_0041'
print(D[D.video_id==v][['rid','question','cat','answer','answer_source','g','cs','mode']].to_string())
g=G[G.video_id==v]
print(g[['rid','ground_truth_prior','meas_why','meas_mode','meas_mpp','geo_value','meas_tracks']].to_string())
for r in g.rid: print(r, json.dumps({k:P[r].get(k) for k in ('q','target','prior_obj')}))
for d in ['tracks_test','tracks_test_v2sc','tracks_test_v3nt','tracks_test_v3','tracks_test_v3nt32']:
    try: T=json.load(open(f'{R}/geo/out/{d}/{v}.json'))
    except Exception as e: print(d,'missing'); continue
    print('==',d,T.get('W'),T.get('H'),list(T.keys()))
    for key in ('queries','queries_d'):
        for q in T.get(key) or []:
            if not isinstance(q,dict): continue
            ob=q.get('obj'); ex=''
            if ob is not None and 'objs' in T:
                b=np.array([s['st']['box'] for s in T['objs'].get(str(ob)) or [] if s and s.get('st')]).reshape(-1,4)
                if len(b):
                    c=(b[:,:2]+b[:,2:])/2
                    ex=f"n={len(b)} w={np.median(b[:,2]-b[:,0]):.0f} h={np.median(b[:,3]-b[:,1]):.0f} span={np.ptp(c,0).round()}"
            print(' ',key,q['key'],'obj',ob,ex, 'det' if q.get('det_boxes') else '')
