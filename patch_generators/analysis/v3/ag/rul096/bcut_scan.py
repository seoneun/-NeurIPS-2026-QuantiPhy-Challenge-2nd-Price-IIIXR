import json, os, sys, numpy as np, pandas as pd, cv2
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp'); sys.path.insert(0,R+'/caw'); sys.path.insert(0,R+'/analysis/v3')
import analyze_caw as A
from lab_rows import labelled
SUB=os.environ.get('SUB','sub_xt_Nzz8')
L=labelled(SUB)
WH={}
def wh(path):
    if path in WH: return WH[path]
    c=cv2.VideoCapture(path); w=c.get(cv2.CAP_PROP_FRAME_WIDTH); h=c.get(cv2.CAP_PROP_FRAME_HEIGHT); c.release()
    if h<=w: H=480; W=int(round(w*480/h))
    else: W=480; H=int(round(h*480/w))
    WH[path]=(W,H); return W,H
def tboxes(split, vid, mt, phrase):
    d={'v1':'', 'v2':'_v2sc'}.get(mt, None)
    if d is None: return None
    p=f'{R}/geo/out/tracks_{split}{d}/{vid}.json'
    if not os.path.exists(p): return None
    j=json.load(open(p))
    if 'objs' in j:
        qm=j['queries'] if isinstance(j['queries'],dict) else {q['key']:q.get('obj') for q in j['queries']}
        for q in (j.get('queries_d') or (j['queries'] if isinstance(j['queries'],list) else [])):
            if 'target' in q.get('roles',[]) and q.get('phrase')==phrase:
                o=qm.get(q['key'])
                if o is None: return None
                return [s['st']['box'] for s in j['objs'][str(o)] if s and s.get('st')]
        return None
    for q in j.get('queries',[]):
        if 'target' in q.get('roles',[]) and q.get('phrase')==phrase:
            return [f['st']['box'] for f in q.get('frames',[]) if f and f.get('st')]
    return None
rows=[]
for split in ['test','val']:
    raw=pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    det=pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split=="test" else "valdetail"}.csv').set_index('Unnamed: 0')
    P={}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j=json.loads(l); P[j['id']]=j
    for i,r in det.iterrows():
        pj=P.get(i)
        if not pj or pj['q'].get('kind')!='size' or not np.isfinite(r['g']): continue
        if i not in raw.index: continue
        rr=raw.loc[i]
        mpp=rr['meas_mpp']; gv=rr['geo_value']
        if not (np.isfinite(mpp) and np.isfinite(gv) and mpp>0): continue
        pj=P.get(i)
        if not pj: continue
        ph=(pj.get('target') or {}).get('phrase')
        B=tboxes(split, rr['video_id'], rr['meas_tracks'], ph)
        if not B: continue
        W,H=wh(R+'/'+rr['video_path'] if not rr['video_path'].startswith('/') else rr['video_path']) if isinstance(rr['video_path'],str) else (854,480)
        B=np.array(B,float); w=np.median(B[:,2]-B[:,0]); h=np.median(B[:,3]-B[:,1])
        px=gv/mpp
        ax='x' if abs(px-w)<=abs(px-h) else 'y'
        if ax=='x': cut=((B[:,0]<=2)|(B[:,2]>=W-3)).mean()
        else: cut=((B[:,1]<=2)|(B[:,3]>=H-3)).mean()
        rows.append(dict(split=split,id=i,vid=r.get('video_id',rr['video_id']),cat=r.get('cat'),q=str(rr['question'])[:70],ans=r['answer'],src=r['answer_source'],g=r['g'],cs=r['cs'],mode=r.get('mode'),px=round(px,1),w=round(w,1),h=round(h,1),ax=ax,cut=round(cut,2),W=W,H=H))
D=pd.DataFrame(rows)
D.to_csv(R+'/cache/tmp/ag_c54cutlb_bcut.csv',index=False)
C=D[D.cut>=0.8]
print('val det index sample', det.index[:3].tolist() if False else '', 'size rows with geo:',len(D),' cut>=0.8:',len(C), C.groupby(['split','cat']).size().to_dict())
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',70); pd.set_option('display.max_rows',300)
Lk=L.copy(); Lk['key']=list(zip(Lk.split,Lk.index))
C=C.copy(); C['key']=list(zip(C.split,C.id))
C=C.merge(Lk[['key','y','m','mg','mc']],on='key',how='left')
C['alt']=np.where(C.cs>C.g, np.maximum(C.cs,C.ans), C.ans)
C['m_alt']=np.where(C.y.notna(), A.row_mra(C.alt.values, C.y.fillna(1).values), np.nan)
print(C.drop(columns=['key']).to_string())
lab=C[C.y.notna()]
print('labelled cut rows',len(lab),'m',lab.m.mean().round(3) if len(lab) else None,'m_alt',lab.m_alt.mean().round(3) if len(lab) else None)
