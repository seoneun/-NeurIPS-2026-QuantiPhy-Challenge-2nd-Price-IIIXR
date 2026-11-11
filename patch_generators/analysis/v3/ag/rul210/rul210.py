"""rul210: 3D depth-key chord lower bound for lab (captured_*) speed-at-t rows in S3/D3, mode 3d_calib.
The target's own timed depth keys t_a <= t < t_b (t_b-t_a <= 0.5 s, t-t_a <= 0.25 s) plus the target track centre at
both key times give 3D points p = d * normalize([(cx-W/2)/f, (cy-H/2)/f, 1]) (f = meas_f * W). v3 = |p_b-p_a|/(t_b-t_a).
Gates: v3 / radial chord in [1.0, 1.6]; answer < 0.95 v3. Value = pipeline fusion of geo/CaW clipped to [v3, 1.3 v3].
A chord is a lower bound of the mean speed over the interval; lab labels sit above ideal physics.
No ids, no labels used to set values."""
import sys, json, re, os
import pandas as pd, numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB=sys.argv[1] if len(sys.argv)>1 else 'sub_xt_Nzz34'
RAWT='geo_v4ntrj5sQ2srngsr_d_vw'
TAG='rul210'
UNIT={'m/s':1.0,'cm/s':100.0,'mm/s':1000.0,'km/h':3.6}
def toks(s): return set(re.findall(r'[a-z]+',str(s).lower().replace('_',' ')))
def load_tj(split,v,mt):
    subs=['_v2sc','_v2'] if mt=='v2' else ['']
    for s in subs:
        p=R+'/geo/out/tracks_%s%s/%s.json'%(split,s,v)
        if os.path.exists(p): return json.load(open(p)),s
    return None,None
def target_frames(tj,phrase,ms):
    if tj.get('version')=='v2':
        qs=tj['queries_d'] if ms=='d' else tj['queries']
        q=[x for x in qs if x.get('phrase')==phrase and not x.get('where') and x.get('obj') is not None]
        if not q: return None
        objs=set(x['obj'] for x in q)
        if len(objs)!=1: return None
        return tj['objs'][str(q[0]['obj'])]
    q=[x for x in tj['queries'] if x.get('phrase')==phrase and not x.get('where')]
    if len(q)!=1: return None
    return q[0]['frames']
def centre_at(fr,fidx,fps,t):
    fi=np.array(fidx,float); k0=int(np.argmin(np.abs(fi-t*fps)))
    if abs(fi[k0]-t*fps)>1.5: return None
    cs=[]
    for k in range(len(fi)):
        if abs(fi[k]-fi[k0])<=1.01:
            x=fr[k]; s=x.get('st') if isinstance(x,dict) else None
            if s and s.get('c') is not None and s.get('border',0)==0: cs.append(s['c'])
    if not cs: return None
    return np.median(np.array(cs,float),0)
def fuse(g,c):
    if g==g and g>0 and c==c and c>0:
        return np.sqrt(g*c) if max(g,c)/min(g,c)<=2 else g
    return g if (g==g and g>0) else np.nan
def run(split):
    D=pd.read_csv(R+'/submissions/%s_%s.csv'%(SUB,'detail' if split=='test' else 'valdetail'),index_col=0)
    raw=pd.read_csv(R+'/preds/%s_%s_raw.csv'%(split,RAWT),index_col=0)
    P={}
    for l in open(R+'/geo/out/parse_%s_v2.jsonl'%split):
        j=json.loads(l); P[j['id']]=j
    out={}; log=[]; TJ={}
    for i,r in D.iterrows():
        j=P.get(i)
        if j is None or j.get('video_source')!='lab' or not str(r.video_id).startswith('captured'): continue
        if r['cat'] not in ('S3','D3'): continue
        q=j['q']
        if q.get('kind')!='speed' or q.get('t') is None or q.get('t1') is not None: continue
        rw=raw.loc[i]
        if not str(rw.meas_mode).startswith('3d_calib'): continue
        u=UNIT.get(q.get('unit'))
        if u is None: continue
        keys=j.get('target_dkeys') or []
        tt=toks((j.get('target') or {}).get('phrase') or q.get('target'))
        if len(keys)!=1 or not toks(keys[0]) or not toks(keys[0])<=tt: continue
        pts=sorted([(d['t'],d['d']) for d in j['depth'] if d['key']==keys[0] and d['t'] is not None])
        t=q['t']
        ia=[n for n,(tk,_) in enumerate(pts) if tk<=t+1e-6]
        if not ia: continue
        a_=ia[-1]
        if a_+1>=len(pts): continue
        (ta,da),(tb,db)=pts[a_],pts[a_+1]
        if not (t<tb-1e-6 and 0<tb-ta<=0.5 and t-ta<=0.25): continue
        fW=float(rw.meas_f)
        if not (fW==fW and fW>0): continue
        key=(r.video_id,rw.meas_tracks)
        if key not in TJ: TJ[key]=load_tj(split,r.video_id,rw.meas_tracks)
        tj,sd=TJ[key]
        if tj is None: continue
        fr=target_frames(tj,(j.get('target') or {}).get('phrase'),rw.merge_src if rw.merge_src=='d' else 'e')
        if fr is None: continue
        fps=float(j['fps']); W,H=tj['W'],tj['H']; f=fW*W
        ca,cb=centre_at(fr,tj['fidx'],fps,ta),centre_at(fr,tj['fidx'],fps,tb)
        if ca is None or cb is None: continue
        def P3(c,d):
            v=np.array([(c[0]-W/2)/f,(c[1]-H/2)/f,1.0]); return d*v/np.linalg.norm(v)
        v3=np.linalg.norm(P3(cb,db)-P3(ca,da))/(tb-ta); vr=abs(db-da)/(tb-ta)
        ratio=v3/vr if vr>0 else np.inf
        a=r.answer/u; g=r.g/u if r.g==r.g else np.nan; c=r.cs/u if r.cs==r.cs else np.nan
        fz=fuse(g,c)
        st='ok' if (1.0<=ratio<=1.6 and a<0.95*v3) else 'gate'
        val=np.nan
        if st=='ok':
            val=v3 if not (fz==fz) or fz<v3 else min(fz,1.3*v3)
            out[str(i)]={'value':float(val*u),'why':'3D depth-key chord lower bound: answer %.3f < 0.95 x v3 %.3f m/s over [%.2f,%.2f] (radial %.3f)'%(a,v3,ta,tb,vr)}
        log.append(dict(id=i,vid=r.video_id,cat=r['cat'],t=t,ta=ta,tb=tb,a=round(a,3),src=r.answer_source,g=round(g,3),c=round(c,3),vr=round(vr,3),v3=round(v3,3),ratio=round(ratio,2),st=st,new=round(val,3),q=j['question'][:50]))
    json.dump(out,open(R+'/cache/tmp/ag_%s_patch_%s.json'%(TAG,split),'w'),indent=1)
    Lg=pd.DataFrame(log)
    print(split,'changed',len(out)); pd.set_option('display.width',300); pd.set_option('display.max_rows',500); pd.set_option('display.max_colwidth',50)
    if len(Lg): print(Lg.to_string())
    return out,Lg
pt,Lt=run('test'); pv,Lv=run('val')
sys.path.insert(0,R+'/cache/tmp'); sys.path.insert(0,R+'/analysis/v3'); sys.path.insert(0,R+'/caw')
from lab_rows import labelled
import analyze_caw as A
L=labelled(SUB)
print('labelled cols',list(L.columns)[:20])
rows=[]
for i,r in L.iterrows():
    p=pt if r.split=='test' else pv
    lg=Lt if r.split=='test' else Lv
    if len(lg) and i in set(lg.id):
        x=lg[lg.id==i].iloc[0]
        new=p[str(i)]['value'] if str(i) in p else np.nan
        rows.append((r.split,i,r.video_id,r['cat'],x.st,x.ratio,r.y,x.a,x.v3,new))
E=pd.DataFrame(rows,columns=['split','id','vid','cat','st','ratio','y','a_si','v3_si','new_u'])
print(E.to_string())
ch=[(s,i) for s,i in zip(L.split,L.index) if str(i) in (pt if s=='test' else pv)]
if ch:
    LL=L.loc[[i for _,i in ch]]
    newv=np.array([(pt if s=='test' else pv)[str(i)]['value'] for s,i in ch])
    print('changed labelled',len(ch)); print(pd.DataFrame({'split':LL.split,'vid':LL.video_id,'cat':LL['cat'],'y':LL.y,'old':LL['answer'] if 'answer' in LL else np.nan,'m_old':LL.m,'new':newv,'m_new':A.row_mra(newv,LL.y.values)}).to_string())
