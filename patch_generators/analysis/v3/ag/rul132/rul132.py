"""rul132: lab speed at the START of the target's own timed depth-key chord.
If the answer is below 0.8 x the focal-free radial chord speed |d(t2)-d(t)|/(t2-t), raise it:
value = own geo if chord <= g <= 1.5 chord, else the chord. Lab labels sit above ideal physics,
so a radial lower estimate is safe in both worlds. No ids, no labels; CSV fps not needed."""
import sys, json, re
import pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB=sys.argv[1] if len(sys.argv)>1 else 'sub_xt_Nzz20'
UNIT={'m/s':1.0,'cm/s':100.0,'mm/s':1000.0,'km/h':3.6}
def toks(s): return set(re.findall(r'[a-z]+',str(s).lower().replace('_',' ')))
def run(split):
    D=pd.read_csv(R+'/submissions/%s_%s.csv'%(SUB,'detail' if split=='test' else 'valdetail'))
    P={}
    for l in open(R+'/geo/out/parse_%s_v2.jsonl'%split):
        j=json.loads(l); P[j['id']]=j
    out={}; log=[]
    for i,r in D.iterrows():
        j=P.get(i)
        if j is None or j.get('video_source')!='lab': continue
        q=j['q']
        if q.get('kind')!='speed' or q.get('t') is None or q.get('t1') is not None: continue
        u=UNIT.get(q.get('unit'))
        if u is None: continue
        keys=j.get('target_dkeys') or []
        if len(keys)!=1 or not toks(keys[0]) or not toks(keys[0])<=toks(q.get('target')): continue
        pts=sorted([(d['t'],d['d']) for d in j['depth'] if d['key']==keys[0] and d['t'] is not None])
        t=q['t']; k=[n for n,(tk,_) in enumerate(pts) if abs(tk-t)<0.02]
        if not k or k[0]+1>=len(pts): continue
        (t1,d1),(t2,d2)=pts[k[0]],pts[k[0]+1]
        if not (0<t2-t1<=0.5): continue
        c=abs(d2-d1)/(t2-t1)
        a=r.answer/u; g=r.g/u if r.g==r.g else np.nan
        if str(r.answer_source).startswith('patch') and not a<0.5*c: pass
        if a<0.8*c:
            v=g if (g==g and c<=g<=1.5*c) else c
            out[str(i)]={'value':float(v*u),'why':'rul132: answer %.3f < 0.8 x own depth-key chord %.3f m/s over [%.2f,%.2f]'%(a,c,t1,t2)}
            log.append((i,r.video_id,j['question'][:60],round(a,3),r.answer_source,round(g,3) if g==g else None,round(c,3),round(v,3)))
    json.dump(out,open(R+'/cache/tmp/ag_c72dkchord_patch_%s.json'%split,'w'),indent=1)
    print(split,len(out)); [print(' ',x) for x in log]
    return D,out
run('test'); Dv,ov=run('val')
sys.path.insert(0,R+'/cache/tmp'); sys.path.insert(0,R+'/analysis/v3')
from lab_rows import labelled
L=labelled(SUB); pt=json.load(open(R+'/cache/tmp/ag_c72dkchord_patch_test.json'))
hit=[(sp,i) for i,sp in zip(L.index,L.split) if (sp=='test' and str(i) in pt) or (sp=='val' and str(i) in ov)]
print('labelled rows changed:',hit)
