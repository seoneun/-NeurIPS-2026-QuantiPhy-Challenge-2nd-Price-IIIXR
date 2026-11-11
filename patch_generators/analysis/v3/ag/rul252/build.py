"""rul252: physical class-range veto for vehicle wheel/tire DIAMETER rows (simulation videos only).
Owner class from question text: bicycle/bike [0.50,0.76] m; truck [0.70,1.25] m; car/taxi/sedan/suv/van [0.55,0.80] m.
Answer inside range -> unchanged. Else edge=nearer bound; |a-edge|/edge<=0.10 -> edge (near-miss clamp);
elif CaW cs inside range -> cs; else edge. Skips motor/robot/cart/aircraft/landing/toy/forklift/train.
usage: python build.py [SUB] [write]"""
import sys, re, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB=sys.argv[1] if len(sys.argv)>1 else 'sub_xt_Nzz40'; WRITE='write' in sys.argv
QRE=re.compile(r'diameter of (the )?.*?(wheel|tire|tyre)s?')
SKIP=re.compile(r'motor|robot|cart|aircraft|landing|toy|forklift|train')
CLS=[(re.compile(r'\b(bicycle|bike)s?\b'),'bicycle',0.50,0.76),(re.compile(r'\btrucks?\b'),'truck',0.70,1.25),
     (re.compile(r'\b(car|taxi|sedan|suv|van)s?\b'),'car',0.55,0.80)]
UF={'m':1.0,'cm':0.01,'mm':0.001,'meters':1.0,'meter':1.0,'centimeters':0.01}
def unit_of(q):
    if re.search(r'\bin (centimet(er|re)s|cm)\b',q): return 0.01
    if re.search(r'\bin (millimet(er|re)s|mm)\b',q): return 0.001
    if re.search(r'\bin (met(er|re)s|m)\b',q): return 1.0
    return None
for split in ('test','val'):
    D=pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split=='test' else 'valdetail'}.csv",index_col=0)
    if "question" not in D.columns: D["question"]=pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv",index_col=0).question.reindex(D.index)
    out={}
    for i in D.index:
        vid=str(D.video_id[i])
        if not vid.startswith('simulation'): continue
        q=str(D.question[i]).lower()
        if not QRE.search(q) or SKIP.search(q): continue
        c=[x for x in CLS if x[0].search(q)]
        if not c: print('noclass',split,i,vid,q); continue
        _,nm,lo,hi=c[0]
        uf=unit_of(q)
        if uf is None: print('nounit',split,i,q); continue
        a=float(D.answer[i])*uf; cs=D.cs[i]; cs=float(cs)*uf if pd.notna(cs) else np.nan
        tag='in' if lo<=a<=hi else 'OUT'
        print(split,i,vid,D.cat[i],nm,'a=%.3f cs=%.3f g=%s'%(a,cs,D.g[i]),D.answer_source[i],tag,'|',q[:90])
        if lo<=a<=hi: continue
        e=lo if a<lo else hi
        if abs(a-e)/e<=0.10: v,w=e,'near-miss clamp'
        elif lo<=cs<=hi: v,w=cs,'CaW in range'
        else: v,w=e,'edge'
        out[str(int(i))]={'value':v/uf,'why':f'rul252: {nm} wheel diameter {a:.3g} m outside [{lo},{hi}] -> {v:.3g} m ({w})'}
    print(split,'changed',len(out),out)
    if WRITE: json.dump(out,open(f'{R}/cache/tmp/ag_d18veh_wheelcls_patch_{split}.json','w'),indent=1)
