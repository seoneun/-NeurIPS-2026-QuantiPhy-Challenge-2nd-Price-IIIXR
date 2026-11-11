# rul055: D3 speed rows answered by geo alone (dis_geo / agree_geo_kind, unpatched, g>0, cs>0) that are asked
# >= 0.75 s after the target's LAST own depth key. geo then evaluates the depth by linearly extrapolating the keys.
# Gate: the target's own box scale says the extrapolated depth is off by >= 1.4x in either direction:
#   rd = Zlin(t) / (Z(t_last) * sqrt(area(t_last)/area(t))),  rd >= 1.4 or rd <= 1/1.4.
# Action: geo's depth (hence its metric scale) at t is unverified -> answer = gm(geo, CaW) (the pipeline's agree fusion).
# Size is used only as a gate, never as a depth.
import sys, json, os, pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TH=float(os.environ.get('C37S_TH',1.4)); TA=float(os.environ.get('C37S_TA',0.75))
SUB=os.environ.get('C37S_SUB','sub_xt_Nzw')
SRC=('dis_geo','agree_geo_kind')
def area_at(tt,A,tq,win=0.1):
    m=np.isfinite(A)&(np.abs(tt-tq)<=win)
    return float(np.nanmedian(A[m])) if m.any() else np.nan
rows=[]
for split in ['test','val']:
    P={}
    for l in open(R+'/geo/out/parse_%s_v2.jsonl'%split):
        j=json.loads(l); P[j['id']]=j
    det=pd.read_csv(R+'/submissions/%s_%s.csv'%(SUB,'detail' if split=='test' else 'valdetail'),index_col=0)
    tdir=R+'/geo/out/tracks_%s_v2sc'%split; TC={}
    patch={}
    for i,r in det.iterrows():
        if r['cat']!='D3': continue
        p=P.get(i)
        if p is None or p['q'].get('kind')!='speed' or p['q'].get('t') is None: continue
        t=float(p['q']['t'])
        tk=set(p.get('target_dkeys') or [])
        ks=sorted([(float(e['t']),float(e['d'])) for e in p['depth'] if e['key'] in tk and e.get('t') is not None])
        if len(ks)<2: continue
        (tp,dp),(tl,dl)=ks[-2],ks[-1]
        after=t-tl
        if after<=0: continue
        zlin=dl+(dl-dp)/(tl-tp)*(t-tl) if tl>tp else dl
        v=r.video_id
        if v not in TC:
            fn=tdir+'/%s.json'%v
            TC[v]=json.load(open(fn)) if os.path.exists(fn) else None
        J=TC[v]
        if J is None: continue
        ph=(p.get('target') or {}).get('phrase','')
        ob=next((qq['obj'] for qq in J.get('queries',[]) if qq.get('phrase')==ph and qq.get('obj') is not None),None)
        if ob is None: continue
        fps=float(p['fps'])                         # CSV fps, never the container fps
        tt=np.array(J['fidx'],float)/fps
        BB=np.array([e['st']['box'] if (e and e.get('st') and e['st'].get('box')) else [np.nan]*4 for e in J['objs'][str(ob)]],float)
        A=(BB[:,2]-BB[:,0])*(BB[:,3]-BB[:,1]); Hh=BB[:,3]-BB[:,1]
        side=(BB[:,0]<=2)|(BB[:,2]>=J['W']-2)
        a1=area_at(tt,A,tl); a2=area_at(tt,A,t)
        h1=area_at(tt,Hh,tl); h2=area_at(tt,Hh,t)
        rdh=zlin/(dl*h1/h2) if h1>0 and h2>0 else np.nan   # diagnostic only: height witness (robust to a left/right border cut)
        cut_tl=bool(side[np.abs(tt-tl)<=0.1].any()); cut_t=bool(side[np.abs(tt-t)<=0.1].any())
        zsz=dl*np.sqrt(a1/a2) if a1>0 and a2>0 else np.nan
        rd=zlin/zsz if zsz>0 else np.nan
        src=str(r.answer_source); g=r.g; cs=r.cs
        fire=(after>=TA and rd==rd and (rd>=TH or rd<=1/TH) and src in SRC and g>0 and cs>0)
        rows.append(dict(split=split,id=i,vid=v,q=p['question'][:55],t=t,tl=tl,after=round(after,2),zlin=round(zlin,2),zsz=round(zsz,2),rd=round(rd,3),rd_h=round(rdh,3),cut_tl=cut_tl,cut_t=cut_t,src=src,ans=round(r.answer,3),g=round(g,3) if g==g else g,cs=cs,fire=fire,jfps=J.get('fps')))
        if fire:
            nv=float(np.sqrt(g*cs))
            patch[str(i)]=dict(value=nv,why='rul055: speed %.2fs after the last own depth key; box scale vs linearly extrapolated depth %.2fx -> gm(geo %.2f, CaW %.2f)'%(after,rd,g,cs))
    json.dump(patch,open(R+'/cache/tmp/ag_c37stale_patch_%s.json'%split,'w'),indent=1)
    print(split,'rul055 rows',len(patch),{k:round(v['value'],3) for k,v in patch.items()})
X=pd.DataFrame(rows)
pd.set_option('display.width',320); pd.set_option('display.max_rows',500); pd.set_option('display.max_colwidth',55)
print(X.sort_values('rd').to_string(index=False))
X.to_csv(R+'/cache/tmp/ag_c37stale_scan.csv')
