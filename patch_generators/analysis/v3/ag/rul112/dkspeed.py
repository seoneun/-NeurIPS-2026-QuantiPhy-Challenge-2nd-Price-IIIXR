# rul112: SIM-only D3 speed-at-t bounded by the target's own timed depth keys
# (box-height interpolation, k = d*h consistency gate). Writes patch json for test/val.
import sys, json, os, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/cache/tmp')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
TAG='rul112'
KCONS=1.15
CAWSRC=re.compile(r'^(caw_|dis_caw_routed)')

def norm(s): return re.sub(r'[\s_]+','',str(s).lower())

def run(split, verbose=True):
    P={}
    for l in open(R+f'/geo/out/parse_{split}_v2.jsonl'):
        j=json.loads(l); P[j['id']]=j
    raw=pd.read_csv(R+f'/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    D=pd.read_csv(R+'/submissions/'+('sub_xt_Nzz14_detail.csv' if split=='test' else 'sub_xt_Nzz14_valdetail.csv')).set_index('Unnamed: 0')
    out={}; rows=[]
    for i,j in P.items():
        if i not in D.index: continue
        if str(raw.loc[i,'video_source'])!='simulation': continue          # G1
        if D.loc[i,'cat']!='D3': continue
        q=j['q']
        if q.get('kind')!='speed' or q.get('t') is None or q.get('t1') is not None or q.get('t2') is not None: continue  # G2
        t=float(q['t'])
        tk=[norm(k) for k in (j.get('target_dkeys') or [])]
        if len(tk)!=1: continue
        ks=[(float(e['t']),float(e['d'])) for e in j.get('depth',[]) if e.get('t') is not None and norm(e['key'])==tk[0]]
        if len(ks)<2: continue                                              # G3a
        vid=j['video_id']; fps=float(raw.loc[i,'fps'])
        p=R+f'/geo/out/tracks_{split}_v2sc/{vid}.json'
        if not os.path.exists(p): continue
        T=json.load(open(p)); W=float(T['W'])
        key=(j.get('target') or {}).get('phrase'); oid=None
        for e in T['queries']:
            if e['key'].split('|')[0]==key: oid=e.get('obj'); break
        if oid is None: continue
        O=T['objs'][str(oid)]; fr=[];hh=[];cx=[];cy=[]
        for f,st in zip(T['fidx'],O):
            if st is None or st.get('st') is None or st['st'].get('border',0): continue
            b=st['st']['box']; fr.append(f/fps); hh.append(b[3]-b[1]); cx.append((b[0]+b[2])/2); cy.append((b[1]+b[3])/2)
        fr=np.array(fr);hh=np.array(hh,float);cx=np.array(cx,float);cy=np.array(cy,float)
        rec=dict(split=split,id=i,vid=vid,t=t,ans=D.loc[i,'answer'],src=D.loc[i,'answer_source'],g=D.loc[i,'g'],cs=D.loc[i,'cs'],nk=len(ks))
        if 'y' in D.columns: rec['y']=D.loc[i,'y']
        if len(fr)==0 or (np.abs(fr-t)<=0.5).sum()<10: rec['why']='G3 frames'; rows.append(rec); continue   # G3b
        def at(arr,tt):
            m=np.abs(fr-tt)<=2.5/fps
            return np.median(arr[m]) if m.sum() else np.nan
        kk=np.array([d*at(hh,tk_) for tk_,d in ks])
        if np.isnan(kk).any(): rec['why']='key uncovered'; rows.append(rec); continue
        cons=kk.max()/kk.min(); rec['cons']=cons
        if cons>KCONS: rec['why']='G4'; rows.append(rec); continue          # G4
        k=kk.mean()
        # G5: box height must vary smoothly across the differentiation window [t-0.25,t+0.25]
        # (3-frame median filter, no frame-to-frame jump >15%): a rigid body's depth cannot jump
        # 15% in one frame, so a jump is occlusion / partial box / instance switch.
        mw=np.abs(fr-t)<=0.25+1e-9
        idx=np.where(mw)[0]
        hf=np.array([np.median(hh[max(0,a-1):a+2]) for a in idx])
        jmp=np.max(np.abs(np.diff(hf))/np.minimum(hf[:-1],hf[1:])) if len(hf)>1 else 9
        rec['jump']=jmp
        if jmp>0.15: rec['why']='G5 jump'; rows.append(rec); continue
        h0,h1,hm=at(hh,t-0.25),at(hh,t+0.25),at(hh,t)
        if np.isnan([h0,h1,hm]).any(): rec['why']='edge'; rows.append(rec); continue
        Rt=k/hm; vr=abs(k/h1-k/h0)/0.5
        m=np.abs(fr-t)<=0.25+1e-9
        if m.sum()<4: rec['why']='few lat'; rows.append(rec); continue
        sx=np.polyfit(fr[m],cx[m],1)[0]; sy=np.polyfit(fr[m],cy[m],1)[0]
        vpx=np.hypot(sx,sy)
        vmax=np.hypot(vr,vpx*Rt/(0.5*W))
        mm=str(raw.loc[i,'meas_mode']); mf=raw.loc[i,'meas_f']
        if mm.startswith('3d_calib') and mf==mf and mf>0: est=np.hypot(vr,vpx*Rt/(mf*W))
        else: est=vmax
        sc=100.0 if q.get('unit')=='cm/s' else 1.0
        vr*=sc; vmax*=sc; est*=sc
        rec.update(Rt=Rt,vr=vr,vmax=vmax,est=est,mm=mm,mf=mf)
        a=float(D.loc[i,'answer']); new=None; why=None
        if CAWSRC.match(str(D.loc[i,'answer_source'])) and a>2*vmax:
            new=est; why=f'{TAG} ceil: {a:.3g}>2*vmax {vmax:.3g} -> est {est:.3g} (k-cons {cons:.3f})'
        elif a<0.8*vr:
            new=vr; why=f'{TAG} floor: {a:.3g}<0.8*vr {vr:.3g} (k-cons {cons:.3f})'
        rec['new']=new; rec['why']=why or 'nochange'
        rows.append(rec)
        if new is not None: out[str(i)]={'value':float(new),'why':why}
    X=pd.DataFrame(rows)
    return out,X

if __name__=='__main__':
    allx=[]
    for split in ['test','val']:
        out,X=run(split)
        json.dump(out,open(R+f'/cache/tmp/ag_{TAG}_patch_{split}.json','w'),indent=1)
        print(split,'changed',len(out)); allx.append(X)
    X=pd.concat(allx); print(X.round(3).to_string())
    X.to_csv(R+f'/cache/tmp/ag_{TAG}_scan.csv',index=False)
