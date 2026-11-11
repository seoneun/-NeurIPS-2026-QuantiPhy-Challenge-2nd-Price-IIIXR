# rul082: gravity-timing floor for free-flight speed rows.
# For a speed row at time t whose target's image track contains a ballistic arc (image-y parabola,
# gated as free flight), the vertical speed alone is g*|t - t_apex|, so answer = max(answer, g*|t - t_apex|).
# Gates (all label-free):
#  - quadratic fit of centroid y over >= 8 frames around a local image-y minimum: R2 >= 0.99,
#    RMS <= max(2 px, 2% of range), range >= 15 px, curvature > 0, apex inside the window;
#  - t inside the window (or 1 frame past it with y still monotone, i.e. no bounce in between); |t - t_apex| >= 0.1 s;
#  - x(t) monotone over the window (excludes pendulum turning points and hand reversals);
#  - gravity consistency: implied focal c*Z/(g/2) within 0.315-1.19 W, Z = target's own depth key nearest the apex
#    (rows without an own depth key are skipped);
#  - perspective/pitch direction: a rising-phase floor needs the target to recede, a falling-phase floor needs it to
#    approach (then the image apex is no later / no earlier than the world apex and the floor is conservative).
#    Direction from >= 2 own timed depth keys, else from the box-height trend over the window (|log ratio| >= 0.1).
#  - apex image y above the image centre and camera vertical drift (track 'cam' cumsum) between t and apex
#    <= max(3 px, 5% of the arc range).
#  - only video_source lab (captured) rows (applied in rul082.py); CSV fps (parse 'fps').
# Bound: with line-of-sight depression angle b at the apex and radial horizontal speed u, the image apex is
# where v_vert = -u*tan(b); the true speed is then >= floor*cos(b) in the worst case, and >= floor when the
# object is above camera height or u >= floor*sin(2b). relax_dir=1 drops the direction/apex/camera gates,
# relax_dir=2 also drops the own-depth-key/focal gate (label checks only).
import sys, json, os, pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
G=9.8
UNIT={'m/s':1.0,'cm/s':100.0,'km/h':3.6,'mm/s':1000.0}
SUB=sys.argv[1] if len(sys.argv)>1 else 'sub_xt_Nzz4'
def load_parse(split):
    P={}
    for l in open(R+'/geo/out/parse_%s_v2.jsonl'%split):
        j=json.loads(l); P[int(j['id'])]=j
    return P
def objs_for(J, tgt):
    out=[]
    for qn in ('queries','queries_d'):
        for q in J.get(qn,[]):
            k=q['key'].split('|')
            if k[0].strip().lower()==tgt.strip().lower() and (len(k)<2 or k[1]==''):
                if q.get('obj') is not None and str(q['obj']) not in [o for _,o in out]: out.append((qn,str(q['obj'])))
    return out
def series(J, oid, fps):
    fidx=J['fidx']; lst=J['objs'].get(oid)
    if lst is None: return None
    T=[];X=[];Y=[];Hh=[]
    for fi,e in zip(fidx,lst):
        if e is None or e.get('st') is None: continue
        b=e['st']['box']
        T.append(fi/fps); X.append((b[0]+b[2])/2); Y.append((b[1]+b[3])/2); Hh.append(max(b[3]-b[1],1.0))
    return np.array(T),np.array(X),np.array(Y),np.array(Hh)
def fit_arcs(T,Y):
    arcs=[]; n=len(T)
    for i in range(2,n-2):
        if not (Y[i]<=Y[i-1] and Y[i]<=Y[i+1] and Y[i]<Y[i-2] and Y[i]<Y[i+2]): continue
        best=None
        for a in range(i-2, -1, -1):
            for b in range(i+2, n):
                if b-a+1<8: continue
                tt=T[a:b+1]; yy=Y[a:b+1]
                p=np.polyfit(tt,yy,2); r=yy-np.polyval(p,tt)
                rms=np.sqrt(np.mean(r**2)); rng=yy.max()-yy.min()
                if p[0]<=0 or rng<15: continue
                ss=np.sum((yy-yy.mean())**2); r2=1-np.sum(r**2)/max(ss,1e-9)
                if r2>=0.99 and rms<=max(2.0,0.02*rng):
                    if best is None or (b-a)>(best[1]-best[0]): best=(a,b,p,r2,rms)
        if best and not any(x[0]==best[0] and x[1]==best[1] for x in arcs): arcs.append(best)
    return arcs
def floor_for(j, J, relax_dir=0, return_all=False):
    q=j.get('q') or {}
    t=float(q['t']); fps=float(j['fps']); W=float(J.get('W',854))
    tk=set(j.get('target_dkeys') or [])
    keys=[(d.get('t'),d['d']) for d in (j.get('depth') or []) if d.get('key') in tk and d.get('d')]
    if not keys and relax_dir<2: return None
    best=None
    for qn,oid in objs_for(J,q.get('target') or ''):
        s=series(J,oid,fps)
        if s is None or len(s[0])<8: continue
        T,X,Y,Hh=s
        for (a,b,p,r2,rms) in fit_arcs(T,Y):
            tap=-p[1]/(2*p[0])
            if not (T[a]<=tap<=T[b]): continue
            if not (T[a]-1.01/fps<=t<=T[b]+1.01/fps): continue
            k=int(np.argmin(abs(T-t)))
            if t>T[b] and not (k>b and np.all(np.diff(Y[b:k+1])>0)): continue
            if t<T[a] and not (k<a and np.all(np.diff(Y[k:a+1])<0)): continue
            if abs(t-tap)<0.1: continue
            Hh_img=float(J.get('H',480))
            yap=float(np.polyval(p,tap))
            apex_ok = yap < Hh_img/2
            if not apex_ok and not relax_dir: continue
            cam=np.array(J.get('cam') or [],float)
            camdy=0.0
            if len(cam):
                cy=np.cumsum(cam[:,1]); fidx=np.array(J['fidx'])
                lo=int(round(min(t,tap)*fps)); hi=int(round(max(t,tap)*fps))
                m=(fidx>=lo)&(fidx<=hi)
                if m.sum()>=2:
                    seg=cy[np.where(m)[0]] if len(cy)==len(fidx) else cy[np.clip(fidx[m],0,len(cy)-1)]; camdy=float(seg.max()-seg.min())
            rng=float(Y[a:b+1].max()-Y[a:b+1].min())
            if camdy>max(3.0,0.05*rng) and not relax_dir: continue
            xx=X[a:b+1]
            if (xx.max()-xx.min()) > abs(xx[-1]-xx[0])+6: continue
            if keys:
                kz=sorted([(abs((kt if kt is not None else tap)-tap),d) for kt,d in keys])[0][1]
                fz=p[0]*kz/(0.5*G)
                if not (0.315*W<=fz<=1.19*W) and relax_dir<2: continue
            else: fz=np.nan
            tk2=sorted([(kt,d) for kt,d in keys if kt is not None])
            dz=np.nan
            if len(tk2)>=2 and tk2[-1][0]>tk2[0][0]: dz=(tk2[-1][1]-tk2[0][1])/(tk2[-1][0]-tk2[0][0])
            if dz==dz and abs(dz)>0.05: app = dz<0; dsrc='keys'
            else:
                hr=np.log(np.median(Hh[b-2:b+1])/np.median(Hh[a:a+3]))
                if abs(hr)<0.1 and not relax_dir: continue
                app = hr>0; dsrc='box' if abs(hr)>=0.1 else 'none'
            phase='rise' if t<tap else 'fall'
            dir_ok = not ((phase=='rise' and app) or (phase=='fall' and not app))
            if not dir_ok and not relax_dir: continue
            rec=dict(dir_ok=dir_ok,apex_ok=apex_ok,yap=yap,camdy=camdy,obj=qn[:9]+':'+oid,tap=tap,win=(T[a],T[b]),n=b-a+1,r2=r2,fzW=fz/W,phase=phase,dsrc=dsrc,vlb=G*abs(t-tap))
            if best is None or rec['n']>best['n']: best=rec
    return best
