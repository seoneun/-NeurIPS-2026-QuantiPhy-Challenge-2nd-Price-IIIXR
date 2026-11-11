# rul145: 3D accel at a single time t: add depth-axis acceleration from a quadratic fit
# of the given camera distances (depth_info) to geo's image-plane acceleration.
import pandas as pd, numpy as np, json, re, sys
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB="sub_xt_Nzz24"; RAW="geo_v4ntrj5sQ2srngsr_d_vw"
UF={"m/s^2":1.0,"cm/s^2":100.0,"mm/s^2":1000.0,"km/s^2":1e-3,"m/s2":1.0,"cm/s2":100.0}
EV=re.compile(r"max height|apex|rebounce|bounce",re.I)
STRONG=("3d_calib",)
def run(split):
    det=pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split=='test' else 'valdetail'}.csv",index_col=0)
    raw=pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv",index_col=0)
    P={}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j=json.loads(l); P[int(j["id"])]=j
    out={}; diag=[]
    for i,j in P.items():
        q=j.get("q") or {}
        if q.get("kind")!="accel" or q.get("t") is None or q.get("t1") is not None: continue
        if i not in raw.index or i not in det.index: continue
        mode=str(raw.loc[i,"meas_mode"])
        if not mode.startswith("3d"): continue
        src=str(det.loc[i,"answer_source"])
        keys=j.get("target_dkeys") or []
        if len(keys)!=1: diag.append((i,"nkeys")); continue
        di=str(raw.loc[i,"depth_info"]); tagged=set()
        for ln in re.split(r"[\n;]",di):
            if EV.search(ln):
                m=re.search(r"t\s*=\s*([\d.]+)",ln)
                tagged.add(round(float(m.group(1)),4) if m else "all")
        if "all" in tagged: diag.append((i,"evtag_unparsed")); continue
        pts=[(float(e["t"]),float(e["d"])) for e in (j.get("depth") or [])
             if e.get("key")==keys[0] and isinstance(e.get("t"),(int,float)) and isinstance(e.get("d"),(int,float))
             and round(float(e["t"]),4) not in tagged]
        pts=sorted(pts)
        if len(pts)<3: diag.append((i,"n<3")); continue
        tk=np.array([p[0] for p in pts]); dk=np.array([p[1] for p in pts])
        if len(set(tk))<len(tk): diag.append((i,"dup_t")); continue
        dd=np.diff(dk)
        if not (np.all(dd>0) or np.all(dd<0)): diag.append((i,"nonmono",j["video_id"])); continue
        t=float(q["t"])
        if t<tk.min()-0.3 or t>tk.max()+0.3: diag.append((i,"t_out")); continue
        c=np.polyfit(tk,dk,2)
        res=float(np.sqrt(np.mean((np.polyval(c,tk)-dk)**2)))
        if len(tk)>3 and res>=0.005: diag.append((i,"resid",res)); continue
        if src.startswith("patch:"): diag.append((i,"patched",src)); continue
        u=UF.get(str(q.get("unit")).replace(" ",""))
        g=raw.loc[i,"geo_value"]
        if u is None or not np.isfinite(g): diag.append((i,"unit/geo")); continue
        az=2*c[0]; gimg=g/u
        v=float(np.sqrt(gimg**2+az**2))*u
        cs=det.loc[i,"cs"]; cur=float(det.loc[i,"answer"])
        fused=v; how="geo"
        if np.isfinite(cs) and cs>0:
            r=v/cs
            if 0.5<=r<=2: fused=float(np.sqrt(v*cs)); how="agree_gm"
            elif not mode.startswith(STRONG): fused=float(cs); how="dis_caw"
        diag.append((i,"cand",j["video_id"],mode,src,round(cur,3),round(g,3),round(az*u,3),round(v,3),round(fused,3),how,len(tk),round(res,5)))
        if fused>1.3*cur:
            out[str(i)]={"value":fused,"why":f"rul145: |a|=sqrt(g_img^2+a_z^2), a_z={az:.3f}m/s^2 from quad fit of {len(tk)} depth keys ({how})"}
    json.dump(out,open(f"{R}/cache/tmp/ag_c79dkquad_patch_{split}.json","w"),indent=1)
    print(split,"fires",len(out))
    for x in diag:
        if x[1] in ("cand","nonmono","resid","patched"): print(" ",x)
    from collections import Counter; print(Counter(x[1] for x in diag))
    return out
O={s:run(s) for s in ("test","val")}
sys.path.insert(0,R+"/cache/tmp"); sys.path.insert(0,R+"/analysis/v3")
from lab_rows import labelled
import analyze_caw as A
L=labelled(SUB)
for s in ("test","val"):
    for k,v in O[s].items():
        m=L[(L.split==s)&(L.index==int(k))] if "split" in L else None
        if m is not None and len(m):
            y=m.y.iloc[0]; print("LAB",s,k,m.m.iloc[0],A.row_mra(np.array([v["value"]]),np.array([y])))
print("lab rows hit:",sum(1 for s in O for k in O[s] if ((L.split==s)&(L.index==int(k))).any()))
