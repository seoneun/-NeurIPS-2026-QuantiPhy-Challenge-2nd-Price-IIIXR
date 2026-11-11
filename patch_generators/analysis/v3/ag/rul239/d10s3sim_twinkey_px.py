import json,numpy as np,pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
raw=pd.read_csv(R+"/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
print([c for c in raw.columns])
for i in ():
    r=raw.loc[i]; print(i,{c:str(r[c])[:150] for c in raw.columns if c not in("depth_info",)})
for v in ("simulation_0026","simulation_0025"):
  for d in ("_v2sc","_v2",""):
    import os
    p=f"{R}/geo/out/tracks_test{d}/{v}.json"
    if not os.path.exists(p): continue
    j=json.load(open(p)); fps=float(raw[raw.video_id==v].fps.iloc[0])
    qs=j.get("queries") or []
    print([q.get("key") for q in qs]); q=next(q for q in qs if str(q.get("key")).split("|")[0] in ("human","person"))
    fr=q["frames"] if "frames" in q else j["objs"][str(q["obj"])]
    F=j["fidx"]; B={f:np.array(e["st"]["box"],float) for f,e in zip(F,fr) if e and e.get("st")}
    fs=sorted(B); hs=[B[f][3]-B[f][1] for f in fs]; ws=[B[f][2]-B[f][0] for f in fs]
    c={f:(B[f][:2]+B[f][2:])/2 for f in fs}
    sp=[np.hypot(*(c[b]-c[a]))/((b-a)/fps) for a,b in zip(fs,fs[1:]) if b-a<=3]
    # robust: displacement over whole span
    tot=np.hypot(*(c[fs[-1]]-c[fs[0]]))/((fs[-1]-fs[0])/fps)
    print(v,d or "v1","fps",fps,"n",len(fs),"med h px",np.median(hs),"med w",np.median(ws),"med px speed",np.median(sp),"net px speed",tot,
          "-> v(1.75 h) med", 1.75*np.median(sp)/np.median(hs),"net",1.75*tot/np.median(hs))
    break
