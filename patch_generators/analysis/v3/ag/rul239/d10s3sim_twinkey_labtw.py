import json,sys,numpy as np,pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+"/cache/tmp"); sys.path.insert(0,R+"/caw")
from lab_rows import labelled
import analyze_caw as A
L=labelled("sub_xt_Nzz38")
for tg in ("rul426","subtwin_px","rul081","rul420","rul075","rul083","rul185"):
    for sp in ("val","test"):
        try: d=json.load(open(f"{R}/cache/tmp/ag_{tg}_patch_{sp}.json"))
        except Exception as e: continue
        for i,e in d.items():
            m=L[(L.split==sp)&(L.index==int(i))]
            if len(m)==0: continue
            r=m.iloc[0]
            print(tg,sp,i,r.video_id,r["cat"],"y",r.y,"patch",round(e["value"],3),"mra",A.row_mra(np.array([e["value"]]),np.array([r.y]))[0] if hasattr(A.row_mra(np.array([e["value"]]),np.array([r.y])),'__len__') else A.row_mra(np.array([e["value"]]),np.array([r.y])),"geo",r.mg if "mg" in r else None, "|",e["why"][:110])
