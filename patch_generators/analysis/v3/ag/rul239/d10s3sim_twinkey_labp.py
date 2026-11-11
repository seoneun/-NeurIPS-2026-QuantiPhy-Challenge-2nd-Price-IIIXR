import json,sys,numpy as np,pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+"/cache/tmp"); sys.path.insert(0,R+"/caw")
from lab_rows import labelled
L=labelled("sub_xt_Nzz38")
P={}
for sp in ("val","test"):
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        p=json.loads(l); P[(sp,p["id"])]=p
pd.set_option("display.width",250)
for i,r in L.iterrows():
    p=P.get((r.split,int(i)))
    if p is None: continue
    q=p["question"].lower()
    if str(r.video_id).startswith("simulation") and ("speed" in q or "acceler" in q) and any(w in q for w in ("person","human","man ","woman","people","walker","pedestrian","boy","girl","runner")):
        print(r.split,i,r.video_id,r["cat"],repr(p["question"][:70]),"| prior",p["prior_text"],"| y",r.y,"ans",round(r.answer,3) if "answer" in r else "", "mg",r.mg,"mc",r.mc,r.answer_source)
