import sys, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R+"/cache/tmp"); sys.path.insert(0, R+"/caw")
import analyze_caw as A
from lab_rows import labelled
ids=[73,74,337,432,623,638,642,738,749,831,1038,1040,1041,1048,1050,1051,1079,1258,1293,1306,1330,1390,1395,1510,2068]
d=pd.read_csv(f"{R}/submissions/sub_xt_Nzz23_detail.csv",index_col=0)
du=pd.read_csv(f"{R}/submissions/sub_xt_Nzu_detail.csv",index_col=0)
f=pd.read_csv(f"{R}/submissions/sub_friend_ours2.csv").set_index("Unnamed: 0").parsed_value
z=pd.read_csv(f"{R}/submissions/sub_xt_Nzz23.csv").set_index("Unnamed: 0").parsed_value
L=labelled('sub_xt_Nzz23'); L=L[L.split=='test'] if 'split' in L else L
fz=pd.read_csv(f"{R}/cache/tmp/ag_a42s0_fuzzycopies.csv")
print(L.columns.tolist()); print(fz.columns.tolist())
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',80)
rows=[]
for i in ids:
    y=L.y.get(i) if i in L.index else np.nan
    rows.append(dict(id=i,vid=d.video_id[i],cat=d.cat[i],q=d.question[i][:80],nzz23=round(z[i],3),src23=d.answer_source[i],
      nzu_src=du.answer_source[i],friend=round(f[i],3),g=d.g[i],cs=d.cs[i],y_lab=y))
T=pd.DataFrame(rows); print(T.to_string())
