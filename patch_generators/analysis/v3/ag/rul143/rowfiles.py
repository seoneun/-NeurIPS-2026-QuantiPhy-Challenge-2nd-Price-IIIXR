import os
import json
r=json.load(open(os.path.join(os.environ.get("QP_ROOT", "."), "analysis/v3/ag/rul143/routes.json")))
W=os.path.join(os.environ.get("QP_ROOT", "."), "analysis/v3/ag/rul143/wt/logs/")
open(W+"test_rows_scene3d.txt","w").write(",".join(map(str,sorted(set(r["test_sub_xt_Nzu_s3"])|set(r["test_sub_xt_Nzz23_s3"])))))
open(W+"test_rows_kin2d.txt","w").write(",".join(map(str,sorted(set(r["test_sub_xt_Nzu_k2"])|set(r["test_sub_xt_Nzz23_k2"])))))
open(W+"val_rows_scene3d.txt","w").write(",".join(map(str,sorted(set(r["val_s3all"])))))
open(W+"val_rows_kin2d.txt","w").write(",".join(map(str,sorted(set(r["val_k2all"])))))
