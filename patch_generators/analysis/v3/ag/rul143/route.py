"""Rows the teammate's routing would send to scene3d / kin2d, for a given fusion detail (Nzu or Nzz23), test and val."""
import json, re, sys, os
import numpy as np, pandas as pd
W = os.path.join(os.environ.get("QP_ROOT", "."), "analysis/v3/ag/rul143/wt")
os.chdir(W)
src = open("scripts/ours/scene3d.py").read(); ns = {"re": re, "np": np}
exec(src[src.index("SURF = "):src.index("def video_path")], ns)
GEO = {"h_above", "h_under", "d_min", "d_ctr", "d_two", "h_gap", "w_ground"}
ks = open("scripts/ours/kin2d.py").read()
out = {}
for split in ["test", "val"]:
    sys.argv = ["kin2d.py", "--stage", "measure", "--split", split]
    kn = {"__file__": "scripts/ours/kin2d.py"}
    exec(ks[:ks.index("# ----------------------------------------------------------------------------- locate (GPU)")], kn)
    P = {json.loads(l)["id"]: json.loads(l) for l in open(f"geo/out/parse_{split}_v2.jsonl")}
    for nm in ["sub_xt_Nzu", "sub_xt_Nzz23"]:
        d = pd.read_csv(f"submissions/{nm}_{'detail' if split=='test' else 'valdetail'}.csv").set_index("Unnamed: 0")
        s3 = [i for i in sorted(P) if (ns["plan"](P[i]["question"]) or {}).get("type") in GEO and P[i]["video_type"][1] == "3"
              and (d.answer_source[i] == "caw_nogeo" or str(d.answer_source[i]).startswith("d3size_"))]
        s3all = [i for i in sorted(P) if (ns["plan"](P[i]["question"]) or {}).get("type") in GEO and P[i]["video_type"][1] == "3"]
        k2 = []; k2all = []
        for r in kn["rows"]():
            i = r["id"]; objs, ends, pairs = kn["plan"](r)
            if objs or ends or pairs:
                k2all.append(i)
                if d.answer_source.get(i) == "caw_nogeo": k2.append(i)
        out[f"{split}_{nm}_s3"] = s3; out[f"{split}_{nm}_k2"] = k2
        out[f"{split}_s3all"] = s3all; out[f"{split}_k2all"] = k2all
        print(split, nm, "scene3d", len(s3), "kin2d", len(k2), "| plan-any s3", len(s3all), "k2", len(k2all))
json.dump(out, open(os.path.join(os.environ.get("QP_ROOT", "."), "analysis/v3/ag/rul143/routes.json"), "w"))
for split in ["test", "val"]:
    for m in ["s3", "k2"]:
        a_, b_ = set(out[f"{split}_sub_xt_Nzu_{m}"]), set(out[f"{split}_sub_xt_Nzz23_{m}"])
        print(split, m, "Nzz23-only", sorted(b_ - a_), "Nzu-only", sorted(a_ - b_))
