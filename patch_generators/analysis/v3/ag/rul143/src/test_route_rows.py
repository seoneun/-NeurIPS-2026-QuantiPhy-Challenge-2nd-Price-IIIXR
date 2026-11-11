"""Test rows our modules will replace in the full recipe (sub_xt_Nzu), by the val routing rules:
  scene3d: scene-geometry plan (h_above, h_under, d_min, d_ctr, d_two, h_gap, w_ground) and the recipe answered with CaW alone
           (caw_nogeo) or a CaW-based 3D size rule (d3size_*)
  kin2d:   planar video, speed / acceleration / gravity prior, the recipe answered with CaW alone (caw_nogeo), and kin2d has a plan
Writes logs/test_rows_scene3d.txt and logs/test_rows_kin2d.txt (comma-separated ids) and prints counts."""
import json
import re
import sys

import numpy as np
import pandas as pd

d = pd.read_csv("submissions/sub_xt_Nzu_detail.csv").set_index("Unnamed: 0")
P = {json.loads(l)["id"]: json.loads(l) for l in open("geo/out/parse_test_v2.jsonl")}
assert set(P) == set(d.index), "parse ids and detail ids differ"
src = open("scripts/ours/scene3d.py").read()
ns = {"re": re, "np": np}
exec(src[src.index("SURF = "):src.index("def video_path")], ns)
GEO = {"h_above", "h_under", "d_min", "d_ctr", "d_two", "h_gap", "w_ground"}
s3 = [i for i in sorted(P) if (ns["plan"](P[i]["question"]) or {}).get("type") in GEO
      and (d.answer_source[i] == "caw_nogeo" or str(d.answer_source[i]).startswith("d3size_"))]
sys.argv = ["kin2d.py", "--stage", "measure", "--split", "test"]
ks = open("scripts/ours/kin2d.py").read()
kn = {"__file__": "scripts/ours/kin2d.py"}
exec(ks[:ks.index("# ----------------------------------------------------------------------------- locate (GPU)")], kn)
k2 = []
for r in kn["rows"]():
    i = r["id"]
    objs, ends, pairs = kn["plan"](r)
    if d.answer_source.get(i) == "caw_nogeo" and (objs or ends or pairs):
        k2.append(i)
open("logs/test_rows_scene3d.txt", "w").write(",".join(map(str, s3)))
open("logs/test_rows_kin2d.txt", "w").write(",".join(map(str, k2)))
print(f"scene3d rows {len(s3)} in {len({P[i]['video_id'] for i in s3})} videos;",
      pd.Series([ns['plan'](P[i]['question'])['type'] for i in s3]).value_counts().to_dict(),
      "| sources", pd.Series([d.answer_source[i] for i in s3]).value_counts().to_dict())
print(f"kin2d rows {len(k2)} in {len({P[i]['video_id'] for i in k2})} videos;",
      pd.Series([P[i]['q'].get('kind') for i in k2]).value_counts().to_dict())
