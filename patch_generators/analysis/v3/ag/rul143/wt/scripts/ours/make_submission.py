#!/usr/bin/env python
"""The teammates' portal-best fusion (sub_xt_Nzu) + our scene3d (scene geometry) + kin2d (planar dynamic-prior rows).

One routing rule for val and test, on the recipe's own answer sources:
  scene3d  a scene-geometry plan (h_above, h_under, d_min, d_ctr, d_two, h_gap, w_ground) with an anchored value, where the recipe
           answered with CaW alone (caw_nogeo) or a CaW-based 3D size rule (d3size_*)
  kin2d    a value (the prior object followed and moving, or a free flight for gravity), where the recipe answered with CaW alone
  a row both modules answer takes kin2d (motion-anchored scale, the rows it was built for)

usage:
  python scripts/ours/make_submission.py val  --s3 geo/out/scene3d_val_detS1.csv  --k2 geo/out/kin2d_val_v5dS1.csv
  python scripts/ours/make_submission.py test --s3 geo/out/scene3d_test_detS.csv --k2 geo/out/kin2d_test_testS.csv \\
         --out submissions/sub_xt_Nzu_ours_id1.csv
val prints the macro MRA; test writes the portal file (same layout as sub_xt_Nzu_id1.csv) and <out>_changes.csv.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["val", "test"])
ap.add_argument("--s3", required=True)
ap.add_argument("--k2", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--allow_tile_anchor", action="store_true",
                help="keep kin2d rows whose prior-object track rests on a tile-search hit (off: such a scale is not used)")
a = ap.parse_args()
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GEO = ["h_above", "h_under", "d_min", "d_ctr", "d_two", "h_gap", "w_ground"]
TH = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


_ns = {"re": __import__("re"), "np": np}
_src = open(os.path.join(R, "scripts/ours/scene3d.py")).read()
exec(_src[_src.index("SURF = "):_src.index("def video_path")], _ns)
plan = _ns["plan"]                                   # scene3d's question-text plan (the fixed lane regex: not "plane")


def route(src, s3, k2, base, questions):
    """{row id: (value, module)} of the rows our modules replace.
    scene3d: 3D videos only (every val row it replaced was 3D; the planar path never replaced a val row); the plan is re-read
    from the question with the current regex. Both modules: a value more than ten times off the recipe's answer (either way)
    is a gross failure and the recipe's answer stays (val replacements span 0.37-6.7x)."""
    out = {}
    ok = lambda v, i: 0.1 <= v / float(base[i]) <= 10.0 if float(base[i]) > 0 else False
    for i, r in s3.iterrows():
        if i not in src.index or pd.isna(r["value"]) or r.get("vt") != "3D":
            continue
        if (plan(questions[i]) or {}).get("type") in GEO and (src[i] == "caw_nogeo" or str(src[i]).startswith("d3size_")) \
                and ok(float(r["value"]), i):
            out[i] = (float(r["value"]), "scene3d")
    for i, r in k2.iterrows():
        if pd.notna(r.get("value")) and i in src.index and src[i] == "caw_nogeo" and ok(float(r["value"]), i) \
                and (a.allow_tile_anchor or not tile_anchor(r["video"])):
            out[i] = (float(r["value"]), "kin2d")          # kin2d over scene3d
    return out


def tile_anchor(video):
    """the kept prior-object track contains a whole-frame tile-search hit: on val such anchors were always another object
    (a static highlight on the dropper, a standing pedestrian), and no accepted val anchor had one"""
    p = os.path.join(os.path.dirname(os.path.abspath(a.k2)), os.path.basename(a.k2)[:-4], f"{video}.json")
    if not os.path.exists(p):
        return False
    import json
    J = json.load(open(p))
    o = J["objs"].get(J.get("prior_obj") or "") or {}
    return any(o.get("src", {}).get(k) == "tile" for k in o.get("pts", {}))


s3 = pd.read_csv(a.s3).set_index("id")
k2 = pd.read_csv(a.k2).set_index("id")
if a.split == "val":
    f = pd.read_csv(f"{R}/submissions/sub_xt_Nzu_valdetail.csv").set_index("Unnamed: 0")
    vq = pd.read_csv(f"{R}/data/val/validation_dataset.csv").set_index("Unnamed: 0").question
    rt = route(f.answer_source, s3, k2, f.answer, vq)
    ans = f.answer.copy()
    for i, (v, _) in rt.items():
        ans[i] = v
    mra = lambda p, y: float((abs(p - y) / y < 1 - TH).mean()) if np.isfinite(p) else 0.0
    for nm, x in (("full recipe (sub_xt_Nzu)", f.answer), ("+ scene3d + kin2d", ans)):
        m = pd.Series([mra(p, y) for p, y in zip(x, f.y)], index=f.index)
        print("val %-26s macro %.4f %s" % (nm, m.groupby(f.cat).mean().mean(), m.groupby(f.cat).mean().round(4).to_dict()))
    print("val rows replaced: %d (scene3d %d, kin2d %d)" % (len(rt), sum(v[1] == "scene3d" for v in rt.values()),
                                                           sum(v[1] == "kin2d" for v in rt.values())))
else:
    sub = pd.read_csv(f"{R}/submissions/sub_xt_Nzu_id1.csv")
    det = pd.read_csv(f"{R}/submissions/sub_xt_Nzu_detail.csv").set_index("Unnamed: 0")
    assert (sub.id.values == det.index.values + 1).all() and (sub.question.values == det.question.values).all()
    rt = route(det.answer_source, s3, k2, pd.Series(sub.parsed_value.values, index=det.index), det.question)
    new = sub.copy()
    ch = []
    for i, (v, mod) in sorted(rt.items()):
        k = i                                             # detail index i is row i of the portal file (id = i + 1)
        ch.append({"id": int(sub.id[k]), "video_id": sub.video_id[k], "question": sub.question[k], "recipe": float(sub.parsed_value[k]),
                   "recipe_source": det.answer_source[i], "ours": v, "module": mod, "ratio": v / float(sub.parsed_value[k])})
        new.loc[k, "parsed_value"] = v
    out = a.out or f"{R}/submissions/sub_xt_Nzu_ours_id1.csv"
    new.to_csv(out, index=False)
    pd.DataFrame(ch).to_csv(out.replace(".csv", "_changes.csv"), index=False)
    md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    print("test rows replaced: %d (scene3d %d, kin2d %d) -> %s  md5 %s" % (
        len(rt), sum(v[1] == "scene3d" for v in rt.values()), sum(v[1] == "kin2d" for v in rt.values()), out, md5))
