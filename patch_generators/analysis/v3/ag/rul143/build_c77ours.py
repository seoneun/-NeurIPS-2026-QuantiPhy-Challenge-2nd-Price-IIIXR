"""rul143: the label-free, evidence-supported subset of the teammate's scene3d / kin2d modules (branch ours-scene3d-kin2d,
commit d057b1d), routed on sub_xt_Nzz23's own answer sources.

Inputs: frozen outputs of the teammate's modules, re-run on espresso (H100 NVL, strict determinism; bit-identical to the
teammate's test values) in analysis/v3/ag/rul143/wt/geo/out/:
  scene3d_{test_detS,val_detS1}.csv   kin2d_{test_testS2,val_v5dS1}.csv (+ kin2d_<split><tag>/<video>.json for the tile check)

Rule (one condition for val and test; no row or video ids):
  scene3d  3D video; Nzz23 answered with CaW alone (caw_nogeo) or a CaW-based 3D size rule (d3size_*); the question plan is
           a clear height to an underside (h_under) or a minimum distance (d_min) between two non-wall parts; the metres come
           from >= 3 stated camera distances that agree within 20 %.
           Dropped types: h_above (pier/cabin heights: the labelled rows got worse), w_ground (lane/road: "lane" is ambiguous
           with the whole carriageway and the two roads of one video got the same mask), walls (wall masks, known failure).
  kin2d    planar video with a speed / acceleration / gravity prior; Nzz23 answered with CaW alone; the question is a motion
           quantity of a followed object or a distance between two objects (trajectory / box centres), scaled by the prior
           object's own motion (prior object moving, track not resting on a whole-frame tile-search hit).
           Dropped: two-end-point sizes (ceiling / room heights hit the frame border, tiles, TV, parking spaces).
  both     the module value within 0.1-10x of Nzz23's answer.
Writes cache/tmp/ag_c77ours_patch_{test,val}.json  {row_id: {value, why}}  (final answers in the question's unit).
"""
import json, os, re, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
C = f"{R}/analysis/v3/ag/rul143"
W = f"{C}/wt"
BASE = os.environ.get("C77_BASE", "sub_xt_Nzz23")
TAGS = {"test": ("_detS", "_testS2"), "val": ("_detS1", "_v5dS1")}

src = open(f"{W}/scripts/ours/scene3d.py").read()
ns = {"re": re, "np": np}
exec(src[src.index("SURF = "):src.index("def video_path")], ns)
plan3 = ns["plan"]
WALL = re.compile(r"\bwalls?\b")


def kin_plan(split):
    ks = open(f"{W}/scripts/ours/kin2d.py").read()
    sys.argv = ["kin2d.py", "--stage", "measure", "--split", split]
    kn = {"__file__": f"{W}/scripts/ours/kin2d.py"}
    exec(ks[:ks.index("# ----------------------------------------------------------------------------- locate (GPU)")], kn)
    out = {}
    for r in kn["rows"]():
        objs, ends, pairs = kn["plan"](r)
        out[r["id"]] = "motion" if objs else "pair" if pairs else "ends" if ends else None
    return out


def tile_anchor(split, tag, video):
    p = f"{W}/geo/out/kin2d_{split}{tag}/{video}.json"
    if not os.path.exists(p):
        return False
    J = json.load(open(p))
    o = J["objs"].get(J.get("prior_obj") or "") or {}
    return any(o.get("src", {}).get(k) == "tile" for k in o.get("pts", {}))


def build(split):
    t3, t2 = TAGS[split]
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    if split == "test":
        sub = pd.read_csv(f"{R}/submissions/{BASE}.csv").set_index("Unnamed: 0").parsed_value
        base = sub.reindex(det.index)
    else:
        base = det.answer
        det["question"] = pd.read_csv(f"{R}/data/val/validation_dataset.csv").set_index("Unnamed: 0").question.reindex(det.index)
    srcs = det.answer_source.astype(str)
    ok = lambda v, i: np.isfinite(v) and base[i] > 0 and 0.1 <= v / float(base[i]) <= 10.0
    out = {}
    s3 = pd.read_csv(f"{W}/geo/out/scene3d_{split}{t3}.csv").set_index("id")
    for i, r in s3.iterrows():
        if i not in det.index or r.get("vt") != "3D" or pd.isna(r["value"]):
            continue
        if not (srcs[i] == "caw_nogeo" or srcs[i].startswith("d3size_")):
            continue
        pl = plan3(det.question[i]) or {}
        if pl.get("type") not in ("h_under", "d_min"):
            continue
        if WALL.search(str(pl.get("x") or "")) or WALL.search(str(pl.get("y") or "")):
            continue
        if not (r.get("n_agree", 0) >= 3 and r.get("scale_src") == "stated distances"):
            continue
        v = float(r["value"])
        if ok(v, i):
            out[int(i)] = {"value": v, "why": f"rul143 scene3d {pl['type']} (MoGe-2 + SAM2.1 + Qwen3-VL-32B parts, scale from "
                                             f"{int(r['n_agree'])} agreeing stated distances; was {srcs[i]} {base[i]:.4g})"}
    kp = kin_plan(split)
    k2 = pd.read_csv(f"{W}/geo/out/kin2d_{split}{t2}.csv").set_index("id")
    for i, r in k2.iterrows():
        if i not in det.index or pd.isna(r.get("value")) or srcs[i] != "caw_nogeo":
            continue
        if kp.get(i) not in ("motion", "pair"):
            continue
        if tile_anchor(split, t2, r["video"]):
            continue
        v = float(r["value"])
        if ok(v, i):
            out[int(i)] = {"value": v, "why": f"rul143 kin2d {kp[i]} ({r['how']}; m/px from the prior object's motion, "
                                             f"{r['scale_why']}; was caw_nogeo {base[i]:.4g})"}
    json.dump({str(k): v for k, v in sorted(out.items())}, open(f"{R}/cache/tmp/ag_c77ours_patch_{split}.json", "w"), indent=1)
    print(split, "rows", len(out), {k: round(v["value"], 4) for k, v in sorted(out.items())})
    return out, det, base


if __name__ == "__main__":
    for s in sys.argv[1:] or ("test", "val"):
        build(s)
