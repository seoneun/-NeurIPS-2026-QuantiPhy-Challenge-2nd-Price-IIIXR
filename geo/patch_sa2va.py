#!/usr/bin/env python
import os, sys, json, shutil, argparse
from collections import Counter
import numpy as np

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/geo")
import track_common as tc
os.environ.setdefault("QP_SCENE_HEAD", "1")
from measure_v3 import scene_phrase

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--mode", default="fill", choices=["fill", "all", "agree", "scene", "fillweak"])
ap.add_argument("--src", default="v2mv")
ap.add_argument("--out", default="v2sa")
ap.add_argument("--in", dest="inp", default="")
ap.add_argument("--iou", type=float, default=0.1)
ap.add_argument("--scene", action="store_true")
ap.add_argument("--t_owl", type=float, default=None)
a = ap.parse_args()
if a.t_owl is None:
    tp = os.environ.get("QP_OWL_JSON") or f"{R}/geo/out/owl_threshold_v2.json"
    a.t_owl = json.load(open(tp))["T_owl"] if os.path.exists(tp) else 0.2
SA = json.load(open(f"{R}/geo/out/sa2va_{a.split}{a.inp}.json"))
SRC = f"{R}/geo/out/tracks_{a.split}_{a.src}" if a.src != "v1" else f"{R}/geo/out/tracks_{a.split}"
OUT = f"{R}/geo/out/tracks_{a.split}_{a.out}"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(".json"):
        shutil.copy(f"{SRC}/{f}", f"{OUT}/{f}")


def valid(st, H, W):
    ok = [s for s in st if s]
    if len(ok) < 0.4 * len(st):
        return False
    return float(np.median([s["area"] for s in ok])) <= 0.25 * H * W


def weak(q, fr, N):
    if fr is None or len(fr) != N:
        return "no_track"
    n = sum(1 for o in fr if o and o.get("st"))
    if n < 6:
        return "few_frames"
    gd, owl = q.get("best_gd") or 0, q.get("best_owl") or 0
    if not (gd >= 0.3 or (gd >= 0.15 and owl >= a.t_owl)):
        return "low_det"
    di = q.get("det_iou") or []
    if di and float(np.mean(di)) < 0.5:
        return "det_iou"
    return None


def static_vs_moving(fr, st, N):
    if fr is None or len(fr) != N:
        return False
    def span(sts):
        c = np.array([s_["c"] for s_ in sts if s_], float)
        ar = [s_["area"] for s_ in sts if s_]
        if len(c) < 3:
            return 0.0, 1.0, 1.0
        return float(np.linalg.norm(c.max(0) - c.min(0))), float(np.median(ar)), float(np.median([s_["rr"][0] for s_ in sts if s_]))
    sd, _, size_d = span([o.get("st") if isinstance(o, dict) else None for o in fr])
    ss, area_s, _ = span(st)
    return sd < 0.25 * size_d and ss >= max(8.0, 0.5 * np.sqrt(max(area_s, 1.0)))


def seq_iou(fr, st):
    v = [tc.iou(o["st"]["box"], s["box"]) for o, s in zip(fr, st) if o and o.get("st") and s]
    return float(np.mean(v)) if v else 0.0


stat = Counter()
npatched = 0
for v, keys in sorted(SA.items()):
    p = f"{SRC}/{v}.json"
    if not os.path.exists(p):
        continue
    tj = json.load(open(p))
    if not tj.get("objs"):
        stat["v1_skip"] += 1
        continue
    N, H, W = len(tj["fidx"]), tj.get("H") or 480, tj.get("W") or 854
    newid = {}
    hit = 0
    for qk in ("queries", "queries_d"):
        for q in tj.get(qk, []):
            s = keys.get(q.get("key"))
            if s is None or not s.get("nseg") or len(s["st"]) != N:
                stat["no_sa2va"] += 1
                continue
            if not valid(s["st"], H, W):
                stat["sa2va_invalid"] += 1
                continue
            fr = tj["objs"].get(str(q.get("obj"))) if q.get("obj") is not None else None
            why = weak(q, fr, N)
            io = seq_iou(fr, s["st"]) if fr is not None and len(fr) == N else 0.0
            stat[f"iou_{'hi' if io >= 0.5 else ('mid' if io >= a.iou else 'lo')}"] += 1
            scene = bool(a.scene and scene_phrase(q.get("phrase")))
            weak_fill = False
            if a.mode in ("scene", "fillweak"):
                scene = bool(scene_phrase(q.get("phrase")))
                if a.mode == "fillweak" and not scene:
                    st_static = static_vs_moving(fr, s["st"], N) if why is None else False
                    weak_fill = why is not None or st_static
                    if st_static:
                        why = "static"
                else:
                    why = None
                if scene:
                    why = "scene"
            take = (a.mode == "all") or (why is not None) or (a.mode == "agree" and io < a.iou) or scene
            if scene and why is None and a.mode != "all":
                why = "scene"
            if not take:
                continue
            if a.mode == "all" and io >= 0.8:
                stat["all_same"] += 1
                continue
            stat[f"take_{why or ('agree' if a.mode == 'agree' else 'all')}"] += 1
            if q["key"] not in newid:
                oid = str(max([int(x) for x in tj["objs"]] + [0]) + 1)
                tj["objs"][oid] = [dict(st=x, lg=1.0) if x else dict(st=None, lg=-10.0) for x in s["st"]]
                newid[q["key"]] = oid
            q.update(obj=int(newid[q["key"]]), best_gd=max(q.get("best_gd") or 0, 0.5), best_owl=max(q.get("best_owl") or 0, 0.5),
                     best_score=max(q.get("best_score") or 0, 0.5), det_iou=[1.0] * max(1, len(q.get("det_iou") or [])), sa2va=True,
                     sa2va_weak=bool(weak_fill))
            hit += 1
    if hit:
        json.dump(tj, open(f"{OUT}/{v}.json", "w"))
        npatched += 1
print(a.mode, dict(stat), "videos patched", npatched)
