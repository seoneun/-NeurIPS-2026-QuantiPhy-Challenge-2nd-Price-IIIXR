"""P1 part 1: prior identity collapse.

In a 3D video whose prior is a moving object, measure_v3 calibrates the focal from the prior's track. When that track is
the same SAM2 object as a DIFFERENTLY named object's track (median per-frame IoU of the grounding boxes > 0.8 over >= 4
frames) and the two phrases conflict (conflict.phrase_conflict: disjoint colour / type modifier or a different head noun),
the prior grounding landed on the other object, so the prior-calibrated focal measures the wrong object's motion.

usage: python analysis/v3/ag/p1pf/collapse.py   -> cache/tmp/ag_p1pf_collapse_{split}.json  {video_id: [hits]}
Inputs only: track files, parse. No labels."""
import os, sys, json, glob
import numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3/ag/p1pf")
from conflict import phrase_conflict  # noqa: E402

IOU_MIN, NMIN = 0.8, 4


def iou(b1, b2):
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])); iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1])); I = ix * iy
    U = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - I
    return I / U if U > 0 else 0.0


def collapse(split, trk="v2sc"):
    P3 = set()
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        x = json.loads(ln)
        if x.get("depth"):
            P3.add(x["video_id"])
    out = {}
    for f in sorted(glob.glob(f"{R}/geo/out/tracks_{split}_{trk}/*.json")):
        T = json.load(open(f)); vid = T.get("video_id") or os.path.basename(f)[:-5]
        if vid not in P3:
            continue
        qs = (T.get("queries_d") or []) + (T.get("queries") or [])
        for q in (T.get("queries_d") or []):
            if "prior" not in (q.get("roles") or []):
                continue
            for q2 in qs:
                if q2 is q or q2["phrase"].lower() == q["phrase"].lower():
                    continue
                v = [iou(a, b) for a, b in zip(q.get("det_boxes") or [], q2.get("det_boxes") or []) if a is not None and b is not None]
                if len(v) < NMIN or np.median(v) <= IOU_MIN:
                    continue
                c, why = phrase_conflict(q["phrase"], q2["phrase"])
                if c:
                    h = dict(prior=q["phrase"], other=q2["phrase"], iou=round(float(np.median(v)), 3), n=len(v), why=why)
                    if h not in out.setdefault(vid, []):
                        out[vid].append(h)
    return out


if __name__ == "__main__":
    for sp in ("val", "test"):
        C = collapse(sp)
        json.dump(C, open(f"{R}/cache/tmp/ag_p1pf_collapse_{sp}.json", "w"), indent=1)
        print(f"===== {sp}: prior identity collapse in {len(C)} 3D videos")
        for v, hs in C.items():
            for h in hs:
                print(f"  {v:28s} prior {h['prior']!r:18} == {h['other']!r:22} iou {h['iou']:.2f} n {h['n']:2d}  {h['why']}")
