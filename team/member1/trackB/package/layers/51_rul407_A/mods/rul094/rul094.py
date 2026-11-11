import json, os, re, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = os.environ.get("C53_SUB", "pred_t149")
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRI = re.compile(r"\s*(?:the )?(?:total )?(diameter|length|width|height|size) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*(m|cm|mm)\s*$")
MOV = ("speed", "avgspeed", "accel")

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)

def used_boxes(split, v, ph, tracks):
    if tracks == "v1":
        p = f"{R}/geo/out/tracks_{split}/{v}.json"
        if not os.path.exists(p): return []
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q["key"].split("|")[0] == ph:
                return [b for b in (q.get("det_boxes") or []) if b]
        return []
    for d in ("v2", "v2sc"):
        p = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
        if not os.path.exists(p): continue
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q["key"].split("|")[0] == ph and q.get("obj") is not None:
                return [s["st"]["box"] for s in T["objs"].get(str(q["obj"])) or [] if s and s.get("st")]
    return []

def alt_tracks(split, v, ph):
    out = []
    for d in ("v3nt", "v2sc"):
        p = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
        if not os.path.exists(p): continue
        T = json.load(open(p)); W, H = T.get("W") or 1e9, T.get("H") or 1e9
        for key in ("queries", "queries_d"):
            for q in T.get(key) or []:
                if not isinstance(q, dict) or q["key"].split("|")[0] != ph or q.get("obj") is None: continue
                b = np.array([s["st"]["box"] for s in T["objs"].get(str(q["obj"])) or [] if s and s.get("st")], float).reshape(-1, 4)
                if len(b) < 5: continue
                inb = (b[:, 0] > 2) & (b[:, 1] > 2) & (b[:, 2] < W - 2) & (b[:, 3] < H - 2)
                if inb.sum() < 3: continue
                c = (b[:, :2] + b[:, 2:]) / 2
                sz = float(np.median(np.maximum(b[inb, 2]-b[inb, 0], b[inb, 3]-b[inb, 1])))
                ex = float(np.max(np.linalg.norm(c - np.median(c, 0), axis=1)) / max(sz, 1e-9))
                out.append(dict(dir=d, cfg=key, obj=q["obj"], b=b, inb=inb, ex=ex))
    return out

def dim_px(b, dim):
    w, h = b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]
    if dim == "height": return float(np.median(h))
    if dim == "width": return float(np.median(w))
    return float(np.median(np.maximum(w, h)))
