import json, os, re, sys, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = os.environ.get("C95_SUB", "pred_t108")
TAG = "rul205"
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRI = re.compile(r"\s*(?:the )?(?:total )?(diameter|length|width|height|size) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*(m|cm|mm)\s*$")
MOV = ("speed", "avgspeed", "accel", "disp", "path")
ALTS = ("v3nt", "v3", "v3nt32")


def key(o):
    if not isinstance(o, dict) or not o.get("phrase"): return None
    return o["phrase"].lower() + "|" + ",".join(o.get("where") or [])


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)


def boxes(T, K, cfg="queries_d"):
    for q in T.get(cfg) or []:
        if isinstance(q, dict) and q.get("key") == K and q.get("obj") is not None:
            return [s["st"]["box"] if s and s.get("st") else None for s in T["objs"].get(str(q["obj"])) or []]
    return None


def used(split, v, K, mt):
    if mt == "v1":
        p = f"{R}/geo/out/tracks_{split}/{v}.json"
        if not os.path.exists(p): return None
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q.get("key") == K: return [b if b else None for b in (q.get("det_boxes") or [])]
        return None
    p = f"{R}/geo/out/tracks_{split}_v2sc/{v}.json"
    if not os.path.exists(p): return None
    return boxes(json.load(open(p)), K)


def stats(bl):
    b = np.array([x for x in bl if x is not None], float).reshape(-1, 4)
    if len(b) == 0: return None
    c = (b[:, :2] + b[:, 2:]) / 2
    side = float(np.median(np.maximum(b[:, 2]-b[:, 0], b[:, 3]-b[:, 1])))
    d = np.linalg.norm(c - np.median(c, 0), axis=1)
    return dict(b=b, n=len(b), side=side, med_exc=float(np.median(d)) / max(side, 1e-9),
                rng_exc=float(np.linalg.norm(np.ptp(c, 0))) / max(side, 1e-9))


def dim_px(b, dim, W, H):
    inb = (b[:, 0] > 2) & (b[:, 1] > 2) & (b[:, 2] < W - 2) & (b[:, 3] < H - 2)
    if inb.sum() < 3: return None
    b = b[inb]; w, h = b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]
    return float(np.median(h if dim == "height" else w if dim == "width" else np.maximum(w, h)))
