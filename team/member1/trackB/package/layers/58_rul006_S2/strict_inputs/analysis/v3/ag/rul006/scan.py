import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
MOTION = {"speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel", "orbitdiam"}


def load_tracks(split, tv, vid):
    p = f"{R}/geo/out/tracks_{split}{tv}/{vid}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def obj_stats(lst):
    fr = [(k, x["st"]) for k, x in enumerate(lst) if x and x.get("st")]
    if len(fr) < 3:
        return None
    c = np.array([s["c"] for _, s in fr], float)
    b = np.array([s["box"] for _, s in fr], float)
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    size = float(np.median(np.sqrt(w * h)))
    span = float(np.linalg.norm(c.max(0) - c.min(0)))
    return dict(ks=np.array([k for k, _ in fr]), c=c, w=float(np.median(w)), h=float(np.median(h)), size=size, span=span,
                cover=len(fr) / len(lst), asp=float(np.median(np.maximum(w, h) / np.maximum(1, np.minimum(w, h)))))




def same_traj(a, b):
    ka = dict(zip(a["ks"].tolist(), a["c"])); d = [np.linalg.norm(ka[k] - cb) for k, cb in zip(b["ks"].tolist(), b["c"]) if k in ka]
    return len(d) >= 3 and np.median(d) < 0.5 * max(a["size"], b["size"])




def qobj(tj, phrase, which="queries"):
    if tj is None:
        return None, None
    for q in tj.get(which) or []:
        if (q.get("phrase") or "").strip() == phrase:
            return q.get("obj"), q
    return None, None


