import re

import numpy as np
import pandas as pd

from lab3d import FW, parse_depth
from lab_traj import load_objs, pick


def centre(o, fidx, fps, t):
    if t is None:
        return float(np.nanmedian(o.cx)), float(np.nanmedian(o.cy))
    k = np.asarray(fidx, float)
    tt = np.floor(k * 25.0 / 24.0 + 0.5) / 25.0
    m = np.isfinite(o.cx.values)
    i = np.argsort(np.abs(tt - t) + np.where(m, 0, 99))[:2]
    w = 1.0 / (np.abs(tt[i] - t) + 1e-3)
    return float(np.sum(o.cx.values[i] * w) / w.sum()), float(np.sum(o.cy.values[i] * w) / w.sum())


ALIAS = {"trashbin": "trash bin", "white cup": "rim"}


def obj(objs, name):
    name = re.sub(r"\s+", " ", name.strip())
    if re.search(r"\b(near|far)\b.*\b(person|people)\b", name):
        ps = [po for po in objs if re.search(r"person|people", po[0])]
        ps = sorted(ps, key=lambda po: -np.nanmedian(po[1].bh))
        return ps[0] if "near" in name else ps[1]
    po = pick(objs, name)
    if po is None and name in ALIAS:
        po = pick(objs, ALIAS[name])
    return po


def key_for(keys, objs, name, t):
    po = obj(objs, name)
    best = None
    for tk, n, z in keys:
        q_ = obj(objs, n)
        if q_ is None or q_[1] is not po[1]:
            continue
        if tk is None or t is None or abs(tk - t) < 0.05:
            if best is None or (tk is not None):
                best = (tk, z)
    return po, best


def dist(r):
    q = r.question.lower().replace("diatance", "distance").replace("soccor", "soccer")
    if re.search(r"distance between (?:the )?two people", q):
        names = ("near person", "far person")
    else:
        m = re.search(r"distance between (?:the )?(.+?) and (?:the )?(.+?)(?: at| in|\?|$)", q)
        names = (m.group(1), m.group(2))
    ts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*s\b", q)]
    t = ts[0] if ts else None
    tr = load_objs(r.video_id.strip())
    keys = parse_depth(r.depth_info)
    pa, ka = key_for(keys, tr["objs"], names[0], t)
    pb, kb = key_for(keys, tr["objs"], names[1], t)
    f = FW * tr["W"]
    P = []
    for po, (tk, z) in ((pa, ka), (pb, kb)):
        u, v = centre(po[1], tr["fidx"], float(r.fps), t if tk is not None else None)
        P.append(z * np.array([(u - tr["W"] / 2) / f, (v - tr["H"] / 2) / f, 1.0]))
    unit = 100.0 if re.search(r"\bin cm\b|\(cm\)|cm\?", q) else 1.0
    return float(np.linalg.norm(P[0] - P[1])) * unit


def main(sub, ids, path):
    pd.DataFrame({"id0": ids, "est": [dist(sub.loc[i]) for i in ids]}).to_csv(path, index=False)
