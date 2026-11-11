import re

import numpy as np

from lab3d import FW, parse_depth
from lab_keydist import centre, obj
from lab_traj import load_objs


def disp(r):
    q = r.question.lower().replace("diasplacement", "displacement").replace("soccor", "soccer").replace(
        "baskteball", "basketball")
    m = re.search(r"displacement of (?:the )?(.+?) (?:ball )?(?:between|from)", q)
    ts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*s\b", q)]
    tr = load_objs(r.video_id.strip())
    po = obj(tr["objs"], m.group(1))
    kz = {}
    for tk, n, z in parse_depth(r.depth_info):
        q_ = obj(tr["objs"], n)
        if tk is not None and q_ is not None and q_[1] is po[1]:
            kz[round(tk, 2)] = z
    f = FW * tr["W"]
    P = []
    for t in ts:
        u, v = centre(po[1], tr["fidx"], float(r.fps), t)
        P.append(kz[round(t, 2)] * np.array([(u - tr["W"] / 2) / f, (v - tr["H"] / 2) / f, 1.0]))
    unit = 100.0 if re.search(r"\bin cm\b", q) else 1.0
    return float(np.linalg.norm(P[1] - P[0])) * unit
