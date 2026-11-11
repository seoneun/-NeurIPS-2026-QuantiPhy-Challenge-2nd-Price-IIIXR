import os, sys
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/geo")
import measure_v3 as MV
MV.FIX = set("abcdef")


def category(res, p):
    mode = str(res.get("mode") or ""); pk = ((p.get("priors") or [{}])[0] or {}).get("kind")
    if not mode.startswith("3d_") or mode.startswith("3d_samedepth"):
        return "2d_" + ("size" if pk == "size" else "dyn")
    if mode.startswith("3d_calib") and "rejected" not in mode and pk != "size":
        return "dyn"
    return "geo"


class _T:
    def __init__(self, tt, uv, sg=None):
        self.t = np.asarray(tt, float); self.uv = np.asarray(uv, float); self.n = len(self.t); self._sg = sg

    def segs(self):
        if self._sg is None:
            self._sg = MV.segment(self.t, self.uv, 0.0)
        return self._sg


def _scale(P, s):
    P = np.array(P, float).copy(); P[:, :2] *= s
    return P


def value(rec, s=1.0):
    calls = (rec or {}).get("calls") or []
    if not calls:
        return None
    c = calls[-1]
    if c["fn"] == "pxq":
        tr = _T(c["tt"], c["uv"], c.get("sg"))
        return MV.px_quantity(tr, c["kind"], c["t"], c["t1"], c["t2"], c["when"], P=_scale(c["P"], s))
    if c["fn"] == "d2":
        v = c["v"][0] if isinstance(c["v"], (tuple, list)) else c["v"]
        v1 = c.get("v1")
        if v is None or v1 is None or not v1 > 0:
            return None
        t1, t2 = _T(c["tt1"], c["uv1"]), _T(c["tt2"], c["uv2"])
        w = MV.M1.dist2_px(t1, t2, c["t"], c["when"], _scale(c["P1"], s), _scale(c["P2"], s))
        return None if w is None else float(w) * float(v) / float(v1)
    return None
