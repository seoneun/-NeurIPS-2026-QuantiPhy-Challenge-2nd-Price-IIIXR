import json, os, re, sys
import numpy as np
R = os.path.abspath(os.environ["QP_ROOT"])
sys.path.insert(0, R + "/geo")
import measure as M1
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
WINGED = re.compile(r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft|hawk|crow|pigeon|parrot|"
                    r"duck|goose|swan|moth|airliner|glider)s?\b")


def key(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def find_track(split, vid, k, dim, phrase, gpx_geo):
    best = None
    for td in ("_v2sc", "_v3nt", "_v2", ""):
        p = f"{R}/geo/out/tracks_{split}{td}/{vid}.json"
        if not os.path.exists(p):
            continue
        tj = json.load(open(p))
        if tj.get("version") == "v2":
            cands = [dict(qq, frames=tj["objs"][str(qq["obj"])]) for qk in ("queries_d", "queries") for qq in tj.get(qk, [])
                     if qq.get("key") == k and qq.get("obj") is not None]
        else:
            cands = [qq for qq in tj.get("queries", []) if qq.get("key") == k and qq.get("frames")]
        for qq in cands:
            T = M1.Track(qq, tj)
            if len(T.oke) < 2:
                continue
            e, _ = T.ext_px(dim, phrase)
            gpx = float(np.percentile(e, 75)) if dim in ("diameter", "radius") else float(np.median(e))
            m = abs(np.log(gpx_geo / gpx)) if gpx > 0 and np.isfinite(gpx_geo) else 9
            if best is None or m < best[0]:
                best = (m, T, tj)
    return best


def fuse(gn, cs):
    if np.isfinite(cs) and cs > 0 and 0.5 <= gn / cs <= 2.0:
        return float(np.exp(0.7 * np.log(gn) + 0.3 * np.log(cs))), "wgm"
    return float(gn), "geo"
