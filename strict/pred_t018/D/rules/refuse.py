import math
import numpy as np
BAND = math.log(2.0)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid", "sruler_geo", "geo_only_mode"}
GM = {"dis_size_gm", "dis_size_low_gm"}
D3GM = {"d3lab_gm", "d3lab_inband_gm", "rul263_unstable_gm", "rul263_dimgm"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}


def agree_w(kind, cat):
    if kind == "size":
        return 0.7 if cat == "D2" else 0.65
    return 0.5


def fuse(r, gn, kind):
    s, a, g = str(r["answer_source"]), float(r["answer"]), float(r["g"])
    if not (np.isfinite(gn) and gn > 0):
        return None, "no new geo"
    if s.startswith("mo_guard:"):
        if s[9:] in GEO and np.isfinite(g) and g > 0 and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    c = float(r["cs"]) if np.isfinite(r["cs"]) else np.nan
    gok = np.isfinite(g) and g > 0
    in_old = gok and np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if kind in GEOK:
        if s in AGREE or s in GEO or s in GM:
            return gn, (f"{s} -> in 2x band, geo kind -> geo" if in_new else f"{s} -> outside 2x band -> geo")
        if s in CAWD and in_new and not in_old:
            return gn, f"{s} entered 2x band, geo kind -> geo"
        return None, s
    if s in AGREE or s in D3GM:
        if not gok:
            return None, s
        if s in AGREE and not in_new:
            return gn, f"{s} left 2x band -> geo"
        lg = math.log(g / c) if (np.isfinite(c) and c > 0) else 0.0
        w = math.log(a / c) / lg if abs(lg) > 1e-6 else agree_w(kind, r["cat"])
        if not (0.3 <= w <= 1.0):
            w = agree_w(kind, r["cat"])
        return a * (gn / g) ** w, f"{s} w{w:.2f}"
    if s in GEO:
        if s == "dis_geo" and in_new:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return gn, s
    if s in GM:
        if in_new:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return math.sqrt(gn * c), s
    if s in CAWD:
        if in_new and not in_old:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"{s} entered 2x band -> gm w{w:.2f}"
        return None, s
    return None, s
