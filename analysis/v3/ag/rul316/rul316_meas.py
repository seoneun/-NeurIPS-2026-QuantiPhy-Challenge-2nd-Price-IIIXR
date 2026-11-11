import os, re, json
import numpy as np
import cv2
import rul316_lib as L

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6,
       "m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001}
PART = re.compile(r"\b(shoulder|shoulders|arm|arms|leg|legs|head|foot|feet|torso|waist|chest|knee|elbow)\b", re.I)
_C = {}


def _json(fn):
    if fn not in _C:
        _C[fn] = json.load(open(fn)) if os.path.exists(fn) else None
    return _C[fn]


def groups(split):
    G = _json(f"{R}/analysis/v3/compliance/qtrack/out/groups_{split}.json")
    return {r: x["gid"] for x in G["groups"] for r in x["rows"]}, G["videos"]


def qtrack(split, gid):
    return _json(f"{R}/geo/out/tracks_{split}_qv3ntmv_mq3/{gid}.json")


def itrack(split, vid):
    return _json(f"{R}/cache/tmp/rul316_tracks_{split}/{vid}.json")


def head(ph):
    ph = re.sub(r"\b(the|a|an|middle|left|right|first|second|third)\b", " ", str(ph or "").lower())
    return re.sub(r"\s+", " ", ph).strip()


def qobj(T, role):
    out = []
    for q in T.get("queries") or []:
        if role in (q.get("roles") or []) and q.get("obj") is not None and str(q["obj"]) in T["objs"]:
            out.append((q["key"], L.ser_qtrack(T, q["obj"])))
    return out


def pick_inst(IT, phrase_words):
    ph = head(phrase_words)
    if IT is None or ph not in IT["phrases"]:
        return None, f"no instance phrase '{ph}'"
    ms = L.instances(IT, ph)
    if not ms:
        return None, f"no '{ph}' instance"
    if re.search(r"\bmiddle\b", str(phrase_words), re.I):
        m = L.middle_of(ms)
        return (m, f"middle of {len(ms)} '{ph}'") if m is not None else (None, f"{len(ms)} '{ph}': no middle")
    return ms[0], f"top '{ph}' (score {ms[0]['score']})"


def frames_of(path, ks):
    c = cv2.VideoCapture(path); out = []
    for k in ks:
        c.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = c.read()
        if ok:
            if f.shape[0] != 480:
                f = cv2.resize(f, (int(round(f.shape[1] * 480 / f.shape[0])), 480), interpolation=cv2.INTER_AREA)
            out.append(f)
    return out


def ruler(split, p, gid, vpath):
    pr = [x for x in p.get("priors") or [] if x.get("value_si")]
    if len(pr) != 1 or pr[0].get("kind") != "size":
        return None, "prior not one size"
    pr = pr[0]; po = str(pr.get("obj") or "").lower(); dim = pr.get("dim")
    if head(po) == "lane":
        px, n = L.lane_px(frames_of(vpath, [0, 10, 20, 30]))
        return (None, "no lane lines") if not px else (pr["value_si"] / px, f"lane lines spacing {px:.1f} px ({n} frames)")
    if head(po) == "ruler":
        return None, "ruler calibre: tick spacing not measured"
    T = qtrack(split, gid) if gid else None
    pp = qobj(T, "prior") if T else []
    if not pp:
        return None, "prior object not found by the per-question tracker (no prior pixels)"
    IT = itrack(split, p["video_id"])
    m, why = pick_inst(IT, po)
    if m is not None:
        S = L.ser_inst(IT, m); src = f"instance {why}"
    else:
        S = pp[0][1]; src = f"per-question track '{pp[0][0]}'"
    e, n = L.ext_px(S, dim)
    if not e or e < 5 or n < 3:
        return None, f"prior extent {e} px on {n} frames"
    return pr["value_si"] / e, f"{src}: {dim or 'size'} {e:.1f} px on {n} frames = {pr['value_si']:g} m"


def moving_hand(IT):
    ms = L.instances(IT, "hand") if IT is not None and "hand" in IT["phrases"] else []
    if not ms:
        return None
    sp = [(L.path_speed(L.ser_inst(IT, m), float(IT["fps"])) or 0.0, m) for m in ms]
    return max(sp, key=lambda z: z[0])[1]


def measure(split, p):
    q = p["q"]; v = p["video_id"]; qs = str(p["question"]).lower()
    rg, V = groups(split); gid = rg.get(int(p["id"]))
    vpath = (V.get(v) or {}).get("path")
    fps = float(p.get("fps") or (V.get(v) or {}).get("fps") or 24.0)
    unit = str(q.get("unit") or "").lower()
    f = USI.get(unit)
    if f is None:
        return dict(g=None, why=f"unit {unit}")
    if q.get("kind") == "size" and PART.search(qs):
        return dict(g=None, why="body-part size: no part measurement")
    mpp, rwhy = ruler(split, p, gid, vpath)
    if mpp is None:
        return dict(g=None, why="no per-question ruler: " + rwhy)
    IT = itrack(split, v); T = qtrack(split, gid) if gid else None
    tg = qobj(T, "target") if T else []
    S, twhy = None, ""
    if re.search(r"\bfingertips?\b", qs):
        m = moving_hand(IT)
        if m is not None:
            S, twhy = L.tip_series(IT, m), "fingertip of the most-moving 'hand' instance"
    elif re.search(r"\bhands?\b", qs) and IT is not None and "hand" in IT["phrases"]:
        m = moving_hand(IT)
        if m is not None:
            S, twhy = L.ser_inst(IT, m), "most-moving 'hand' instance"
    elif re.search(r"\bmiddle\b", qs) and IT is not None:
        m, w = pick_inst(IT, q.get("target") or "")
        if m is not None:
            S, twhy = L.ser_inst(IT, m), "instance " + w
    kind = q.get("kind")
    is_vel = unit.endswith("/s") and not unit.endswith("/s^2")
    if re.search(r"\bpillars?\b", qs) and q.get("kind") == "dist2" and not q.get("target2"):
        if IT is None or "pillar" not in IT["phrases"]:
            return dict(g=None, why="no pillar instances")
        pil = L.instances(IT, "pillar")
        if len(pil) < 3:
            return dict(g=None, why="fewer than 3 pillar instances")
        pb = np.array([m["st"][0]["box"] for m in pil], float)
        y0, y1, pw = np.median(pb[:, 1]), np.median(pb[:, 3]), np.median(pb[:, 2] - pb[:, 0])
        lo = 1.5 * pw
        if re.search(r"\bsecond\b|\bupper\b|\bfirst floor\b", qs):
            ab = np.array([m["st"][0]["box"] for m in L.instances(IT, "arch")] if "arch" in IT["phrases"] else [], float).reshape(-1, 4)
            ab = ab[ab[:, 3] < y0]
            if len(ab) < 2:
                return dict(g=None, why="fewer than 2 arch instances above the pillars")
            y0, y1, lo = np.median(ab[:, 1]), np.median(ab[:, 3]), 0.5 * np.median(ab[:, 2] - ab[:, 0])
            band = "second-floor arch band"
        elif re.search(r"\bground\b", qs):
            band = "ground pillar band"
        else:
            return dict(g=None, why="pillar floor not named")
        full = vpath.replace("/test_480p/", "/test/") if "/test_480p/" in vpath else vpath
        c = cv2.VideoCapture(full); ok, fr = c.read()
        if not ok:
            return dict(g=None, why="no frame")
        s = fr.shape[1] / float(IT["W"])
        per, ac = L.period_px(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), y0 * s, y1 * s, lo * s, IT["W"] / 3.0 * s)
        px = per / s
        return dict(g=px * mpp / f, kind="dist", why=f"{band} y {y0:.0f}-{y1:.0f}: period {px:.2f} px (ac {ac:.2f}) x {mpp:.5f} m/px; ruler {rwhy}")
    if S is None and tg:
        S, twhy = tg[0][1], f"per-question target '{tg[0][0]}'"
    if kind == "dist2" and q.get("target2"):
        dd = qobj(T, "dist") if T else []
        if len(dd) < 2:
            return dict(g=None, why="dist objects missing")
        px = L.dist_at(dd[0][1], dd[1][1], q.get("when"))
        what, kk = f"centre distance at {q.get('when') or 'start'}", "dist"
    elif S is None:
        return dict(g=None, why="no target track")
    elif kind == "accel" and q.get("t") is not None:
        px, what, kk = L.accel_at(S, q["t"], fps), f"accel at {q['t']} s", "motion"
    elif is_vel and q.get("t") is not None:
        px, what, kk = L.speed_c(S, q["t"], fps), f"speed at {q['t']} s (central +-{max(1, int(round(0.1 * fps)))} fr)", "motion"
    elif is_vel and q.get("t1") is not None and q.get("t2") is not None:
        px, what, kk = L.avg_vel(S, q["t1"], q["t2"], fps), f"avg velocity {q['t1']}-{q['t2']} s", "motion"
    elif is_vel:
        px, what, kk = L.path_speed(S, fps), "path speed over the track", "motion"
    elif kind == "disp":
        px, what, kk = L.disp(S), "displacement first->last frame", "motion"
    elif kind == "size":
        px, n = L.ext_px(S, q.get("dim"))
        what, kk = f"{q.get('dim')} extent", "size"
    else:
        return dict(g=None, why=f"kind {kind} not measured")
    if px is None:
        return dict(g=None, why=f"{what}: no value")
    return dict(g=px * mpp / f, kind=kk, why=f"{twhy}: {what} {px:.2f} px(/s) x {mpp:.5f} m/px @ {fps:g} fps; ruler {rwhy}")
