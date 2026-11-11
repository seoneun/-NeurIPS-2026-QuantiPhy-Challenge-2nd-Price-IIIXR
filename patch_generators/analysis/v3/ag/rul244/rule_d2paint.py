"""rul244: width of a painted simulated road in a 2D (planar-motion, speed-prior) video, measured between the
road's painted/curb lines along or across the prior mover's own path, scaled by the video's own 2d_ratio metres-per-pixel.

Rows: simulation, video_type[1]=='2', parse kind size, dim width, target phrase contains 'road'/'street', prior kind speed with
an object that has a track; not a _segmented copy.
Scale: median meas_mpp of the same video's 2d_ratio rows in the geo raw (the pipeline's own prior-derived scale at the mover).
Lines: background = per-pixel median of 9 frames (movers removed); paint mask = white top-hat (th>30, S<50, V>110) or
yellow (15<=H<=35, S>=80, V>=100); HoughP segments >= 100 px.
Path: mover track; ground point = box bottom-centre (perspective) or box centre (ortho); path line by least squares.
View: 'ortho' if every long segment (>=150 px) that is not parallel to the path (>5 deg) is within 3 deg of every other such
segment (no vanishing point); else 'persp'.
Mode 'cross' (question says cross/crossing/crossed): segments at >= 30 deg (ortho) or >= 15 deg (persp) to the path, extended, intersect the path line
inside the mover's travelled extent; crossings clustered (gap <= 25 px); need >= 3 clusters (two edges + a centre line);
width_px = distance between the outermost cluster means.  Gates: ortho -> the outer clusters' lines within 10 deg of
perpendicular to the path; persp -> path within 5 deg of horizontal, ground-row std <= 4 px (constant depth) and the
vanishing point of the two outer lines within 10% of the image width of the image centre column (road along the view axis,
so the horizontal path crosses it at right angles).
Mode 'on' (question says 'are on'/'is on'/'drives on'/'rides on'): ortho only; segments parallel to the path (<= 3 deg),
signed offsets from the path line; nearest line cluster on each side with |offset| >= 10 px (the mover's own centre line
excluded by needing both sides); the outer edge = nearest WHITE cluster beyond any yellow centre line on that side;
width_px = sum of both offsets.
Output width = width_px * mpp.  Fire if |log(W/answer)| > log(1.25).  Inputs only; labels only evaluate.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul244/rule_d2paint.py [build]"""
import sys, os, json, re
import numpy as np, pandas as pd, cv2
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = os.environ.get("D2P_TAG", "rul244")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz39"
VERB = bool(os.environ.get("D2P_VERB"))


def bg(path, n=9):
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for f in np.linspace(0, N - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        if ok:
            fr.append(im)
    return np.median(np.stack(fr), 0).astype(np.uint8)


def paint_segments(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    H, S, V = [hsv[..., i].astype(int) for i in range(3)]
    masks = {"w": ((th > 30) & (S < 50) & (V > 110)), "y": ((H >= 15) & (H <= 35) & (S >= 80) & (V >= 100) & (th > 15))}
    out = []
    for col, m in masks.items():
        L = cv2.HoughLinesP(m.astype(np.uint8) * 255, 1, np.pi / 360, 40, minLineLength=100, maxLineGap=25)
        if L is None:
            continue
        for x1, y1, x2, y2 in L.reshape(-1, 4).astype(float):
            out.append((col, np.array([x1, y1]), np.array([x2, y2])))
    return out


def ang_between(d1, d2):
    return float(np.degrees(np.arccos(min(1.0, abs(d1 @ d2)))))


def unit(p, q):
    d = q - p; return d / max(np.linalg.norm(d), 1e-9)


def intersect(p, d, q, e):
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _ = np.linalg.solve(A, q - p)
    return p + t * d


def cluster(vals, gap=25.0):
    vals = sorted(vals, key=lambda v: v[0]); cl = []
    for v in vals:
        if cl and v[0] - cl[-1][-1][0] <= gap:
            cl[-1].append(v)
        else:
            cl.append([v])
    return cl


def measure(split, vid, r, qtext, prior_obj):
    tp = f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json"
    if not os.path.exists(tp):
        return np.nan, "no tracks"
    T = json.load(open(tp))
    qmap = {q["key"].split("|")[0]: q.get("obj") for q in T["queries"]}
    o = qmap.get(prior_obj)
    if o is None:
        return np.nan, "no mover track"
    B = np.array([s["st"]["box"] for s in T["objs"][str(o)] if s and s.get("st")], float)
    if len(B) < 10:
        return np.nan, "short track"
    img = bg(str(r.video_path)); Hh, Ww = img.shape[:2]
    sx, sy = Ww / float(T["W"]), Hh / float(T["H"]); B[:, [0, 2]] *= sx; B[:, [1, 3]] *= sy
    segs = paint_segments(img)
    ctr = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    u0 = unit(ctr[0], ctr[-1])
    if np.linalg.norm(ctr[-1] - ctr[0]) < 0.2 * Ww:
        return np.nan, "mover travel too short"
    longs = [(c, unit(p, q)) for c, p, q in segs if np.linalg.norm(q - p) >= 150 and ang_between(unit(p, q), u0) > 5]
    ortho = len(longs) >= 2 and all(ang_between(a[1], b[1]) <= 3 for a in longs for b in longs)
    view = "ortho" if ortho else "persp"
    gp = ctr if ortho else np.stack([(B[:, 0] + B[:, 2]) / 2, B[:, 3]], 1)
    m0 = gp.mean(0); _, _, vt = np.linalg.svd(gp - m0); u = vt[0]
    if u @ u0 < 0:
        u = -u
    nrm = np.array([-u[1], u[0]]); proj = (gp - m0) @ u; lo, hi = proj.min(), proj.max()
    mode = "on" if re.search(r"\b(are|is) on\b|\b(drives|rides|travels|moves) (on|along)\b", qtext, re.I) else (
        "cross" if re.search(r"\bcross", qtext, re.I) else None)
    if mode is None:
        return np.nan, "no mode"
    if mode == "cross":
        cr = []
        for c, p, q in segs:
            d = unit(p, q)
            if ang_between(d, u) < (30 if view == "ortho" else 15):
                continue
            x = intersect(m0, u, p, d)
            if x is None:
                continue
            t = float((x - m0) @ u)
            if lo - 0.05 * (hi - lo) <= t <= hi + 0.05 * (hi - lo):
                cr.append((t, c, p, d))
        cl = cluster(cr)
        if VERB:
            print("  CR", view, [(round(v[0]), v[1], np.round(v[2]).tolist(), round(float(np.degrees(np.arctan2(v[3][1], v[3][0]))) % 180)) for v in sorted(cr, key=lambda z: z[0])], "lo/hi", round(lo), round(hi), "m0", np.round(m0))
        if len(cl) < 3:
            return np.nan, f"{view} cross: {len(cl)} clusters"
        L, Rr = cl[0], cl[-1]
        tl = float(np.mean([v[0] for v in L])); trr = float(np.mean([v[0] for v in Rr]))
        dl = L[int(np.argmax([np.linalg.norm(v[3]) for v in L]))]; dr = Rr[0]
        if view == "ortho":
            if min(90 - ang_between(dl[3], u), 90 - ang_between(dr[3], u)) < 0 or max(90 - ang_between(dl[3], u), 90 - ang_between(dr[3], u)) > 10:
                return np.nan, "ortho cross: edges not perpendicular"
        else:
            if ang_between(u, np.array([1.0, 0])) > 5 or np.std(gp[:, 1]) > 4:
                return np.nan, "persp cross: path not at constant depth"
            vp = intersect(dl[2], dl[3], dr[2], dr[3])
            if vp is None or abs(vp[0] - Ww / 2) > 0.1 * Ww or vp[1] > m0[1]:
                return np.nan, "persp cross: vp %s off centre" % (None if vp is None else np.round(vp))
        wpx = trr - tl
        why = f"{view} cross: clusters {[round(float(np.mean([v[0] for v in c])), 0) for c in cl]} wpx {wpx:.0f}"
    else:
        if view != "ortho":
            return np.nan, "on-mode needs ortho view"
        par = []
        for c, p, q in segs:
            d = unit(p, q)
            if ang_between(d, u) > 3:
                continue
            off = float(((p + q) / 2 - m0) @ nrm)
            par.append((off, c))
        side = {}
        for sg in (1, -1):
            cl = cluster([(abs(o), c) for o, c in par if o * sg >= 10])
            whites = [c for c in cl if all(v[1] == "w" for v in c)]
            if not whites:
                return np.nan, "on: no white edge on side %d" % sg
            side[sg] = float(np.mean([v[0] for v in whites[0] if v[1] == "w"]))
        wpx = side[1] + side[-1]
        why = f"ortho on: edges {side[1]:.0f}/{side[-1]:.0f} px"
    return wpx, why


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    PQ = {int(j["id"]): j for j in map(json.loads, open(f"{R}/geo/out/parse_{split}_v2.jsonl"))}
    out, log = {}, []
    for i in det.index:
        r = raw.loc[i]; vid = str(r.video_id)
        if not vid.startswith("simulation") or vid.endswith("_segmented") or str(r.video_type)[1:2] != "2":
            continue
        J = PQ.get(int(i), {}); q = J.get("q") or {}
        if q.get("kind") != "size" or q.get("dim") != "width" or not re.search(r"\b(road|street)\b", str(q.get("target") or "")):
            continue
        pri = [p for p in (J.get("priors") or []) if p.get("kind") == "speed" and p.get("obj")]
        if not pri:
            continue
        vr = raw[(raw.video_id == vid) & (raw.meas_mode == "2d_ratio")].meas_mpp.dropna()
        if len(vr) == 0:
            log.append((split, int(i), vid, None, "no 2d_ratio mpp")); continue
        mpp = float(vr.median())
        if vr.max() / vr.min() > 1.05:
            log.append((split, int(i), vid, None, "mpp disagree")); continue
        wpx, why = measure(split, vid, r, str(r.question), str(pri[0]["obj"]))
        fu = {"cm": 100.0, "mm": 1000.0}.get(str(q.get("unit") or "m").lower(), 1.0)
        a = float(det.answer[i])
        w = wpx * mpp if np.isfinite(wpx) else np.nan
        # pipeline fusion: measurement acts as geo (2d_ratio scale = strong mode); CaW cs in the question's unit
        cs = float(det.cs[i]) if "cs" in det.columns and pd.notna(det.cs[i]) else np.nan
        gv = w * fu if np.isfinite(w) else np.nan
        if np.isfinite(gv) and np.isfinite(cs) and cs > 0 and max(gv, cs) / min(gv, cs) <= 2.0:
            fv, fz = float(np.sqrt(gv * cs)), "agree_gm"
        else:
            fv, fz = gv, "geo_strong"
        fire = bool(np.isfinite(w) and w > 2.0 and abs(np.log(fv / a)) > np.log(1.25))
        log.append((split, int(i), vid, round(a, 3), None if not np.isfinite(w) else round(w, 3), cs, None if not np.isfinite(fv) else round(fv, 3), fz, fire, f"mpp {mpp:.4f} {why}"))
        if fire:
            out[str(int(i))] = {"value": fv, "why": f"{TAG}: sim 2D road width geo {w:.2f} m = {wpx:.0f} px x own 2d_ratio mpp {mpp:.4f} ({why}); {fz} with CaW {cs}"}
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    return out, log


if __name__ == "__main__":
    allo = {}
    for sp in ("test", "val"):
        o, lg = run(sp); allo[sp] = o
        for x in lg:
            print("ROW", x)
    sys.path.insert(0, R + "/cache/tmp")
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    for sp in ("val", "test"):
        l = L[(L.split == sp) & L.index.isin([int(k) for k in allo[sp]])]
        for i, rr in l.iterrows():
            nv = allo[sp][str(i)]["value"]
            print("LAB", sp, i, rr.video_id, "y", rr.y, "old", round(rr.answer, 3), A.row_mra(np.array([rr.answer]), np.array([rr.y]))[0],
                  "new", round(nv, 3), A.row_mra(np.array([nv]), np.array([rr.y]))[0])
    print("changed test", len(allo["test"]), "val", len(allo["val"]))
