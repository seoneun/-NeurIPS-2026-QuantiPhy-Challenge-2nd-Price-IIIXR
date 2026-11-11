#!/usr/bin/env python
"""rul434: planar (S2/D2) size questions about a periodic scene element (floor / pool / wall tiles) that have no geo
answer (source caw_nogeo, meas_why scene_target) are measured from the image itself: the tile pitch in pixels is the
dominant period of the image-gradient profile inside the scene box (Sobel |d/dx| summed over rows for the horizontal
side, |d/dy| summed over columns for the vertical side), and pixels are converted to metres with the prior's px-per-metre
scale (speed prior: the prior object's VLM-track centre displacement per second / the speed; size prior: the prior
object's box side along the prior's dimension / the size).  A running-bond layout (alternate tile rows offset) shows the
profile a fraction of the pitch, so the pitch is also measured per tile row (bands between the perpendicular grout
lines): when the band pitch is k >= 2 times the global one, the profiles of every k-th band are summed and the full
pitch is read there.  Five interior frames must agree.  No ids, no labels in the logic.
usage: python rul434.py [write]     debug images: cache/audit/rul434/row_<id0>.png"""
import os, sys, json
import numpy as np, pandas as pd, cv2
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul434"; WRITE = "write" in sys.argv[1:]
HORIZ = {"width", "length", "diameter"}
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}
PMIN = 20            # shortest period searched (px)
MIN_AC = 0.3         # autocorrelation score a band must reach to count
NFRAMES = 5          # interior frames measured; their pitches must agree
TOL = 0.12           # relative tolerance for frame agreement and band/global ratio
DBG = f"{R}/cache/audit/{TAG}"


def obj_of(tr, phrase):
    ph = str(phrase or "").strip().lower()
    for qk in ("queries", "queries_d"):
        for q in tr.get(qk) or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("obj") is not None:
                return q["obj"]
    return None


def states(tr, obj):
    return [(k, s["st"]) for k, s in enumerate(tr["objs"].get(str(obj)) or []) if s and s.get("st")]


def read_frames(vid, sp, idxs):
    cap = cv2.VideoCapture(f"{R}/data/{sp}_480p/{vid}.mp4"); out = {}; k = 0
    while True:
        ok, f = cap.read()
        if not ok or k > max(idxs):
            break
        if k in idxs:
            out[k] = f
        k += 1
    return out


def profile(gray, box, axis):
    """axis 0: column profile of |d/dx| (vertical grout lines); axis 1: row profile of |d/dy|. returns (profile, offset)"""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    g = gray[y1:y2, x1:x2].astype(np.float32)
    if axis == 0:
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0), x1
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1), y1


def period(p):
    """dominant period of the profile's (unbiased) autocorrelation between PMIN and half the length: (period, score)"""
    n = len(p)
    if n < 2 * PMIN + 3:
        return None, 0.0
    q = p - p.mean()
    ac = np.correlate(q, q, "full")[n - 1:] / (n - np.arange(n))
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]; seg = ac[PMIN:n // 2 + 1]
    loc = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not loc:
        return None, 0.0
    b = max(loc, key=lambda i: seg[i]); y0, y1, y2 = seg[b - 1], seg[b], seg[b + 1]
    d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2)) if (y0 - 2 * y1 + y2) != 0 else 0.0
    return float(PMIN + b + d), float(y1)


def peaks(p, pitch, off):
    """positions of the grout lines: local maxima of the smoothed profile, at least 0.6 pitch apart, above mean + 0.5 sd"""
    k = max(3, int(pitch * 0.1) | 1)
    ps = cv2.GaussianBlur(p.reshape(1, -1), (k, 1), 0).ravel()
    thr = ps.mean() + 0.5 * ps.std()
    idx = [i for i in range(1, len(ps) - 1) if ps[i] >= ps[i - 1] and ps[i] > ps[i + 1] and ps[i] > thr]
    idx.sort(key=lambda i: -ps[i]); keep = []
    for i in idx:
        if all(abs(i - j) >= 0.6 * pitch for j in keep):
            keep.append(i)
    return sorted(int(i + off) for i in keep)


def on_lattice(pk, pitch, tol=0.1):
    """how many of the peak positions sit on a lattice of the given pitch (common phase = circular mean): (inliers, n)"""
    if len(pk) < 2:
        return 0, len(pk)
    x = np.array(pk, float); ph = np.angle(np.mean(np.exp(2j * np.pi * x / pitch))) / (2 * np.pi) * pitch
    res = np.abs(((x - ph + pitch / 2) % pitch) - pitch / 2) / pitch
    return int((res <= tol).sum()), len(pk)


def measure(gray, box, axis):
    """tile pitch along axis inside box. returns dict(pitch, bond, global, perp, bands=[(lo,hi,pitch,score,peaks)], perp_peaks)"""
    p, off = profile(gray, box, axis); pg, sg = period(p)
    if pg is None or sg < MIN_AC:
        return None
    gpk = peaks(p, pg, off); ni, npk = on_lattice(gpk, pg)
    if ni < 2 or ni < 0.6 * npk:           # the gradient peaks are not a regular lattice (oblique view, clutter)
        return None
    pp, offp = profile(gray, box, 1 - axis); pv, sv = period(pp)
    lo, hi = (box[1], box[3]) if axis == 0 else (box[0], box[2])
    bands = []; perp_pk = []
    if pv is not None and sv >= MIN_AC:
        perp_pk = [v for v in peaks(pp, pv, offp) if lo < v < hi]
        cuts = [lo] + perp_pk + [hi]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if c1 - c0 < 0.3 * pv:
                continue
            bb = [box[0], c0, box[2], c1] if axis == 0 else [c0, box[1], c1, box[3]]
            pb, ob = profile(gray, bb, axis); pbp, sb = period(pb)
            bands.append(dict(lo=int(c0), hi=int(c1), pitch=pbp, score=sb, prof=pb, off=ob, peaks=peaks(pb, pbp, ob) if pbp else []))
    good = [b for b in bands if b["pitch"] and b["score"] >= MIN_AC]
    pitch, bond = pg, "grid"
    if len(good) >= 2:
        ratio = float(np.median([b["pitch"] for b in good])) / pg; k = int(round(ratio))
        if k >= 2 and abs(ratio - k) <= TOL * k:
            # running bond: every k-th tile row shares the phase; sum their profiles and read the full pitch there
            est = []
            for r in range(k):
                grp = [b for j, b in enumerate(bands) if j % k == r and b["score"] >= MIN_AC]
                if not grp:
                    continue
                ps = np.zeros(len(p))
                for b in grp:
                    ps += b["prof"]
                pk_, sk_ = period(ps)
                if pk_ is not None and sk_ >= MIN_AC and abs(pk_ / pg - k) <= TOL * k:
                    est.append(pk_)
            pitch = float(np.median(est)) if est else float(np.median([b["pitch"] for b in good]))
            bond = f"running bond 1/{k}"
        elif abs(ratio - 1) > TOL:
            return None          # tile rows and the whole box disagree: no confident period
    return dict(pitch=pitch, bond=bond, glob=pg, score=sg, perp=pv, bands=bands, perp_peaks=perp_pk, peaks=gpk, lattice=(ni, npk))


def scale_px_per_m(tv, po, pr, fps):
    st = states(tv, po)
    if len(st) < 0.8 * len(tv["fidx"]):
        return None, ""
    if pr["kind"] == "speed":
        c = np.array([[(s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2] for _, s in st])
        fi = np.array([tv["fidx"][k] for k, _ in st], float)
        d = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(fi), 1)
        v = float(np.median(d)) * fps
        if v < 0.5 * fps / 24:
            return None, ""
        return v / pr["value_si"], f"prior {pr['obj']} {np.median(d):.2f} px/frame x {fps:g} fps = {v:.1f} px/s / {pr['value_si']:g} m/s"
    pax = 0 if (pr.get("dim") in HORIZ or pr.get("dim") is None) else 1
    e = float(np.median([s["box"][2] - s["box"][0] if pax == 0 else s["box"][3] - s["box"][1] for _, s in st]))
    if e < 10:
        return None, ""
    return e / pr["value_si"], f"prior {pr['obj']} {pr.get('dim') or 'size'} {e:.0f} px / {pr['value_si']:g} m"


def draw(img, box, axis, m, text, fn):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)
    for v in m["perp_peaks"]:
        if axis == 0: cv2.line(img, (x1, v), (x2 - 1, v), (0, 200, 0), 1)
        else: cv2.line(img, (v, y1), (v, y2 - 1), (0, 200, 0), 1)
    bands = [b for b in m["bands"] if b["score"] >= MIN_AC] or [dict(lo=(y1 if axis == 0 else x1), hi=(y2 if axis == 0 else x2), peaks=m["peaks"])]
    for b in bands:
        for v in b["peaks"]:
            if axis == 0: cv2.line(img, (v, b["lo"]), (v, b["hi"] - 1), (0, 0, 255), 1)
            else: cv2.line(img, (b["lo"], v), (b["hi"] - 1, v), (0, 0, 255), 1)
    cv2.putText(img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(fn), exist_ok=True); cv2.imwrite(fn, img)


def rule(sp):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
    PP = {p["id"]: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    FPS = {k: float(v.get("csv") or v.get("file")) for k, v in json.load(open(f"{R}/geo/out/file_fps_{sp}.json")).items()}
    out = {}
    for i, r in A.iterrows():
        p = PP[i]; q = p["q"]; tgt = str((p.get("target") or {}).get("phrase") or "").lower()
        if not (str(r.src_type) in ("captured", "internet", "simulation") and r["cat"] in ("S2", "D2") and q.get("kind") == "size"
                and str(r.source) == "caw_nogeo" and str(r.meas_why) == "scene_target" and q.get("unit") in UNIT and "tile" in tgt.split() + tgt.replace("tiles", "tile").split()):
            continue
        pr = [x for x in p["priors"] if x.get("kind") in ("size", "speed") and x.get("value_si") and x.get("ok", True)]
        fn = f"{R}/geo/out/tracks_{sp}_v3ntmv/{r.video}.json"
        if len(pr) != 1 or not os.path.exists(fn):
            continue
        pr = pr[0]; tv = json.load(open(fn)); po = obj_of(tv, (p.get("prior_obj") or {}).get("phrase"))
        if po is None:
            continue
        scale, swhy = scale_px_per_m(tv, po, pr, FPS.get(r.video, p.get("fps") or 24.0))
        if not scale:
            continue
        axis = 1 if (q.get("dim") == "height" or q.get("vertical")) else 0
        n = len(tv["fidx"]); ks = sorted({int(round(n * f)) for f in np.linspace(0.2, 0.8, NFRAMES)})
        F = read_frames(r.video, sp, {tv["fidx"][k] for k in ks})
        so = obj_of(tv, tgt); meas = []
        for k in ks:
            f = F.get(tv["fidx"][k])
            if f is None:
                continue
            H, W = f.shape[:2]; box = [0, 0, W, H]      # no track for the scene phrase: the whole frame
            if so is not None and so != po:                # scene track: its box, and only if the tiles fill a good part of the frame
                sb = tv["objs"][str(so)][k]
                b = (sb or {}).get("st", {}).get("box") if sb else None
                if not (b and (b[2] - b[0]) * (b[3] - b[1]) >= 0.25 * W * H):
                    continue
                box = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
            m = measure(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), box, axis)
            if m:
                meas.append((k, f, box, m))
        if len(meas) < 3:
            continue
        pitches = np.array([m["pitch"] for *_, m in meas]); med = float(np.median(pitches))
        if (np.abs(pitches / med - 1) <= TOL).sum() < 3:
            continue
        k, f, box, m = min(meas, key=lambda t: abs(t[3]["pitch"] - med))
        val = med / scale / UNIT[q["unit"]]
        pk = [b["peaks"] for b in m["bands"] if b["score"] >= MIN_AC] if m["bond"] != "grid" else [m["peaks"]]
        why = (f"{'horizontal' if axis == 0 else 'vertical'} pitch {med:.1f} px ({m['bond']}, {len(meas)} frames {pitches.min():.0f}-{pitches.max():.0f}, "
               f"perp {m['perp'] or 0:.0f} px, lattice {m['lattice'][0]}/{m['lattice'][1]}) / {scale:.1f} px/m [{swhy}]; grout at {pk}")
        out[i] = (float(val), why)
        draw(f.copy(), box, axis, m, f"{i} pitch {med:.1f}px scale {scale:.1f}px/m -> {val:.3g} {q['unit']} ({m['bond']}) frame {tv['fidx'][k]}", f"{DBG}/row_{i}.png")
    return out


if __name__ == "__main__":
    res = {}
    for sp in ("test", "val"):
        res[sp] = rule(sp); print(sp, len(res[sp]), "rows")
        A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
        for i, (v, why) in res[sp].items():
            lab = f" | y {A.y[i]:.4g} MRA {A.mra[i]:.1f} -> {float(FC.row_mra(np.array([v]), np.array([A.y[i]]))[0]):.1f}" if sp == "val" else ""
            print(f"  {i} {A.video[i]} {A['cat'][i]} {str(A.question[i])[:55]}: {A.answer[i]:.4g} -> {v:.4g} ({A.source[i]}){lab}  [{why}]")
    if WRITE:
        os.makedirs(f"{R}/cache/tmp", exist_ok=True)
        for sp in res:
            json.dump({str(i): dict(value=round(float(v), 7), why=f"{TAG}: {why}") for i, (v, why) in res[sp].items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print("written cache/tmp/ag_%s_patch_{test,val}.json" % TAG)
