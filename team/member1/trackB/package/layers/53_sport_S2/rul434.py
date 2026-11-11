#!/usr/bin/env python
import os, sys, json
import numpy as np, pandas as pd, cv2
TAG = "rul434"
HORIZ = {"width", "length", "diameter"}
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}
PMIN = 20
MIN_AC = 0.3
NFRAMES = 5
TOL = 0.12


def obj_of(tr, phrase):
    ph = str(phrase or "").strip().lower()
    for qk in ("queries", "queries_d"):
        for q in tr.get(qk) or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("obj") is not None:
                return q["obj"]
    return None


def states(tr, obj):
    return [(k, s["st"]) for k, s in enumerate(tr["objs"].get(str(obj)) or []) if s and s.get("st")]


def read_frames(vdir, vid, idxs):
    cap = cv2.VideoCapture(f"{vdir}/{vid}.mp4"); out = {}; k = 0
    while True:
        ok, f = cap.read()
        if not ok or k > max(idxs):
            break
        if k in idxs:
            out[k] = f
        k += 1
    return out


def profile(gray, box, axis):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    g = gray[y1:y2, x1:x2].astype(np.float32)
    if axis == 0:
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0), x1
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1), y1


def period(p):
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
    if len(pk) < 2:
        return 0, len(pk)
    x = np.array(pk, float); ph = np.angle(np.mean(np.exp(2j * np.pi * x / pitch))) / (2 * np.pi) * pitch
    res = np.abs(((x - ph + pitch / 2) % pitch) - pitch / 2) / pitch
    return int((res <= tol).sum()), len(pk)


def measure(gray, box, axis):
    p, off = profile(gray, box, axis); pg, sg = period(p)
    if pg is None or sg < MIN_AC:
        return None
    gpk = peaks(p, pg, off); ni, npk = on_lattice(gpk, pg)
    if ni < 2 or ni < 0.6 * npk:
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
            return None
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


def rule(audit, parse, fpsf, tdir, vdir):
    A = pd.read_csv(audit).set_index("id0")
    PP = {p["id"]: p for p in map(json.loads, open(parse))}
    FPS = {k: float(v.get("csv") or v.get("file")) for k, v in json.load(open(fpsf)).items()}
    out = {}
    for i, r in A.iterrows():
        p = PP[i]; q = p["q"]; tgt = str((p.get("target") or {}).get("phrase") or "").lower()
        if not (str(r.src_type) in ("captured", "internet", "simulation") and r["cat"] in ("S2", "D2") and q.get("kind") == "size"
                and str(r.source) == "caw_nogeo" and str(r.meas_why) == "scene_target" and q.get("unit") in UNIT and "tile" in tgt.split() + tgt.replace("tiles", "tile").split()):
            continue
        pr = [x for x in p["priors"] if x.get("kind") in ("size", "speed") and x.get("value_si") and x.get("ok", True)]
        fn = f"{tdir}/{r.video}.json"
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
        F = read_frames(vdir, r.video, {tv["fidx"][k] for k in ks})
        so = obj_of(tv, tgt); meas = []
        for k in ks:
            f = F.get(tv["fidx"][k])
            if f is None:
                continue
            H, W = f.shape[:2]; box = [0, 0, W, H]
            if so is not None and so != po:
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
        m = min(meas, key=lambda t: abs(t[3]["pitch"] - med))[3]
        val = med / scale / UNIT[q["unit"]]
        pk = [b["peaks"] for b in m["bands"] if b["score"] >= MIN_AC] if m["bond"] != "grid" else [m["peaks"]]
        why = (f"{'horizontal' if axis == 0 else 'vertical'} pitch {med:.1f} px ({m['bond']}, {len(meas)} frames {pitches.min():.0f}-{pitches.max():.0f}, "
               f"perp {m['perp'] or 0:.0f} px, lattice {m['lattice'][0]}/{m['lattice'][1]}) / {scale:.1f} px/m [{swhy}]; grout at {pk}")
        out[i] = (float(val), why)
    return out


if __name__ == "__main__":
    res = rule(*sys.argv[1:6])
    with open(sys.argv[6], "w") as f:
        json.dump({str(i): dict(value=round(float(v), 7), why=f"{TAG}: {why}") for i, (v, why) in res.items()}, f, indent=1)
