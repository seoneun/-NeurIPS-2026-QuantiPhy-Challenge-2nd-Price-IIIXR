#!/usr/bin/env python
"""rul136: painted-line scene targets ("long side of a marked parking space / box formed by the white lines on the
ground") that the pipeline left as a stalled caw_nogeo value are measured on the video's own prior ruler.

Detector (static painted lines, v22 sruler / lane_ruler style):
  * background = temporal median of the clip (removes the moving robots); checks on 3 mid-clip frames (n/4, n/2, 3n/4).
  * lower half, bright low-saturation pixels (top-hat), HoughLinesP for long near-horizontal lines (>= 30% of the width);
    the two lines bounding the band are a consecutive pair 10..80 px apart.
  * stripes = bright runs inside the band, linked row to row and fitted as x(y) lines; a stripe must cover >= 70% of
    the band rows with <= 2.5 px rms; stripes crossing the band = bay sides.
  * long side in px = spacing of consecutive stripes evaluated at the prior object's ground-contact y (box bottom),
    which must lie within 10% of the frame height of the band. If the band height is larger it is used instead.
Value = px * meas_mpp (the 2d_ratio prior ruler of the same video).
Accept: the 3 frames agree with the background measurement within 5%; if >= 3 stripes, the bays agree within 8%;
 value within [0.5, 2] x CaW raw.
usage: python rule.py   (writes cache/tmp/ag_c73pline_patch_{test,val}.json)"""
import json, os, re, sys
import numpy as np, pandas as pd, cv2
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo")
import track_common as tc
cv2.setNumThreads(2)
SUB = "sub_xt_Nzz20"
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
PAT = re.compile(r'parking (space|spot|bay)|box formed by the white lines|marked .* on the (ground|floor)|painted (box|rectangle)', re.I)
TRK_DIRS = ["_v2sc", "_v2scu", "_v2", "_v2full", "_v3nt", "_v3", ""]
DBG = os.environ.get("C73_DBG")


def bright_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    m = ((th > 45) & (s < 60)).astype(np.uint8)
    m[: g.shape[0] // 2] = 0
    return m


def hlines(m):
    H, W = m.shape
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 360, 60, minLineLength=int(0.3 * W), maxLineGap=25)
    ys = []
    if L is None:
        return []
    for x1, y1, x2, y2 in L.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if min(ang, 180 - ang) < 3:
            ys.append(((y1 + y2) / 2, abs(x2 - x1)))
    ys.sort()
    groups = []
    for y, l in ys:                                    # merge parallel detections of the same painted line
        if groups and y - groups[-1][-1][0] <= 4:
            groups[-1].append((y, l))
        else:
            groups.append([(y, l)])
    return [float(np.average([a for a, b in g], weights=[b for a, b in g])) for g in groups]


def stripes(m, y0, y1):
    """bright runs strictly inside the band (y0, y1), linked into straight stripes x = a*y + b"""
    rows = list(range(int(np.ceil(y0)) + 4, int(np.floor(y1)) - 3))
    if len(rows) < 6:
        return []
    pts = []
    for y in rows:
        r = m[y].astype(np.int8)
        d = np.diff(np.r_[0, r, 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for a, b in zip(st, en):
            if 3 <= b - a <= 45:
                pts.append((y, (a + b - 1) / 2.0))
    tracks = []
    for y, x in sorted(pts):
        best = None
        for t in tracks:
            yl, xl = t[-1]
            if 0 < y - yl <= 4:
                pred = xl if len(t) < 3 else np.polyval(np.polyfit([p[0] for p in t[-6:]], [p[1] for p in t[-6:]], 1), y)
                if abs(pred - x) <= 6 and (best is None or abs(pred - x) < best[0]):
                    best = (abs(pred - x), t)
        if best:
            best[1].append((y, x))
        else:
            tracks.append([(y, x)])
    out = []
    for t in tracks:
        if len(t) < 0.7 * len(rows):
            continue
        yy, xx = np.array(t).T
        a, b = np.polyfit(yy, xx, 1)
        rms = float(np.sqrt(np.mean((np.polyval([a, b], yy) - xx) ** 2)))
        if rms <= 2.5 and abs(a) < 6:
            out.append((a, b))
    out.sort(key=lambda ab: ab[0] * (y0 + y1) / 2 + ab[1])
    return out


def measure(img, ygr):
    """-> dict(spacing px at ygr, band, n stripes, bays) or None"""
    H, W = img.shape[:2]
    m = bright_mask(img)
    ys = hlines(m)
    best = None
    for y0, y1 in zip(ys[:-1], ys[1:]):
        if not (10 <= y1 - y0 <= 80):
            continue
        if not (y0 - 0.1 * H <= ygr <= y1 + 0.1 * H):
            continue
        S = stripes(m, y0, y1)
        if len(S) < 2:
            continue
        xs = [a * ygr + b for a, b in S]
        bays = np.diff(xs)
        bays = bays[bays > 0.05 * W]
        if not len(bays):
            continue
        cand = dict(y0=y0, y1=y1, n=len(S), bays=[float(v) for v in bays], sp=float(np.median(bays)), band=float(y1 - y0),
                    S=[(float(a), float(b)) for a, b in S])
        if best is None or abs((y0 + y1) / 2 - ygr) < abs((best["y0"] + best["y1"]) / 2 - ygr):
            best = cand
    return best


def prior_ground_y(vid, split):
    for d in TRK_DIRS:
        p = f"{R}/geo/out/tracks_{split}{d}/{vid}.json"
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        qs = [q for q in t.get("queries", []) if "prior" in (q.get("roles") or [])]
        if not qs:
            continue
        o = t["objs"][str(qs[0]["obj"])] if isinstance(t["objs"], dict) else t["objs"][qs[0]["obj"]]
        bxf = [(f, s["st"]["box"]) for f, s in zip(t["fidx"], o) if s and s.get("st")]
        bx = np.array([b for f, b in bxf]) if bxf else np.zeros((0, 4))
        allb = {}
        objs = t["objs"].values() if isinstance(t["objs"], dict) else t["objs"]
        for oo in objs:
            for f, s_ in zip(t["fidx"], oo):
                if s_ and s_.get("st"):
                    allb.setdefault(f, []).append(s_["st"]["box"])
        return bx, d, allb, bxf
    return np.zeros((0, 4)), None, {}, []


def uncut_speed_ruler(prior_text, bxf, W, H, fps):
    """speed prior 'speed of X = v m/s' (no time): mpp = v / centre speed fitted on the uncut frames only
    (a box cut at the border slows its centroid; cf. rul117)"""
    m = re.search(r"^(?:average )?speed of .*?=\s*([0-9.]+)\s*m/s\s*$", str(prior_text).strip(), re.I)
    if not m or len(bxf) < 8:
        return None, "not a plain speed prior / short track"
    f = np.array([a for a, b in bxf], float); b = np.array([bb for a, bb in bxf], float)
    ok = (b[:, 0] > 2) & (b[:, 2] < W - 2) & (b[:, 1] > 2) & (b[:, 3] < H - 2)
    if ok.sum() < 8:
        return None, f"only {int(ok.sum())} uncut frames"
    t = f[ok] / fps; cx = (b[ok, 0] + b[ok, 2]) / 2; cy = (b[ok, 1] + b[ok, 3]) / 2
    vx = np.polyfit(t, cx, 1)[0]; vy = np.polyfit(t, cy, 1)[0]
    v = float(np.hypot(vx, vy))
    res = np.hypot(cx - np.polyval(np.polyfit(t, cx, 1), t), cy - np.polyval(np.polyfit(t, cy, 1), t))
    span = v * (t.max() - t.min())
    if v <= 0 or span < 50 or np.sqrt(np.mean(res ** 2)) > 0.05 * span:
        return None, f"uncut motion not linear/short (v {v:.1f}px/s span {span:.0f})"
    return float(m.group(1)) / v, f"uncut v {v:.1f}px/s n {int(ok.sum())}"


def ruler(raw, vid):
    r = raw[(raw.video_id == vid) & (raw.meas_mode == "2d_ratio")]
    mpp = pd.to_numeric(r.meas_mpp, errors="coerce").dropna()
    mpp = mpp[mpp > 0]
    return float(mpp.median()) if len(mpp) else None


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        P[int(j["id"])] = j
    patch, log = {}, []
    for i, r in det.iterrows():
        q = str(raw.loc[i, "question"]) if i in raw.index else ""
        if not PAT.search(q):
            continue
        vid = r.video_id
        kind = (P.get(int(i), {}).get("q") or {}).get("kind")
        why = None
        if not (str(vid).startswith("simulation") and r["cat"] in ("S2", "D2")):
            why = "not sim 2D"
        elif kind not in ("size", "other"):
            why = f"kind {kind}"
        elif r.answer_source != "caw_nogeo":
            why = f"source {r.answer_source}"
        mpp = ruler(raw, vid) if why is None else None
        if why is None and not mpp:
            why = "no 2d_ratio ruler"
        res = dict(id=int(i), split=split, video=vid, cat=r["cat"], old=float(r.answer), cs=float(r.cs) if pd.notna(r.cs) else None, q=q[:90])
        if why is None:
            bx, tdir, allb, bxf = prior_ground_y(vid, split)
            fr = tc.load_video(raw.loc[i, "video_path"].replace("/test_480p/", f"/{split}_480p/"))
            H, W = fr[0].shape[:2]
            mpp_geo = mpp
            mpp_u, uwhy = uncut_speed_ruler(raw.loc[i, "ground_truth_prior"], bxf, W, H, float(raw.loc[i, "fps"]))
            cutfrac = 1 - float(np.mean((bx[:, 0] > 2) & (bx[:, 2] < W - 2) & (bx[:, 1] > 2) & (bx[:, 3] < H - 2))) if len(bx) else 1.0
            res.update(mpp_geo=mpp_geo, mpp_uncut=mpp_u, uncut_why=uwhy, prior_cut_frac=round(cutfrac, 2))
            if cutfrac > 0.2:                # geo's ruler came from a border-cut prior track (slow cut centroid)
                if mpp_u:
                    mpp = mpp_u              # speed prior: uncut-frame ruler (same correction as the accepted rul439 patch)
                else:
                    why = f"prior track cut in {cutfrac:.0%} of frames and no uncut ruler ({uwhy})"
            unc = bx[(bx[:, 0] > 2) & (bx[:, 2] < W - 2) & (bx[:, 1] > 2) & (bx[:, 3] < H - 2)] if len(bx) else bx
            if why is not None:
                pass
            elif len(unc) < 5:
                why = f"prior track cut/short ({len(unc)} uncut, {tdir})"
            else:
                ygr = float(np.median(unc[:, 3]))
                bg = np.median(np.stack(fr[:: max(1, len(fr) // 40)]), 0).astype(np.uint8)
                mb = measure(bg, ygr)
                n = len(fr)
                res.update(ygr=round(ygr, 1), tdir=tdir, mpp_used=mpp, bg=None if mb is None else {k: v for k, v in mb.items() if k != "S"})
                if mb is None:
                    why = "no painted band/stripes"
                else:
                    sp = mb["sp"]
                    ym = (mb["y0"] + mb["y1"]) / 2
                    xb = [a * ym + b for a, b in mb["S"]]
                    conf = [0] * len(xb); fv = []; extra = 0
                    fk = sorted(allb)
                    for f in (n // 4, n // 2, 3 * n // 4):
                        boxes = allb.get(min(fk, key=lambda k: abs(k - f)), []) if fk else []
                        occ = lambda x: any(b_[0] - 5 <= x <= b_[2] + 5 and b_[1] <= mb["y1"] and b_[3] >= mb["y0"] for b_ in boxes)
                        Sf = stripes(bright_mask(fr[f]), mb["y0"], mb["y1"])      # same band, this frame
                        Sf = [ab for ab in Sf if not occ(ab[0] * ym + ab[1])]
                        xf = [a * ym + b for a, b in Sf]
                        hit = [any(abs(x - xs) <= 0.05 * sp for x in xf) for xs in xb]
                        conf = [c + h for c, h in zip(conf, hit)]
                        ex = [round(x) for x in xf if not any(abs(x - xs) <= 0.05 * sp for xs in xb)]
                        ex += [round(xs) for xs, h in zip(xb, hit) if not h and not occ(xs)]   # bg stripe missing though unoccluded
                        extra += len(ex)
                        if all(hit) and len(xb) >= 2:
                            mt = [Sf[int(np.argmin([abs(x - xs) for x in xf]))] for xs in xb]
                            fv.append(float(np.median(np.diff([a * ygr + b for a, b in mt]))))
                        if DBG:
                            print("   frame", f, "stripes@ym", [round(x) for x in xf], "bg", [round(x) for x in xb], "extra", ex,
                                  "boxes", [[round(v) for v in b_] for b_ in boxes])
                    res.update(frames_conf=conf, frames_sp=[round(v, 1) for v in fv], extra=extra)
                    side = "long" if re.search(r"\blong", q, re.I) else ("short" if re.search(r"\bshort", q, re.I) else None)
                    if side != "long":
                        why = f"side {side}"
                    elif min(conf) < 2 or extra > 0 or not fv or max(abs(v / sp - 1) for v in fv) > 0.05:
                        why = f"frames disagree conf {conf} extra {extra} sp {fv} vs {sp:.1f}"
                    elif len(mb["bays"]) >= 2 and max(mb["bays"]) / min(mb["bays"]) > 1.08:
                        why = f"irregular bays {mb['bays']}"
                    else:
                        px = max(sp, mb["band"])
                        val = px * mpp
                        res["val"] = round(val, 3)
                        if res["cs"] and not (0.5 * res["cs"] <= val <= 2 * res["cs"]):
                            why = f"outside [0.5,2] x CaW {res['cs']}"
                        else:
                            patch[str(int(i))] = dict(value=round(float(val), 4),
                                                      why=f"rul136: painted bay {px:.0f}px at ground y {ygr:.0f} x mpp {mpp:.5f} ({mb['n']} stripes)")
        res["why"] = why or "FIRED"
        log.append(res)
        print(json.dumps(res, default=str))
    json.dump(patch, open(f"{R}/cache/tmp/ag_c73pline_patch_{split}.json", "w"), indent=1)
    print(split, "patched", len(patch))
    return patch


if __name__ == "__main__":
    for sp in ("test", "val"):
        run(sp)
