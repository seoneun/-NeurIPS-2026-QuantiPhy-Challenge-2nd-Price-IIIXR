#!/usr/bin/env python
"""Rule 7 (tag gap): "spacing / headway / gap" questions about ONE class ("What is the average vehicle spacing in meters?",
test row 2074) ask for the distance between ADJACENT INSTANCES of the class, not for one object's size. The parse finds no
target pair (kind dist2, ok=False), so the row falls back to CaW (caw_nogeo, 10.5 m on 2074).

Method (label-free: frames + scene ruler + prior text only):
  * trigger: question matches spacing|headway|gap(s)|clearance|"distance between adjacent/consecutive/successive ..." with a
    length unit, no named second target (target2), not already patched
  * scale: only videos with a painted-lane scene ruler (geo/out/scene_ruler_<split>.json kind=lane, the top-down drone shots
    whose prior is "lane width = X m"); the ruler gives the road direction theta, the lane-line positions and the lane width
    in px. The same ruler already answers the video's sibling rows (submit_v3 --sruler_geo any)
  * vehicles = moving foreground inside the road band: frames registered to frame 0 (phase correlation), temporal median
    background, |frame - bg| > thr, opening/closing, connected components above a minimum area that are vehicle-shaped
    (0.35-3 lane widths along the road, <= 1.2 across); components touching the frame border are dropped (cut boxes)
  * every component is assigned to the lane holding its centroid; within a lane and frame the vehicles are sorted along the
    road; the ADJACENT centre-to-centre distances are the spacings (traffic-flow "spacing"/"headway" = front-to-front
    distance = centre distance for one vehicle length). For "gap"/"clearance" the mean vehicle length is subtracted
    (bumper-to-bumper)
  * row value = median, over the frames with the most adjacent pairs, of the frame's mean spacing (the "average" over the
    vehicles in view) x prior / lane px
  * robustness gate: the value is recomputed on a 5x3 grid of (thr 20-50, amin 0.04-0.09); the row is written only if max/min <= 1.15
  * FINAL answer: a scene-ruler geo row; the pipeline answers these with geo alone (submit_v3 --sruler_geo any). The plain
    fusion gives the same here (CaW ratio > 2 and 2d_ratio is a strong mode -> geo); the code applies the plain fusion rule
    explicitly as a second check and reports it
Outputs cache/tmp/ag_gap_patch_{test,val}.json ({row id: {value, why}}) and debug overlays in cache/tmp/ag_gap_dbg/.
usage: tools/run_limited.sh --io python analysis/v3/ag/gap/gap.py [--debug 1]"""
import os, sys, re, json, math, argparse
import numpy as np, pandas as pd, cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="sub_xt_Nzh")
ap.add_argument("--thr", type=float, default=30.0, help="foreground threshold (max channel abs diff to the median background)")
ap.add_argument("--amin", type=float, default=0.06, help="minimum component area in lane_px^2")
ap.add_argument("--gate", type=float, default=1.15, help="max/min of the (thr, amin) sweep above which the row is not written")
ap.add_argument("--debug", type=int, default=1)
ap.add_argument("--tag", default="gap")
a = ap.parse_args()
DBG = f"{R}/cache/tmp/ag_gap_dbg"
os.makedirs(DBG, exist_ok=True)

TRIG = re.compile(r"\b(spacing|headway|gaps?|clearance)\b|distance between (?:the )?(?:two )?(?:adjacent|consecutive|successive|"
                  r"neighbou?ring)\b", re.I)
GAPW = re.compile(r"\b(gaps?|clearance)\b", re.I)        # bumper-to-bumper; spacing/headway = front-to-front (centre) distance
UNIT = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 100.0, "mm": 1000.0, "km": 1e-3, "ft": 3.28084, "feet": 3.28084}
SWEEP = [(t, m) for t in (20.0, 25.0, 30.0, 40.0, 50.0) for m in (0.04, 0.06, 0.09)]


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


class Scene:
    """registered 480p frames, road band geometry and the temporal-median background (computed once per video)"""

    def __init__(self, vid, split, sr):
        fr = read_frames(f"{R}/data/{split}_480p/{vid}.mp4")
        self.ok = len(fr) >= 10
        if not self.ok:
            return
        H, W = fr[0].shape[:2]
        s = W / float(sr["W"])                                   # ruler px (its own resolution) -> 480p px
        self.lane = sr["px"] * s
        self.lines = np.sort(np.array(sr["lines"], float) * s)
        th = math.radians(sr["theta"])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        c = -xx * math.sin(th) + yy * math.cos(th)                # across-road coordinate (scene_ruler convention)
        al = xx * math.cos(th) + yy * math.sin(th)                # along-road coordinate
        band = (c >= self.lines[0] - 1) & (c <= self.lines[-1] + 1)
        if band.sum() < 100:
            self.ok = False
            return
        # the drone drifts by a pixel or two: register every frame to frame 0 (translation, phase correlation on the whole
        # frame, which is mostly static ground), otherwise the painted lines become foreground and swallow the vehicles
        g0 = cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = cv2.createHanningWindow((W, H), cv2.CV_32F)
        sh = []
        for k in range(len(fr)):
            gk = cv2.cvtColor(fr[k], cv2.COLOR_BGR2GRAY).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(g0, gk, win)
            sh.append((dx, dy))
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                fr[k] = cv2.warpAffine(fr[k], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.shifts = np.array(sh)
        ys, xs = np.where(band)
        self.box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        y0, y1, x0, x1 = self.box
        stack = np.stack([f[y0:y1, x0:x1] for f in fr[::2]], 0)
        self.bg = np.median(stack, axis=0).astype(np.int16)
        del stack
        # painted lines (bright thin structures of the background) and their 1-px rims never count as foreground
        gb = cv2.cvtColor(self.bg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        th_ = cv2.morphologyEx(gb, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        paint = cv2.dilate((th_ > 25).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.bnd = band[y0:y1, x0:x1] & ~paint
        self.cc, self.aa = c[y0:y1, x0:x1], al[y0:y1, x0:x1]
        self.fr, self.H, self.W, self.th = fr, H, W, th

    def detect(self, thr, amin_l):
        """per-frame vehicle components: (frame, lane, along-centre, along-length, area, bbox)"""
        y0, y1, x0, x1 = self.box
        lane, lines, W, H = self.lane, self.lines, self.W, self.H
        amin = amin_l * lane * lane
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        per = []
        for k, f in enumerate(self.fr):
            d = np.abs(f[y0:y1, x0:x1].astype(np.int16) - self.bg).max(axis=2)
            m = ((d > thr) & self.bnd).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            comp = []                                             # [lane, al_min, al_max, c_min, c_max, sum_al, area, box, cut]
            for j in range(1, n):
                bx, by, bw, bh, ar = st[j]
                if ar < amin:
                    continue
                gx0, gy0 = bx + x0, by + y0
                cut = gx0 <= 1 or gy0 <= 1 or gx0 + bw >= W - 1 or gy0 + bh >= H - 1
                pm = lab == j
                av, cv = self.aa[pm], self.cc[pm]
                li = int(np.searchsorted(lines, float(cv.mean()))) - 1   # lane between lines[li] and lines[li+1]
                if li < 0 or li >= len(lines) - 1:
                    continue
                comp.append([li, float(av.min()), float(av.max()), float(cv.min()), float(cv.max()), float(av.sum()), int(ar),
                             [int(gx0), int(gy0), int(gx0 + bw), int(gy0 + bh)], cut])
            # one vehicle can fall apart into fragments (a white car in shadow: bonnet / roof / boot separated by the dark
            # windows): two vehicles in one lane cannot have centres closer than one car length (>= ~1 lane width, 3.5-4.5 m),
            # so same-lane components whose area-weighted centres are < 1 lane width apart are merged (group centre vs next)
            comp.sort(key=lambda x: (x[0], x[5] / x[6]))
            mg = []
            for x in comp:
                if (mg and mg[-1][0] == x[0] and abs(x[5] / x[6] - mg[-1][5] / mg[-1][6]) < 1.0 * lane
                        and max(mg[-1][2], x[2]) - min(mg[-1][1], x[1]) <= 3.0 * lane):
                    y = mg[-1]
                    y[1], y[2], y[3], y[4] = min(y[1], x[1]), max(y[2], x[2]), min(y[3], x[3]), max(y[4], x[4])
                    y[5] += x[5]; y[6] += x[6]; y[8] = y[8] or x[8]
                    y[7] = [min(y[7][0], x[7][0]), min(y[7][1], x[7][1]), max(y[7][2], x[7][2]), max(y[7][3], x[7][3])]
                else:
                    mg.append(list(x))
            for li, a0, a1, c0, c1, sa, ar, bb, cut in mg:
                if cut:
                    continue                                      # cut by the frame border
                la, lc = a1 - a0 + 1, c1 - c0 + 1
                if not (0.35 * lane <= la <= 3.0 * lane and lc <= 1.2 * lane):
                    continue                                      # not vehicle-shaped (residual line streaks, merged blobs)
                per.append((k, li, sa / ar, la, ar, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])))
        return pd.DataFrame(per, columns=["f", "lane", "al", "len", "area", "box"])


def spacing(P):
    """adjacent same-lane centre spacings -> (row value px, details) or (None, reason)"""
    if P.empty:
        return None, {"why": "no vehicles"}
    rows = []
    for (k, li), g in P.groupby(["f", "lane"]):
        al_s = np.sort(g.al.values)
        for q in range(len(al_s) - 1):
            rows.append((k, li, float(al_s[q + 1] - al_s[q])))
    if not rows:
        return None, {"why": "no frame with two vehicles in one lane", "nveh_max": int(P.groupby("f").size().max())}
    S = pd.DataFrame(rows, columns=["f", "lane", "sp"])
    npair = S.groupby("f").size()
    best = npair[npair == npair.max()].index
    fmean = S[S.f.isin(best)].groupby("f").sp.mean()
    f0 = S[S.f == S.f.min()].sp.mean()
    return float(fmean.median()), {"npair_max": int(npair.max()), "nframes_best": int(len(best)),
                                   "frames_best": [int(x) for x in best[:12]],
                                   "fmean_px": [round(float(x), 1) for x in fmean.values[:12]],
                                   "first_frame_mean_px": round(float(f0), 1),
                                   "all_pairs_med_px": round(float(S.sp.median()), 1),
                                   "veh_len_med_px": round(float(P.len.median()), 1),
                                   "veh_len_best_px": round(float(P[P.f.isin(best)].len.median()), 1),
                                   "lanes_used": sorted(int(x) for x in S.lane.unique())}


def measure(vid, split, sr, prior_si, gapq):
    sc = Scene(vid, split, sr)
    if not sc.ok:
        return None, "no scene", {}
    mpp = prior_si / sc.lane
    sweep = {}
    for thr, am in SWEEP:
        sp, inf = spacing(sc.detect(thr, am))
        if sp is not None:
            v = sp - (inf["veh_len_best_px"] if gapq else 0.0)
            sweep[f"{thr:g}/{am:g}"] = round(v * mpp, 3)
    P = sc.detect(a.thr, a.amin)
    sp, inf = spacing(P)
    if sp is None:
        return None, inf["why"], inf
    v_px = sp - (inf["veh_len_best_px"] if gapq else 0.0)
    inf.update({"lane_px": round(sc.lane, 2), "mpp": round(mpp, 5), "sp_px": round(sp, 1), "value_px": round(v_px, 1),
                "definition": "bumper-to-bumper gap (centre spacing - vehicle length)" if gapq else "centre-to-centre spacing",
                "veh_len_med_m": round(inf["veh_len_med_px"] * mpp, 2),
                "spacing_over_len": round(sp / inf["veh_len_best_px"], 2),
                "sweep_m": sweep, "shift_max_px": [round(float(np.abs(sc.shifts[:, 0]).max()), 2),
                                                   round(float(np.abs(sc.shifts[:, 1]).max()), 2)]})
    vs = list(sweep.values())
    inf["sweep_ratio"] = round(max(vs) / min(vs), 3) if vs and min(vs) > 0 else None
    if a.debug:
        best = inf["frames_best"]
        for k in sorted(set([0, int(best[0]), len(sc.fr) // 4, len(sc.fr) // 2])):
            im = sc.fr[k].copy()
            for ln in sc.lines:
                pts = []
                for xq in (0, sc.W - 1):
                    if abs(math.cos(sc.th)) > 1e-3:               # c = -x sin + y cos -> y = (c + x sin) / cos
                        pts.append((xq, int(round((ln + xq * math.sin(sc.th)) / math.cos(sc.th)))))
                if len(pts) == 2:
                    cv2.line(im, pts[0], pts[1], (0, 255, 255), 1)
            for _, r in P[P.f == k].iterrows():
                bx, by, bw, bh = r.box
                cv2.rectangle(im, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
            cv2.putText(im, f"f{k}", (8, 20), 0, 0.6, (0, 0, 255), 2)
            y0, y1 = sc.box[0], sc.box[1]
            cv2.imwrite(f"{DBG}/{vid}_f{k}.jpg", im[max(0, y0 - 40):min(sc.H, y1 + 40)], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if inf["sweep_ratio"] is None or inf["sweep_ratio"] > a.gate or len(vs) < len(SWEEP):
        return None, f"unstable over the (thr, amin) sweep ({inf['sweep_ratio']}, {len(vs)}/{len(SWEEP)} settings)", inf
    return v_px * mpp, "ok", inf


def fuse(g, cs, mode="2d_ratio"):
    """the pipeline's plain fusion: agree within 2x -> geometric mean; else geo in strong modes, else CaW"""
    if not np.isfinite(cs) or cs <= 0:
        return g, "geo_only"
    if max(g / cs, cs / g) <= 2.0:
        return math.sqrt(g * cs), "agree_gm"
    return (g, "dis_geo") if mode in ("2d_ratio", "3d_calib") else (cs, "dis_caw")


def run(split):
    det = pd.read_csv(f"{R}/submissions/{a.sub}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    P = {int(json.loads(l)["id"]): json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")}
    srp = f"{R}/geo/out/scene_ruler_{split}.json"
    SR = json.load(open(srp)) if os.path.exists(srp) else {}
    out, ntrig = {}, 0
    for i, r in det.iterrows():
        p = P.get(int(i), {})
        qtxt = str(r.question) if "question" in det.columns and pd.notna(r.get("question")) else str((p.get("q") or {}).get("q", ""))
        if not TRIG.search(qtxt):
            continue
        ntrig += 1
        q = p.get("q") or {}
        src = str(r.answer_source)
        tag = f"[{split}] {i} {r.video_id} | {qtxt.strip()[:70]} | {src} ans={r.answer:.4g}"
        if src.startswith("patch"):
            print(tag, "-> skip: already patched"); continue
        if q.get("target2"):
            print(tag, "-> skip: a named pair (target2), not instances of one class"); continue
        if q.get("kind") not in ("dist2", "other", None):
            print(tag, f"-> skip: kind {q.get('kind')}"); continue
        unit = str(q.get("unit") or "m").lower()
        if unit not in UNIT:
            print(tag, f"-> skip: unit {unit} (not a length)"); continue
        sr = SR.get(str(r.video_id))
        if not sr or not sr.get("ok") or sr.get("kind") != "lane":
            print(tag, "-> skip: no lane scene ruler (no label-free spacing measurement)"); continue
        pri = [x for x in p.get("priors", []) if x.get("ok") and x.get("kind") == "size" and re.search(r"\blanes?\b", str(x.get("obj")), re.I)]
        if not pri:
            print(tag, "-> skip: prior is not the lane width"); continue
        gapq = bool(GAPW.search(qtxt))
        val_si, why, info = measure(str(r.video_id), split, sr, float(pri[0]["value_si"]), gapq)
        print(tag, "->", why, json.dumps(info))
        if val_si is None:
            continue
        g = val_si * UNIT[unit]
        cs = float(r.cs) if pd.notna(r.cs) else np.nan
        fin_plain, src_plain = fuse(g, cs)
        rat = max(g / cs, cs / g) if np.isfinite(cs) and cs > 0 else np.nan
        print(f"    geo {g:.3f}  CaW {cs:.4g}  ratio {rat:.2f}  sruler fusion -> geo {g:.3f}; plain fusion -> {src_plain} {fin_plain:.3f}")
        out[str(i)] = {"value": float(g), "why": (f"gap: mean adjacent {info['definition']} {info['value_px']:.1f} px of moving "
                                                   f"vehicles in one lane ({info['npair_max']} pairs/frame, median of "
                                                   f"{info['nframes_best']} frames) x {pri[0]['value_si']} m / {info['lane_px']:.1f} px "
                                                   f"lane; sruler row -> geo (CaW {cs:.4g}, ratio {rat:.2f}; plain fusion {src_plain})")}
    json.dump(out, open(f"{R}/cache/tmp/ag_{a.tag}_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] {ntrig} trigger rows; wrote {len(out)} rows -> cache/tmp/ag_{a.tag}_patch_{split}.json")
    return out


if __name__ == "__main__":
    for sp in ("val", "test"):
        run(sp)
