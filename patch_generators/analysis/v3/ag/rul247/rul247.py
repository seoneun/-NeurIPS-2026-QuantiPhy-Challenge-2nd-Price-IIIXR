"""rul247: height of a rigid box that rests on / slides along an incline.
geo reads 'height' as the median axis-aligned box height bh over usable frames. A cuboid tilted by th has
bh = h cos(th) + L sin(th) > h. When the box lies on an incline, its base is parallel to its own path, so its height is
the min-area-rectangle (rr) side perpendicular to the path, read on the frames where the rr angle is the incline tilt.
(rul041 instead needs level frames and a 9.81 gravity witness: the witness compares the slide-accel ruler with
g = 9.81 / a_ff, but sim gravity in prior units is unknown and the slide ruler itself moves 10-25 % with the fit
window, so its veto is noise. Here gravity is used only as a bracket, never as a value.)
Rule (label-free, code only):
  rows : q.kind size, q.dim height, target head noun in BOXN, raw meas_mode 2d_ratio, final answer_source in
         {dis_geo, geo_only, geo_only_mode, agree_wgm, agree_gm} (never patch:*, copies, 3D, CaW-routed).
  track: the track geo used (meas_tracks v1 -> tracks_{split}, else _<tv>, _v2), key phrase|where|rel.
  frames: area 0.4..2.5 x median, border == 0; n >= 10.
  incline frames: rr off-axis angle within 3 deg of the dominant tilt (mode of off-axis >= 10 deg, 10..40 deg);
         >= 8 frames and >= 40 % of usable frames; their centroid path (principal axis) spans >= 2 box heights and
         its direction matches the direction of rr side 0 (rr angle + 90) within 5 deg -> base parallel to the path,
         height = rr side 1. Check: that assignment predicts the observed sign of (bw - bh) on those frames.
  value: h_px = median rr side 1 on incline frames; geo's own px = g / meas_mpp; r = h_px / px; fire if r < 1/1.15.
  gravity bracket (only when the target has its own clean vertical free fall: >= 5 frames, centre-x ptp <= 3 px,
         quadratic rms <= 2 px, a_ff > 0): rulers {meas_mpp, 9.8 / a_ff}; fire only if g > 1.1 x h_px x max(ruler)
         (the old value is too high under either ruler). The new value always uses geo's own meas_mpp.
  fusion: g' = h_px x meas_mpp; pipeline rule (CaW within 2x -> g'^0.65 cs^0.35, else g'); if the agreement status
         is unchanged, answer x r^w (w = 1 dis_geo / geo_only, 0.65 agree_wgm, 0.5 agree_gm).
usage: python rul247.py test|val
"""
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
SUB = os.environ.get("IN_SUB", "sub_xt_Nzz39")
TAG = "rul247"
BOXN = {"box", "toolbox", "crate", "case", "chest", "carton", "package", "parcel", "suitcase", "cube", "block", "brick"}
SRC_W = {"dis_geo": 1.0, "geo_only": 1.0, "geo_only_mode": 1.0, "agree_wgm": 0.65, "agree_gm": 0.5}
G_BRACKET = 9.8
split = sys.argv[1]


def off_axis(a):
    a = abs(float(a)) % 90.0
    return min(a, 90.0 - a)


def wrap90(a):
    return (a + 90.0) % 180.0 - 90.0


def load_track(vid, tv, key):
    fns = [f"{R}/geo/out/tracks_{split}/{vid}.json"] if tv == "v1" else \
          [f"{R}/geo/out/tracks_{split}_{tv}/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json"]
    fn = next((f for f in fns if os.path.exists(f)), None)
    if fn is None:
        return None, None
    vj = json.load(open(fn))
    q = next((q for q in vj["queries"] if q["key"] == key), None)
    if q is None:
        return None, vj
    if "objs" in vj:
        if q.get("obj") is None:
            return None, vj
        return [(f or {}).get("st") for f in vj["objs"][str(q["obj"])]], vj
    return [(f or {}).get("st") for f in q["frames"]], vj


def freefall(st, oks, fidx, fps):
    """longest leading run of usable frames with a clean vertical free fall -> a_ff px/s^2 or None"""
    run = []
    for j in oks:
        if run and j - run[-1] > 2:
            break
        run.append(j)
    best = None
    for n in range(len(run), 4, -1):
        js = run[:n]
        t = np.array([fidx[j] for j in js], float) / fps
        c = np.array([st[j]["c"] for j in js], float)
        if np.ptp(c[:, 0]) > 3.0 or c[-1, 1] - c[0, 1] < 5:
            continue
        p = np.polyfit(t - t[0], c[:, 1], 2)
        rms = float(np.sqrt(np.mean((c[:, 1] - np.polyval(p, t - t[0])) ** 2)))
        if 2 * p[0] > 0 and rms <= 2.0:
            best = (2 * p[0], n, rms)
            break
    return best


det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0)
P = {}
for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
    d = json.loads(l); P[d["id"]] = d
out, log = {}, []
for i, p in P.items():
    q = p.get("q") or {}
    if q.get("kind") != "size" or q.get("dim") != "height" or i not in raw.index or i not in det.index:
        continue
    tgt = p.get("target") or {}
    phrase = (tgt.get("phrase") or q.get("target") or "").strip().lower()
    hd = (re.sub(r"[^a-z ]", " ", phrase).split() or [""])[-1]
    if hd not in BOXN:
        continue
    rw, dt = raw.loc[i], det.loc[i]
    rec = dict(id=i, vid=rw.video_id, src=dt.answer_source, ans=round(float(dt.answer), 4), g=dt.g, cs=dt.cs)
    if "y" in det.columns:
        rec["y"] = det.y.loc[i]
    if rw.meas_mode != "2d_ratio":
        rec["why"] = "mode"; log.append(rec); continue
    if dt.answer_source not in SRC_W:
        rec["why"] = "source"; log.append(rec); continue
    mpp = float(rw.meas_mpp)
    if not (mpp == mpp and mpp > 0 and dt.g == dt.g):
        rec["why"] = "no_ruler"; log.append(rec); continue
    rel = tgt.get("rel")
    key = (tgt.get("phrase") or q.get("target") or "") + "|" + ",".join(tgt.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")
    st, vj = load_track(rw.video_id, str(rw.meas_tracks), key)
    if st is None:
        rec["why"] = "no_track"; log.append(rec); continue
    area = np.array([s["area"] if s else 0 for s in st], float)
    med = np.median(area[area > 0]) if (area > 0).any() else 0
    bd = np.array([s.get("border", 9) if s else 9 for s in st])
    oks = np.where((area > 0.4 * med) & (area < 2.5 * med) & (bd == 0))[0]
    if len(oks) < 10:
        rec["why"] = "few_frames"; log.append(rec); continue
    ang = np.array([float(st[j]["rr"][2]) for j in oks])
    off = np.array([off_axis(a) for a in ang])
    tl = off[off >= 10]
    if len(tl) < 8:
        rec["why"] = "no_tilt"; log.append(rec); continue
    h_, e_ = np.histogram(tl, bins=np.arange(10, 46, 1.0))
    k = int(np.argmax(h_)); mode = float(np.median(tl[(tl >= e_[k] - 2) & (tl <= e_[k + 1] + 2)]))
    inc = np.abs(off - mode) <= 3.0
    rec.update(n=len(oks), tilt=round(mode, 1), ninc=int(inc.sum()))
    if not (10 <= mode <= 40) or inc.sum() < 8 or inc.mean() < 0.4:
        rec["why"] = "no_incline_frames"; log.append(rec); continue
    J = oks[inc]
    rr = np.array([st[j]["rr"][:2] for j in J], float)
    box = np.array([st[j]["box"] for j in J], float)
    bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    c = np.array([st[j]["c"] for j in J], float)
    Pc = c - c.mean(0); u = np.linalg.svd(Pc)[2][0]
    span = float(np.ptp(Pc @ u))
    pang = wrap90(float(np.degrees(np.arctan2(u[1], u[0]))))
    a0 = wrap90(float(np.median(ang[inc])) + 90.0)
    dpath = min(abs(a0 - pang), 180 - abs(a0 - pang))
    rec.update(span=round(span, 1), path=round(pang, 1), side0=round(a0, 1), dpath=round(dpath, 1))
    if span < 2 * np.median(bh) or dpath > 5:
        rec["why"] = "path_not_incline"; log.append(rec); continue
    th = np.radians(mode)
    A_, B_ = float(np.median(rr[:, 0])), float(np.median(rr[:, 1]))
    pred = (A_ * np.cos(th) + B_ * np.sin(th)) - (A_ * np.sin(th) + B_ * np.cos(th))
    obs = float(np.median(bw - bh))
    rec.update(A=round(A_, 1), B=round(B_, 1), pred_dwh=round(pred, 1), obs_dwh=round(obs, 1))
    if pred * obs < 0:
        rec["why"] = "assignment_mismatch"; log.append(rec); continue
    px = float(dt.g) / mpp
    r = B_ / px
    rec.update(px=round(px, 1), r=round(r, 3))
    if r >= 1 / 1.15:
        rec["why"] = "small_r"; log.append(rec); continue
    ff = freefall(st, list(oks), vj["fidx"], float(rw.fps))
    if ff is not None:
        rulers = [mpp, G_BRACKET / ff[0]]
        hi = B_ * max(rulers)
        rec.update(a_ff=round(ff[0]), ff_n=ff[1], mpp_g=round(G_BRACKET / ff[0], 5), hi=round(hi, 4))
        if not float(dt.g) > 1.1 * hi:
            rec["why"] = "gravity_bracket_contains_old"; log.append(rec); continue
    g2 = B_ * mpp
    cs = float(dt.cs) if dt.cs == dt.cs else np.nan
    src = dt.answer_source
    ag_old = src.startswith("agree"); ag_new = cs == cs and abs(np.log10(g2 / cs)) <= np.log10(2)
    if cs != cs or ag_new == ag_old:
        new = float(dt.answer) * r ** SRC_W[src]; how = f"{src} x r^{SRC_W[src]}"
    elif ag_new:
        new = g2 ** 0.65 * cs ** 0.35; how = "now agrees -> wgm 0.65"
    else:
        new = g2; how = "now disagrees -> geo"
    rec.update(new=round(new, 4), how=how, why="FIRE"); log.append(rec)
    out[str(i)] = dict(value=float(new), why=f"{TAG}: box on incline (tilt {mode:.0f} deg, path {pang:.0f} deg, {int(inc.sum())} frames); "
                       f"height = rr side perpendicular to path {B_:.0f}px vs geo {px:.0f}px (r {r:.3f}); "
                       f"{'gravity bracket hi %.3f' % rec['hi'] if ff is not None else 'no free fall'}; {how}")
L = pd.DataFrame(log)
pd.set_option("display.width", 340); pd.set_option("display.max_colwidth", 40)
print(f"[{split}] {TAG}: {len(L)} scanned, {len(out)} fire")
if len(L):
    print(L.to_string())
json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
print("wrote", f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json")
