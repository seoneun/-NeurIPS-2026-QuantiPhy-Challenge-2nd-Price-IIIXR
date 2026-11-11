#!/usr/bin/env python
import os, sys, json, re, argparse
import numpy as np, pandas as pd, cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

cv2.setNumThreads(2)
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--meas", default="_v3m_rrj5plm")
ap.add_argument("--videos", default="")
ap.add_argument("--lag", type=float, default=0.12)
ap.add_argument("--thr", type=float, default=22.0)
a = ap.parse_args()

m = pd.read_csv(f"{R}/geo/out/meas_{a.split}_v2{a.meas}.csv")
m["w"] = m.why.fillna("").map(lambda s: re.sub(r"\(.*", "", str(s)))
P = {}
for ln in open(f"{R}/geo/out/parse_{a.split}_v2.jsonl", encoding="utf-8"):
    x = json.loads(ln); P[int(x["id"])] = x
MOTION_KINDS = {"speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel", "orbitdiam"}
need, meta = {}, {}
WHYS = ["prior_static_track", "target_static_track"] + (os.environ.get("MV_EXTRA_WHYS", "").split(",") if os.environ.get("MV_EXTRA_WHYS") else [])
for _, r in m[m.w.isin(WHYS) & (m.cam_mag.fillna(1) <= 0.06)].iterrows():
    v_ = str(r["video_id"])
    if v_.startswith("internet"):
        continue
    x = P.get(int(r["id"])) or {}
    role = "prior_obj" if str(r["w"]).startswith("prior_") else "target"
    if role == "prior_obj" and str(r["w"]) != "prior_static_track":
        pk_ = ((P.get(int(r["id"])) or {}).get("priors") or [{}])[0].get("kind")
        if pk_ not in ("speed", "accel"):
            continue
    if role == "target" and str(r["kind"]) not in MOTION_KINDS:
        continue
    ph = ((x.get(role) or {}).get("phrase") or "").strip()
    pr_ = (x.get("priors") or [{}])[0]
    grav = bool(pr_.get("gravity") or (pr_.get("kind") == "accel" and 9.0 <= float(pr_.get("value_si") or 0) <= 10.0))
    pph = ((x.get("prior_obj") or {}).get("phrase") or "").strip()
    if grav and ph != pph:
        continue
    if ph:
        need.setdefault(v_, {})[ph] = "prior" if role == "prior_obj" else "target"
if a.videos:
    keep = set(a.videos.split(","))
    need = {k: v for k, v in need.items() if k in keep}

df = load_split(a.split)
vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
fpsd = {str(v).strip(): float(f) for v, f in zip(df.video_id, df.fps)}


def blobs(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(float(cen[i][0]), float(cen[i][1]), int(st[i][cv2.CC_STAT_AREA]),
             [int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])])
            for i in range(1, n) if st[i][cv2.CC_STAT_AREA] >= min_area]


out = {}
for v, ph in sorted(need.items()):
    tp = f"{R}/geo/out/tracks_{a.split}_v2/{v}.json"
    if not os.path.exists(tp):
        tp = f"{R}/geo/out/tracks_{a.split}/{v}.json"
    if not os.path.exists(tp):
        print(v, "no track file"); continue
    tj = json.load(open(tp))
    frames_all = tc.load_video(str(vpath[v]))
    frames = [frames_all[i] for i in tj["fidx"] if i < len(frames_all)]
    del frames_all
    N = len(frames); H, W = frames[0].shape[:2]
    fps = fpsd.get(v, 30.0) / float(tj.get("stride") or 1)
    F = np.stack([cv2.GaussianBlur(f, (5, 5), 0) for f in frames]).astype(np.int16)
    ker = np.ones((3, 3), np.uint8)
    min_area = max(12, int(0.00004 * H * W))
    best_movers, used_lag = [], None
    for lag_s in (a.lag, 0.4, 1.2):
      L = max(2, int(round(lag_s * fps)))
      if 2 * L >= N - 4:
        break
      det = []
      for k in range(L, N - L):
        d1 = np.abs(F[k] - F[k - L]).max(-1); d2 = np.abs(F[k] - F[k + L]).max(-1)
        mk = ((d1 > a.thr) & (d2 > a.thr)).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, ker)
        mk = cv2.dilate(mk, ker, iterations=2)
        for b in blobs(mk, min_area):
            det.append((k,) + b)
      tracks = []
      for k, x, y, ar, bb in det:
        best, bd = None, 1e9
        for t in tracks:
            kk, xx, yy, aa, _ = t[-1]
            if kk >= k or k - kk > 3 * L:
                continue
            dd = np.hypot(x - xx, y - yy)
            if dd < max(25.0, 3.0 * np.sqrt(max(aa, ar))) and dd < bd:
                best, bd = t, dd
        if best is None:
            tracks.append([(k, x, y, ar, bb)])
        else:
            best.append((k, x, y, ar, bb))
      movers = []
      for t in tracks:
        ks = np.array([p[0] for p in t]); xy = np.array([[p[1], p[2]] for p in t]); ar = np.array([p[3] for p in t])
        if len(t) < max(4, 0.08 * N):
            continue
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        if span < 8:
            continue
        sel = np.linspace(0, len(t) - 1, min(5, len(t))).round().astype(int)
        movers.append(dict(pts=[[int(ks[i]), round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1)] for i in sel],
                           path=round(path, 1), span=round(span, 1), cover=round(len(t) / N, 3), area=int(np.median(ar)),
                           k0=int(ks.min()), k1=int(ks.max()), lag=round(lag_s, 2)))
      movers.sort(key=lambda d: -d["cover"] * d["span"])
      if movers and max(d["cover"] for d in movers) >= 0.2:
        best_movers, used_lag = movers, lag_s
        break
      if not best_movers and movers:
        best_movers, used_lag = movers, lag_s
    movers = best_movers
    out[v] = dict(W=W, H=H, N=N, fps=fps, need=ph, lag=used_lag, track_src=tp.split("/")[-2], movers=movers[:6])
    print(v, ph, "N", N, "movers", len(movers), [(d["cover"], d["span"], d["area"], d["pts"][0]) for d in movers[:4]], flush=True)
json.dump(out, open(f"{R}/geo/out/movers_{a.split}.json", "w"), indent=1)
print("wrote", len(out), "videos")
