#!/usr/bin/env python
import json, os, sys
import numpy as np
import pandas as pd
import cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/geo")
sys.path.insert(0, f"{R}/tools")
cv2.setNumThreads(2)
import measure_v3 as MV
from qp_common import load_split

split = sys.argv[1] if len(sys.argv) > 1 else "val"
EXP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
MINF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
SRC = sys.argv[5] if len(sys.argv) > 5 else "480"
VIDS = set(sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[6] else None
MODE = sys.argv[7] if len(sys.argv) > 7 else "single"
SPOT = len(sys.argv) > 8 and sys.argv[8] == "spot"
MARK = len(sys.argv) > 8 and sys.argv[8] == "mark"
TRAIL = len(sys.argv) > 8 and sys.argv[8] == "trail"
FLIP = len(sys.argv) > 8 and sys.argv[8] == "flip"
TRK = sys.argv[9] if len(sys.argv) > 9 else "v2"
SKIP = set()
if TRK.startswith("v3") and os.path.exists(f"{R}/caw/sc/inputs/{split}_crop_map.csv"):
    SKIP = set(pd.read_csv(f"{R}/caw/sc/inputs/{split}_crop_map.csv").id.astype(int))
OUT = f"{R}/cache/tmp/crops_{split}{TAG}"
os.makedirs(OUT, exist_ok=True)
MV.FIX = set("abcdef")
MV.TRACK_VER = TRK
byv, vjs = MV.load_split_tracks(split, "e")
df = load_split(split)
df.index = df["Unnamed: 0"].astype(int)
rows, meta = [], []
for v, recs in byv.items():
    if v not in vjs or (VIDS is not None and v not in VIDS):
        continue
    vj = vjs[v]
    V = MV.Video(vj, "e")
    W, H = V.W, V.H
    cap = None
    frames = None
    for r in recs:
        if r["id"] in SKIP:
            continue
        tgt = V.get(r["target"])
        if tgt is None or tgt.n < 2:
            continue
        is_two = bool(r.get("target2")) or r["q"].get("kind") == "dist2"
        if MODE == "withprior":
            if is_two:
                continue
            pr_tr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
            if pr_tr is None or pr_tr.n < 2 or pr_tr is tgt:
                continue
        elif (MODE == "two") != is_two:
            continue
        boxes = np.array([tgt.st[j]["box"] for j in tgt.okc], float)
        if MODE == "two":
            t2 = V.get(r["target2"]) if r.get("target2") else None
            if t2 is None or t2.n < 2:
                continue
            boxes = np.concatenate([boxes, np.array([t2.st[j]["box"] for j in t2.okc], float)])
        if MODE == "withprior":
            boxes = np.concatenate([boxes, np.array([pr_tr.st[j]["box"] for j in pr_tr.okc], float)])
        x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
        area = float(np.median((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))) / (W * H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max(EXP * (x1 - x0), EXP * (y1 - y0) * W / H, MINF * min(W, H) * W / H)
        w = min(w, W)
        h = w * H / W
        if h > H:
            h, w = H, H * W / H
        xa = int(round(min(max(cx - w / 2, 0), W - w)))
        ya = int(round(min(max(cy - h / 2, 0), H - h)))
        wi, hi = int(round(w)), int(round(h))
        zoom = W / max(wi, 1)
        if zoom < 1.15:
            continue
        if frames is None:
            path = df[df.video_id.astype(str).str.strip() == v].video_path.iloc[0]
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                hh, ww = f.shape[:2]
                s = min(hh, ww)
                if s > 480 and SRC != "orig":
                    f = cv2.resize(f, (int(round(ww * 480 / s)), int(round(hh * 480 / s))), interpolation=cv2.INTER_AREA)
                frames.append(f)
            cap.release()
        fps = float(vj["fps"])
        op = f"{OUT}/crop{TAG}_{r['id']}.mp4"
        wr = cv2.VideoWriter(op, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        sc = frames[0].shape[1] / float(W)
        if TRAIL and r["q"].get("kind") not in ("speed", "avgspeed", "hvel", "accel", "disp", "path"):
            continue
        if MARK or TRAIL:
            fidx = np.array(vj["fidx"]); okj = [j for j in range(len(tgt.st)) if tgt.st[j]]
            okf = fidx[okj] if okj else np.array([])
            okc_ = np.array([tgt.st[j]["c"] for j in okj], float) if okj else np.zeros((0, 2))
        bx0, by0 = cx - 0.6 * (x1 - x0), cy - 0.6 * (y1 - y0)
        bx1, by1 = cx + 0.6 * (x1 - x0), cy + 0.6 * (y1 - y0)
        xs, ys, ws, hs = int(round(xa * sc)), int(round(ya * sc)), int(round(wi * sc)), int(round(hi * sc))
        fno = 0
        for f in frames:
            c = f[ys:ys + hs, xs:xs + ws]
            if SPOT:
                b = cv2.GaussianBlur(c, (0, 0), 8 * sc) // 2
                ix0, iy0 = int(max(0, (bx0 - xa) * sc)), int(max(0, (by0 - ya) * sc))
                ix1, iy1 = int(min(ws, (bx1 - xa) * sc + 1)), int(min(hs, (by1 - ya) * sc + 1))
                b[iy0:iy1, ix0:ix1] = c[iy0:iy1, ix0:ix1]
                c = b
            out_f = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if sc * wi > W else cv2.INTER_CUBIC)
            if TRAIL and len(okf):
                zx, zy = W / float(wi), H / float(hi)
                pts = [(int(round((c[0] - xa) * zx)), int(round((c[1] - ya) * zy))) for f_, c in zip(okf, okc_) if f_ <= fno]
                if len(pts) >= 2:
                    cv2.polylines(out_f, [np.array(pts, np.int32)], False, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            if MARK and len(okf):
                k = int(np.argmin(np.abs(okf - fno)))
                if abs(int(okf[k]) - fno) <= 2 * max(1, int(vj.get("stride", 1))):
                    bx = tgt.st[okj[k]]["box"]
                    zx, zy = W / float(wi), H / float(hi)
                    p0 = (int(round((bx[0] - xa) * zx)) - 3, int(round((bx[1] - ya) * zy)) - 3)
                    p1 = (int(round((bx[2] - xa) * zx)) + 3, int(round((bx[3] - ya) * zy)) + 3)
                    cv2.rectangle(out_f, p0, p1, (0, 0, 255), 2)
            if FLIP:
                out_f = cv2.flip(out_f, 1)
            wr.write(out_f)
            fno += 1
        wr.release()
        o = df.loc[r["id"]].copy()
        o["video_id"] = f"crop{TAG}_{r['id']}"
        rows.append(o)
        meta.append(dict(id=r["id"], video_id=v, area_frac=area, zoom=zoom, crop=[xa, ya, wi, hi]))
cols = ["Unnamed: 0", "video_id", "video_source", "video_type", "fps", "inference_type", "question", "ground_truth_prior",
        "depth_info", "ground_truth_posterior"]
out = pd.DataFrame(rows)[cols].rename(columns={"Unnamed: 0": ""})
out.to_csv(f"{R}/caw/sc/inputs/{split}_crop{TAG}.csv", index=False)
pd.DataFrame(meta).to_csv(f"{R}/caw/sc/inputs/{split}_crop{TAG}_map.csv", index=False)
m = pd.DataFrame(meta)
print('')
