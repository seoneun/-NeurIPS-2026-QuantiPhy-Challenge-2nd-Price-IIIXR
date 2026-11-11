#!/usr/bin/env python
import os, sys, json, math, time, argparse
import numpy as np, cv2, torch

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools")
sys.path.insert(0, f"{R}/geo")
sys.path.insert(0, f"{R}/third_party/MoGe")
sys.path.insert(0, f"{R}/third_party/utils3d-moge")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max_frames", type=int, default=160)
ap.add_argument("--videos", default="")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
OUT = f"{R}/geo/out/moge_{a.split}"
os.makedirs(OUT, exist_ok=True)

from moge.model.v2 import MoGeModel
dev = "cuda"
model = MoGeModel.from_pretrained(f"{R}/models/moge-2-vitl-normal/model.pt").to(dev).eval()

df = load_split(a.split)
vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
vids = sorted(vpath)
if a.videos:
    vids = [v for v in vids if v in a.videos.split(",")]
t00 = time.time()
for vi, v in enumerate(vids):
    op = f"{OUT}/{v}.npz"
    if os.path.exists(op) and not a.overwrite:
        continue
    t0 = time.time()
    frames = tc.load_video(vpath[v])
    N = len(frames)
    stride = max(1, int(math.ceil(N / a.max_frames)))
    fl = list(range(0, N, stride))
    J = np.unique(np.linspace(0, len(fl) - 1, min(a.k, len(fl))).round().astype(int))
    H, W = frames[0].shape[:2]
    fx, fy, deps = [], [], []
    with torch.no_grad():
        for j in J:
            img = torch.tensor(frames[fl[j]] / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
            o = model.infer(img, use_fp16=True)
            K = o["intrinsics"].float().cpu().numpy()
            fx.append(float(K[0, 0] * W)); fy.append(float(K[1, 1] * H))
            d = o["depth"].float().cpu().numpy()
            m = o["mask"].cpu().numpy().astype(bool) if "mask" in o else np.isfinite(d)
            d = np.where(m & np.isfinite(d), d, np.nan)
            d2 = cv2.resize(d, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
            deps.append(d2.astype(np.float16))
    np.savez_compressed(op, j=J, fx=np.array(fx), fy=np.array(fy), depth=np.stack(deps), W=W, H=H, N=N, stride=stride)
    del frames
    print('', flush=True)
print("DONE", round(time.time() - t00), flush=True)
