#!/usr/bin/env python
import os, re, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import track_common as tc
from transformers import AutoProcessor, GroundingDinoForObjectDetection

ap = argparse.ArgumentParser()
ap.add_argument("split")
ap.add_argument("--height", type=float, default=1.70)
ap.add_argument("--thr", type=float, default=0.3)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
gp = AutoProcessor.from_pretrained(f"{R}/models/grounding-dino-base")
gm = GroundingDinoForObjectDetection.from_pretrained(f"{R}/models/grounding-dino-base").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
PER = re.compile(r"person|people|\bman\b|woman|girl|boy|human|player|child|kid|lady|guy", re.I)


@torch.no_grad()
def detect(frame, text="person."):
    H, W = frame.shape[:2]
    s = 800 / min(H, W)
    if max(H, W) * s > 1333:
        s = 1333 / max(H, W)
    h, w = int(round(H * s)), int(round(W * s))
    x = torch.from_numpy(frame[None]).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = (F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True) - MEAN) / STD
    tok = gp.tokenizer([text], return_tensors="pt", padding=True).to(dev)
    out = gm(pixel_values=x, input_ids=tok.input_ids, attention_mask=tok.attention_mask,
             token_type_ids=tok.get("token_type_ids"), pixel_mask=torch.ones((1, h, w), dtype=torch.long, device=dev))
    res = gp.post_process_grounded_object_detection(out, tok.input_ids, threshold=a.thr, text_threshold=a.thr, target_sizes=[(H, W)])[0]
    b = res["boxes"].cpu().numpy().astype(float); sc = res["scores"].cpu().numpy().astype(float)
    keep = tc.nms(b, sc, 0.6) if len(b) else []
    return [b[i] for i in keep]


def full_body(b, H):
    x1, y1, x2, y2 = b
    return y1 > 2 and y2 < H - 2 and (y2 - y1) >= 1.6 * max(x2 - x1, 1) and (y2 - y1) > 20


def f_from(b, Z, W, H):
    hpx = b[3] - b[1]
    u, v = (b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2
    f = hpx * Z / a.height
    for _ in range(3):
        z = Z / np.sqrt(1 + (u / f) ** 2 + (v / f) ** 2)
        f = hpx * z / a.height
    return f / W


raw = pd.read_csv(f"{R}/preds/{a.split}_geo_v4ntrj5sdsrngsr_d_vw_raw.csv").set_index("Unnamed: 0")
vpath = raw.groupby("video_id").video_path.first().to_dict()
vids = {}
for ln in open(f"{R}/geo/out/parse_{a.split}_v2.jsonl"):
    x = json.loads(ln)
    for e in x.get("depth") or []:
        if PER.search(e["key"]):
            vids.setdefault(x["video_id"], {"fps": x.get("fps") or 30.0, "ents": set()})["ents"].add((e["key"], e["t"], e["d"]))
print('', flush=True)
out = {}
for n, (v, info) in enumerate(sorted(vids.items())):
    if v not in vpath:
        continue
    frames = tc.load_video(str(vpath[v]))
    if not frames:
        continue
    H, W = frames[0].shape[:2]; N = len(frames); fps = float(info["fps"])
    ents = sorted(info["ents"], key=lambda z: (z[0], -1 if z[1] is None else z[1]))
    keys = sorted({k for k, _, _ in ents})
    times = sorted({t for _, t, _ in ents if t is not None})
    fidx = sorted({min(N - 1, int(round(t * fps))) for t in times}) if times else []
    if any(t is None for _, t, _ in ents):
        fidx = sorted(set(fidx) | set(np.linspace(0, N - 1, 8).round().astype(int).tolist()))
    dets = {j: [b for b in detect(frames[j]) if full_body(b, H)] for j in fidx}
    ests = []
    for k, t, Z in ents:
        js = [min(N - 1, int(round(t * fps)))] if t is not None else [j for j in fidx]
        for j in js:
            bs = dets.get(j) or []
            if not bs:
                continue
            bs = sorted(bs, key=lambda b: -(b[3] - b[1]))
            kl = k.lower()
            if len(keys) == 1 or not any(w_ in kl for w_ in ("near", "far", "left", "right")):
                if len(bs) != 1:
                    continue
                b = bs[0]
            elif "near" in kl or "far" in kl:
                if len(bs) != 2:
                    continue
                b = bs[0] if "near" in kl else bs[1]
            else:
                if len(bs) != 2:
                    continue
                two = sorted(bs[:2], key=lambda b: b[0] + b[2])
                b = two[0] if "left" in kl else two[1]
            ests.append(dict(key=k, t=t, Z=Z, j=j, h_px=float(b[3] - b[1]), f_over_W=float(f_from(b, Z, W, H))))
    if ests:
        fw = np.array([e["f_over_W"] for e in ests])
        out[v] = dict(f_over_W=float(np.median(fw)), n=len(ests), spread=float(fw.max() / fw.min()), W=W, H=H, ests=ests)
    print('', flush=True)
json.dump(out, open(f"{R}/geo/out/person_focal_{a.split}.json", "w"), indent=1)
print("wrote", len(out))
