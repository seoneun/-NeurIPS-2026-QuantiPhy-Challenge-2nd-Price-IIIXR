#!/usr/bin/env python
import os, sys, json, time, argparse
import numpy as np, torch

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
a = ap.parse_args()
dev = "cuda"
from transformers import Sam2VideoModel, Sam2VideoProcessor
sp = Sam2VideoProcessor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
sm = Sam2VideoModel.from_pretrained(f"{R}/models/sam2.1-hiera-large").to(dev, dtype=torch.bfloat16).eval()


@torch.no_grad()
def track_points(frames, prompts):
    sess = sp.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
    k0 = prompts[len(prompts) // 2][0]
    for k, p in prompts:
        sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=int(k), obj_ids=[1],
                                           input_points=[[[[float(p[0]), float(p[1])]]]], input_labels=[[[1]]])
    res = {}
    for rev in (False, True):
        for o in sm.propagate_in_video_iterator(sess, start_frame_idx=int(k0), reverse=rev):
            if rev and o.frame_idx == k0:
                continue
            m = sp.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]], binarize=False)[0]
            m = (m[:, 0] > 0).cpu().numpy()[0]
            lg = float(o.object_score_logits.float().cpu().numpy().ravel()[0])
            st = tc.mask_stats(m) if lg > 0 else None
            res[int(o.frame_idx)] = dict(st=st, lg=lg)
    del sess
    torch.cuda.empty_cache()
    return res


def summ(objs):
    c = np.array([o["st"]["c"] for o in objs if o and o.get("st")], float)
    ar = np.array([o["st"]["area"] for o in objs if o and o.get("st")], float)
    if len(c) < 3:
        return 0.0, 0.0, 0.0
    return float(np.linalg.norm(c.max(0) - c.min(0))), float(len(c) / len(objs)), float(np.median(ar))


def box_iou_seq(A, B):
    v = [tc.iou(x["st"]["box"], y["st"]["box"]) for x, y in zip(A, B) if x and y and x.get("st") and y.get("st")]
    return float(np.mean(v)) if v else 0.0


MV = json.load(open(f"{R}/geo/out/movers_{a.split}.json"))
df = load_split(a.split)
vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
out = {}
t00 = time.time()
for vi, (v, info) in enumerate(sorted(MV.items())):
    if not info["movers"]:
        continue
    tp = f"{R}/geo/out/{info['track_src']}/{v}.json"
    tj = json.load(open(tp))
    frames_all = tc.load_video(vpath[v])
    frames = [frames_all[i] for i in tj["fidx"] if i < len(frames_all)]
    del frames_all
    N = len(frames); H, W = frames[0].shape[:2]
    kept = []
    for mv in info["movers"][:4]:
        pts = mv["pts"]
        sel = [pts[0], pts[len(pts) // 2], pts[-1]] if len(pts) >= 3 else pts
        res = track_points(frames, [(k, [x, y]) for k, x, y in sel])
        objs = [res.get(k, dict(st=None, lg=-10.0)) for k in range(N)]
        span, cover, area = summ(objs)
        ok = cover >= 0.4 and span >= max(8.0, 0.5 * np.sqrt(max(area, 1.0))) and area <= 0.08 * H * W
        print('', flush=True)
        if not ok:
            continue
        dup = next((t for t in kept if box_iou_seq(t["objs"], objs) > 0.4), None)
        if dup is not None:
            if cover * span > dup["cover"] * dup["span"]:
                dup.update(objs=objs, span=span, cover=cover, area=area)
            continue
        kept.append(dict(objs=objs, span=span, cover=cover, area=area, taken=None))
    for t in kept:
        for q in tj.get("queries", []):
            oid = q.get("obj")
            if oid is None or str(oid) not in tj["objs"] or q.get("phrase") in info["need"]:
                continue
            ob = tj["objs"][str(oid)]
            if len(ob) != N:
                continue
            s2, c2, _ = summ(ob)
            if s2 >= 8 and box_iou_seq(ob, t["objs"]) > 0.4:
                t["taken"] = q.get("phrase")
    out[v] = dict(need=info["need"], track_src=info["track_src"], tracks=kept)
    print('', flush=True)
json.dump(out, open(f"{R}/geo/out/movertracks_{a.split}.json", "w"))
print("DONE", len(out), round(time.time() - t00), flush=True)
