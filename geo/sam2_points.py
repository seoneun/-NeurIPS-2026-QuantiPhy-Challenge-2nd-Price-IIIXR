#!/usr/bin/env python
import os, sys, json, time, shutil, argparse
import numpy as np, torch, cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--src", default="v2")
a = ap.parse_args()
dev = "cuda"
from transformers import Sam2VideoModel, Sam2VideoProcessor
sp = Sam2VideoProcessor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
sm = Sam2VideoModel.from_pretrained(f"{R}/models/sam2.1-hiera-large").to(dev, dtype=torch.bfloat16).eval()
OUT = f"{R}/geo/out/tracks_{a.split}_{a.src}mo"
os.makedirs(OUT, exist_ok=True)

pts = {}
for ln in open(f"{R}/geo/out/molmoground_{a.split}.jsonl", encoding="utf-8"):
    x = json.loads(ln)
    if x["points"]:
        pts.setdefault((x["video_id"], x["phrase"]), []).append((x["k"], x["points"]))


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


df = load_split(a.split)
vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
src_dir = f"{R}/geo/out/tracks_{a.split}_{a.src}"
vids = sorted(set(v for v, _ in pts))
t00 = time.time()
n_patched = 0
for f in os.listdir(src_dir):
    if f.endswith(".json") and not os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{src_dir}/{f}", f"{OUT}/{f}")
for vi, v in enumerate(vids):
    p = f"{src_dir}/{v}.json"
    if not os.path.exists(p):
        continue
    tj = json.load(open(p))
    if "objs" not in tj:
        continue
    frames_all = tc.load_video(vpath[v])
    frames = [frames_all[i] for i in tj["fidx"] if i < len(frames_all)]
    del frames_all
    changed = False
    for (vv, ph), plist in pts.items():
        if vv != v:
            continue
        prompts = [(k, ps[0]) for k, ps in sorted(plist) if k < len(frames)]
        if not prompts:
            continue
        res = track_points(frames, prompts)
        newo = [res.get(k, dict(st=None, lg=-10.0)) for k in range(len(frames))]
        if sum(1 for x in newo if x["st"]) < 3:
            continue
        qs = [q for qk in ("queries", "queries_d") for q in tj.get(qk, []) if q.get("phrase") == ph]
        oid = next((str(q["obj"]) for q in qs if q.get("obj") is not None), None)
        if oid is None:
            oid = str(max([int(x) for x in tj["objs"]] + [0]) + 1)
            k0 = prompts[len(prompts) // 2][0]
            b0 = newo[k0]["st"]["box"] if newo[k0]["st"] else [0, 0, 1, 1]
            tj.setdefault("queries", []).append(dict(key=ph + "|", phrase=ph, where=[], rel=None, roles=["target"], det_scores=[], det_boxes=[],
                                                      det_gd=[], det_owl=[], det_ngood=[], best_score=0.5, best_gd=0.5, best_owl=0.5,
                                                      anchor=[int(k0), b0], obj=int(oid), det_iou=[]))
            qs = [tj["queries"][-1]]
        tj["objs"][oid] = newo
        for q in qs:
            q.update(obj=int(oid), best_gd=max(q.get("best_gd") or 0, 0.5), best_owl=max(q.get("best_owl") or 0, 0.5),
                     best_score=max(q.get("best_score") or 0, 0.5), det_iou=[1.0] * len(q.get("det_iou") or [1]), molmo=True)
        changed = True; n_patched += 1
    if changed:
        json.dump(tj, open(f"{OUT}/{v}.json", "w"))
    print('', flush=True)
print("DONE patched objects", n_patched, round(time.time() - t00), flush=True)
