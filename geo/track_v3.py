#!/usr/bin/env python
import os, sys, json, math, time, argparse
import numpy as np, torch

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools")
sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--max_frames", type=int, default=160)
ap.add_argument("--det_frames", type=int, default=16)
ap.add_argument("--videos", default="")
ap.add_argument("--coords", default="rel1000")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--vg_tag", default="")
ap.add_argument("--out_tag", default="v3")
a = ap.parse_args()
OUT = f"{R}/geo/out/tracks_{a.split}_{a.out_tag}"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
from transformers import Sam2VideoModel, Sam2VideoProcessor
sp = Sam2VideoProcessor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
sm = Sam2VideoModel.from_pretrained(f"{R}/models/sam2.1-hiera-large").to(dev, dtype=torch.bfloat16).eval()


@torch.no_grad()
def sam2_track(frames, objs):
    res, groups = {}, {}
    for oid, fi, b in objs:
        groups.setdefault(fi, []).append((oid, b))
    for fi, ob in groups.items():
        sess = sp.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
        ids = [oid + 1 for oid, _ in ob]
        sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=fi, obj_ids=ids, input_boxes=[[b for _, b in ob]])
        for rev in (False, True):
            for o in sm.propagate_in_video_iterator(sess, start_frame_idx=fi, reverse=rev):
                if rev and o.frame_idx == fi:
                    continue
                m = sp.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]], binarize=False)[0]
                m = (m[:, 0] > 0).cpu().numpy()
                lg = o.object_score_logits.float().cpu().numpy().ravel()
                for j, oid in enumerate(o.object_ids):
                    if lg[j] > 0:
                        st = tc.mask_stats(m[j])
                        poly = tc.mask_poly(m[j]) if st else None
                        col = tc.mask_colors(frames[int(o.frame_idx)], m[j]) if (st and int(o.frame_idx) == fi) else None
                    else:
                        st, poly, col = None, None, None
                    res.setdefault(oid - 1, {})[int(o.frame_idx)] = (st, float(lg[j]), poly, col)
        del sess
        torch.cuda.empty_cache()
    return res


def valid_box(b, W, H):
    x0, y0, x1, y1 = b
    x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2 or (x1 - x0) * (y1 - y0) > 0.9 * W * H:
        return None
    return [x0, y0, x1, y1]


def main():
    vg = [json.loads(l) for l in open(f"{R}/geo/out/vlmground_{a.split}{a.vg_tag}.jsonl")]
    byv = {}
    for x in vg:
        byv.setdefault(x["video_id"], []).append(x)
    df = load_split(a.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    fps = {str(v).strip(): float(f) for v, f in zip(df.video_id, df.fps)}
    vids = sorted(byv)
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
    elif os.path.exists(f"{R}/geo/out/track_v3_{a.split}_videos.txt"):
        keep = set(l.strip() for l in open(f"{R}/geo/out/track_v3_{a.split}_videos.txt") if l.strip())
        vids = [v for v in vids if v in keep]
        print('', flush=True)
    t00 = time.time()
    for vi, v in enumerate(vids):
        op = f"{OUT}/{v}.json"
        if os.path.exists(op) and not a.overwrite:
            continue
        t0 = time.time()
        frames_all = tc.load_video(vpath[v])
        N = len(frames_all)
        stride = max(1, int(math.ceil(N / a.max_frames)))
        fidx = list(range(0, N, stride))
        frames = [frames_all[i] for i in fidx]
        del frames_all
        H, W = frames[0].shape[:2]
        di = np.unique(np.linspace(0, len(frames) - 1, min(a.det_frames, len(frames))).round().astype(int)).tolist()
        Q = {}
        for x in byv[v]:
            q = Q.setdefault(x["key"], dict(key=x["key"], phrase=x["phrase"], where=x["where"], roles=x["roles"],
                                            expr=x["expr"], multi=x["multi"], boxes={}))
            bl = [valid_box(b, W, H) for b in x["boxes_" + a.coords]]
            q["boxes"][x["j"]] = [b for b in bl if b is not None]
        vframes = sorted({x["j"] for x in byv[v]})
        anchors = {}
        for k, q in Q.items():
            if q["multi"]:
                continue
            bs = [(j, q["boxes"][j][0]) for j in vframes if q["boxes"].get(j)]
            if not bs:
                continue
            ar = np.array([(b[2] - b[0]) * (b[3] - b[1]) for _, b in bs])
            i = int(np.argmin(np.abs(np.log(ar / np.median(ar)))))
            anchors[k] = bs[i]
        second = {}
        for k, q in Q.items():
            if not q["multi"]:
                continue
            first = next((kk for kk, qq in Q.items() if not qq["multi"] and qq["phrase"] == q["phrase"]), None)
            best, per = None, {}
            for j in vframes:
                bl = q["boxes"].get(j) or []
                ref = (Q[first]["boxes"].get(j) or [None])[0] if first else None
                if ref is None and bl:
                    ref = bl[0]
                cand = sorted([b for b in bl if tc.iou(b, ref) < 0.3], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])) if ref else []
                per[j] = cand[0] if cand else None
                if cand:
                    sz = (cand[0][2] - cand[0][0]) * (cand[0][3] - cand[0][1])
                    if best is None or sz > best[2]:
                        best = (j, cand[0], sz)
            second[k] = per
            if best:
                anchors[k] = (best[0], best[1])
        cnt = {j: sum(1 for k in anchors if (second[k].get(j) if Q[k]["multi"] else (Q[k]["boxes"].get(j) or [None])[0]) is not None)
               for j in vframes}
        if cnt:
            jc = max(vframes, key=lambda j: (cnt[j], -abs(j - vframes[len(vframes) // 2])))
            for k in list(anchors):
                bc = second[k].get(jc) if Q[k]["multi"] else (Q[k]["boxes"].get(jc) or [None])[0]
                if bc is None:
                    continue
                b0 = anchors[k][1]
                ra = ((bc[2] - bc[0]) * (bc[3] - bc[1])) / max((b0[2] - b0[0]) * (b0[3] - b0[1]), 1e-6)
                if 0.5 <= ra <= 2.0:
                    anchors[k] = (jc, bc)
        uniq, alias = [], {}
        for k, (fi, b) in anchors.items():
            hit = next((u for u in uniq if u[1] == fi and tc.iou(u[2], b) > 0.9), None)
            if hit is None:
                uniq.append((len(uniq), fi, b)); alias[k] = uniq[-1][0]
            else:
                alias[k] = hit[0]
        t1 = time.time()
        tr = sam2_track(frames, uniq) if uniq else {}
        t_sam = time.time() - t1
        need_poly = {alias[k] for k, q in Q.items() if k in alias and "dist" in q["roles"]}
        objs = {}
        for oid, _, _ in uniq:
            d = tr.get(oid, {})
            fr = []
            for j in range(len(frames)):
                st, lg, poly, col = d.get(j, (None, -99.0, None, None))
                e = dict(st=st, lg=round(lg, 2))
                if poly is not None and oid in need_poly:
                    e["poly"] = poly
                if col is not None:
                    e["col"] = col
                fr.append(e)
            objs[str(oid)] = fr
        ql = []
        for k, q in Q.items():
            o = dict(key=k, phrase=q["phrase"], where=q["where"], rel=None, roles=q["roles"], expr=q["expr"])
            vb = second[k] if q["multi"] else {j: (q["boxes"].get(j) or [None])[0] for j in vframes}
            o["obj"] = alias.get(k)
            ious = []
            if o["obj"] is not None:
                fr = objs[str(o["obj"])]
                for j in vframes:
                    st = fr[j]["st"] if j < len(fr) else None
                    if st is None or vb.get(j) is None:
                        continue
                    ious.append(tc.iou(st["box"], vb[j]))
            agree = float(sum(x >= 0.5 for x in ious)) / max(len(vframes), 1)
            o.update(det_scores=[1.0 if vb.get(j) is not None else 0.0 for j in vframes],
                     det_boxes=[[round(c, 1) for c in vb[j]] if vb.get(j) is not None else None for j in vframes],
                     det_gd=[agree] * len(vframes), det_owl=[agree] * len(vframes), det_ngood=[1] * len(vframes),
                     best_score=agree, best_gd=agree, best_owl=agree,
                     anchor=[anchors[k][0], [round(c, 1) for c in anchors[k][1]]] if k in anchors else None,
                     det_iou=[round(x, 3) for x in ious], vlm_frames=vframes)
            ql.append(o)
        out = dict(video_id=v, N=N, stride=stride, fidx=fidx, fps=fps[v], H=H, W=W, det_frames=di, cam=tc.camera_motion(frames),
                   objs=objs, version="v2", source="vlm_qwen3vl8b", t_sam=round(t_sam, 1), queries=ql, queries_d=ql,
                   resolved_e=[], resolved_d=[])
        out["t_sec"] = round(time.time() - t0, 1)
        json.dump(out, open(op, "w"))
        print('', flush=True)
    print("DONE", round(time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
