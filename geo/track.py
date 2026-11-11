import os, sys, re, json, time, math, argparse
import numpy as np, cv2, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "tools"))
from qp_common import load_split, R

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--max_frames", type=int, default=160)
ap.add_argument("--det_frames", type=int, default=16)
ap.add_argument("--videos", default="")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
OUT = f"{R}/geo/out/tracks_{args.split}"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

from transformers import AutoProcessor, GroundingDinoForObjectDetection, Sam2VideoModel, Sam2VideoProcessor
gp = AutoProcessor.from_pretrained(f"{R}/models/grounding-dino-base")
gm = GroundingDinoForObjectDetection.from_pretrained(f"{R}/models/grounding-dino-base").to(dev).eval()
sp = Sam2VideoProcessor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
sm = Sam2VideoModel.from_pretrained(f"{R}/models/sam2.1-hiera-large").to(dev, dtype=torch.bfloat16).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

COLOR_WORDS = ["white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver"]


def load_video(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(h, w)
        if s > 480:
            f = cv2.resize(f, (int(round(w * 480 / s)), int(round(h * 480 / s))), interpolation=cv2.INTER_AREA)
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fr


@torch.no_grad()
def gdino(frames, text, thr=0.2, chunk=4):
    if len(frames) > chunk:
        out = []
        for i in range(0, len(frames), chunk):
            out += gdino(frames[i:i + chunk], text, thr, chunk)
        return out
    K = len(frames)
    H, W = frames[0].shape[:2]
    s = 800 / min(H, W)
    if max(H, W) * s > 1333:
        s = 1333 / max(H, W)
    h, w = int(round(H * s)), int(round(W * s))
    x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    x = (x - MEAN) / STD
    tok = gp.tokenizer([text] * K, return_tensors="pt", padding=True).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = gm(pixel_values=x, input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                 token_type_ids=tok.get("token_type_ids"), pixel_mask=torch.ones((K, h, w), dtype=torch.long, device=dev))
    out.logits = out.logits.float(); out.pred_boxes = out.pred_boxes.float()
    res = gp.post_process_grounded_object_detection(out, tok.input_ids, threshold=thr, text_threshold=thr,
                                                    target_sizes=[(H, W)] * K)
    dets = []
    for r in res:
        b = r["boxes"].cpu().numpy().astype(float)
        sc = r["scores"].cpu().numpy().astype(float)
        keep = nms(b, sc, 0.6)
        dets.append([(b[i].tolist(), float(sc[i])) for i in keep])
    return dets


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(b, sc, thr):
    order = np.argsort(-sc)
    keep = []
    for i in order:
        if all(iou(b[i], b[j]) < thr for j in keep):
            keep.append(i)
    return keep


def color_frac(img, box, cw):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return 0.0
    cx0, cy0, cx1, cy1 = x0 + int(0.2 * w), y0 + int(0.2 * h), x1 - int(0.2 * w), y1 - int(0.2 * h)
    patch = img[max(cy0, 0):max(cy1, 1), max(cx0, 0):max(cx1, 1)]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    m = {"white": (S < 50) & (V > 170), "black": V < 65, "grey": (S < 45) & (V >= 65) & (V <= 190),
         "gray": (S < 45) & (V >= 65) & (V <= 190), "silver": (S < 45) & (V >= 90),
         "red": ((H < 9) | (H > 170)) & (S > 90) & (V > 50), "orange": (H >= 8) & (H < 22) & (S > 100) & (V > 80),
         "yellow": (H >= 20) & (H < 36) & (S > 80) & (V > 80), "green": (H >= 36) & (H < 88) & (S > 50) & (V > 40),
         "blue": (H >= 88) & (H < 132) & (S > 60) & (V > 40), "purple": (H >= 125) & (H < 160) & (S > 40) & (V > 30),
         "pink": (((H >= 150) & (H <= 179)) | (H < 8)) & (S > 30) & (S < 160) & (V > 120),
         "brown": (H >= 5) & (H < 25) & (S > 60) & (V < 160) & (V > 30)}[cw]
    return float(m.mean())


def choose(cands, img, phrase, where, W, H):
    if not cands:
        return None
    cands = [(b, s) for b, s in cands if (b[2] - b[0]) * (b[3] - b[1]) < 0.85 * W * H]
    if not cands:
        return None
    cws = [c for c in COLOR_WORDS if re.search(r"\b" + c + r"\b", phrase or "")]
    adj = []
    for b, s in cands:
        a = s
        if cws:
            fr = max(color_frac(img, b, c) for c in cws)
            a = s * (0.35 + 1.3 * min(fr, 0.5))
        adj.append(a)
    adj = np.array(adj)
    best = adj.max()
    good = [i for i in range(len(cands)) if adj[i] >= max(0.15, 0.55 * best)]
    w = [x for x in where if x != "#second"]
    pick = int(np.argmax(adj))
    if w and len(good) >= 1:
        xc = [(cands[i][0][0] + cands[i][0][2]) / 2 for i in good]
        yc = [(cands[i][0][1] + cands[i][0][3]) / 2 for i in good]
        ar = [(cands[i][0][2] - cands[i][0][0]) * (cands[i][0][3] - cands[i][0][1]) for i in good]
        key = None
        ws = " ".join(w)
        if "left" in ws: key = np.array(xc)
        elif "right" in ws: key = -np.array(xc)
        elif re.search(r"upper|top", ws): key = np.array(yc)
        elif re.search(r"lower|bottom", ws): key = -np.array(yc)
        elif re.search(r"middle|center|centre", ws): key = np.abs(np.array(xc) - W / 2)
        elif re.search(r"big|large|biggest|largest|front|near|closest|tall", ws): key = -np.array(ar)
        elif re.search(r"small|smallest|back|far|farthest|short", ws): key = np.array(ar)
        if key is not None:
            pick = good[int(np.argmin(key))]
    if "#second" in where:
        order = [i for i in np.argsort(-adj) if i in good]
        if len(order) < 2:
            return None
        b0 = cands[order[0]][0]
        others = [i for i in order[1:] if iou(cands[i][0], b0) < 0.3]
        if not others:
            return None
        pick = others[0]
    sc = adj[pick] * (1.0 if (not w or len(good) >= 2) else 0.6)
    return cands[pick][0], float(sc), len(good)


def mask_stats(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    H, W = m.shape
    border = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= W - 1) + int(y1 >= H - 1)
    c = np.cov(pts.T) if len(xs) > 5 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    return dict(area=int(len(xs)), box=[int(x0), int(y0), int(x1), int(y1)], c=[float(xs.mean()), float(ys.mean())],
                rr=[float(max(rw, rh)), float(min(rw, rh)), float(ang)], pca=[float(4 * math.sqrt(max(ev[0], 0))), float(4 * math.sqrt(max(ev[1], 0)))],
                border=border, bottom=float(y1), top=float(y0))


@torch.no_grad()
def sam2_track(frames, anchors):
    res = {}
    groups = {}
    for oi, fi, b in anchors:
        groups.setdefault(fi, []).append((oi, b))
    for fi, objs in groups.items():
        sess = sp.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
        ids = [oi + 1 for oi, _ in objs]
        sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=fi, obj_ids=ids, input_boxes=[[b for _, b in objs]])
        for rev in (False, True):
            for o in sm.propagate_in_video_iterator(sess, start_frame_idx=fi, reverse=rev):
                if rev and o.frame_idx == fi:
                    continue
                m = sp.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]], binarize=False)[0]
                m = (m[:, 0] > 0).cpu().numpy()
                lg = o.object_score_logits.float().cpu().numpy().ravel()
                for j, oid in enumerate(o.object_ids):
                    res.setdefault(oid - 1, {})[int(o.frame_idx)] = (mask_stats(m[j]) if lg[j] > 0 else None, float(lg[j]))
        del sess
        torch.cuda.empty_cache()
    return res


def camera_motion(frames):
    g = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    out = [[0.0, 0.0, 1.0]]
    for a, b in zip(g[:-1], g[1:]):
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        out.append([2 * dx, 2 * dy, float(resp)])
    return out


def queries_for(recs):
    qs = {}
    def add(ph, where, role):
        if not ph:
            return None
        k = ph + "|" + ",".join(where)
        if k not in qs:
            qs[k] = dict(key=k, phrase=ph, where=list(where), roles=[])
        qs[k]["roles"].append(role)
        return k
    for r in recs:
        t = r["target"]; add(t["phrase"], t["where"], "target")
        if r.get("target2"):
            t2 = r["target2"]; add(t2["phrase"], t2["where"], "target2")
        po = r["prior_obj"]
        if po and po.get("phrase"):
            add(po["phrase"], po.get("where") or [], "prior")
    return list(qs.values())


def main():
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{args.split}.jsonl")]
    df = load_split(args.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    fps = {str(v).strip(): float(f) for v, f in zip(df.video_id, df.fps)}
    byv = {}
    for r in recs:
        byv.setdefault(r["video_id"], []).append(r)
    vids = sorted(byv)
    if args.videos:
        vids = [v for v in vids if v in args.videos.split(",")]
    vids = vids[args.shard::args.nshards]
    t00 = time.time()
    for vi, v in enumerate(vids):
        op = f"{OUT}/{v}.json"
        if os.path.exists(op) and not args.overwrite:
            continue
        t0 = time.time()
        frames_all = load_video(vpath[v])
        N = len(frames_all)
        stride = max(1, int(math.ceil(N / args.max_frames)))
        fidx = list(range(0, N, stride))
        frames = [frames_all[i] for i in fidx]
        del frames_all
        H, W = frames[0].shape[:2]
        qs = queries_for(byv[v])
        di = np.unique(np.linspace(0, len(frames) - 1, min(args.det_frames, len(frames))).round().astype(int)).tolist()
        dimgs = [frames[i] for i in di]
        t1 = time.time()
        dets_by_phrase = {}
        for q in qs:
            if q["phrase"] not in dets_by_phrase:
                dets_by_phrase[q["phrase"]] = gdino(dimgs, q["phrase"].strip(" .") + " .")
        t_det = time.time() - t1
        anchors, qinfo = [], []
        for qi, q in enumerate(qs):
            dets = dets_by_phrase[q["phrase"]]
            per = []
            for k, (fi, img) in enumerate(zip(di, dimgs)):
                c = choose(dets[k], img, q["phrase"], q["where"], W, H)
                per.append(c)
            scores = [c[1] if c else 0.0 for c in per]
            kbest = int(np.argmax(scores))
            q["det_scores"] = [round(s, 3) for s in scores]
            q["det_boxes"] = [[round(x, 1) for x in c[0]] if c else None for c in per]
            q["det_ngood"] = [c[2] if c else 0 for c in per]
            q["n_det_frames"] = int(sum(s >= 0.25 for s in scores))
            q["best_score"] = float(scores[kbest])
            if scores[kbest] >= 0.15:
                anchors.append((qi, di[kbest], per[kbest][0]))
                q["anchor"] = [di[kbest], [round(x, 1) for x in per[kbest][0]]]
            else:
                q["anchor"] = None
        if anchors:
            tot = np.zeros(len(di))
            for qi, _, _ in anchors:
                s = np.array(qs[qi]["det_scores"]); tot += s / max(s.max(), 1e-6)
            kc = int(np.argmax(tot))
            new = []
            for qi, fi, b in anchors:
                s = qs[qi]["det_scores"]
                if s[kc] >= 0.5 * max(s) and s[kc] >= 0.25 and qs[qi]["det_boxes"][kc] is not None:
                    new.append((qi, di[kc], qs[qi]["det_boxes"][kc])); qs[qi]["anchor"] = [di[kc], qs[qi]["det_boxes"][kc]]
                else:
                    new.append((qi, fi, b))
            anchors = new
        uniq, alias = [], {}
        for qi, fi, b in anchors:
            hit = next((u for u in uniq if u[1] == fi and iou(u[2], b) > 0.9), None)
            if hit is None:
                uniq.append((qi, fi, b)); alias[qi] = qi
            else:
                alias[qi] = hit[0]
        t1 = time.time()
        tr0 = sam2_track(frames, uniq) if uniq else {}
        tr = {qi: tr0.get(alias[qi], {}) for qi in alias}
        t_sam = time.time() - t1
        for qi, q in enumerate(qs):
            d = tr.get(qi, {})
            q["frames"] = []
            for j in range(len(frames)):
                st, lg = d.get(j, (None, -99.0))
                q["frames"].append(dict(st=st, lg=round(lg, 2)))
            ious = []
            for k, fi in enumerate(di):
                st = q["frames"][fi]["st"]
                dets = dets_by_phrase[q["phrase"]][k]
                if st is None or not dets:
                    continue
                ious.append(max(iou(st["box"], b) for b, s in dets))
            q["det_iou"] = [round(x, 3) for x in ious]
        cam = camera_motion(frames)
        out = dict(video_id=v, N=N, stride=stride, fidx=fidx, fps=fps[v], H=H, W=W, det_frames=di, queries=qs, cam=cam,
                   t_sec=round(time.time() - t0, 1), t_det=round(t_det, 1), t_sam=round(t_sam, 1))
        json.dump(out, open(op, "w"))
        print('', flush=True)
    print("DONE", time.time() - t00, flush=True)


main()
