import os, sys, re, json, time, math, argparse
import numpy as np, cv2, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_common import load_split, R

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--max_frames", type=int, default=160)
ap.add_argument("--det_frames", type=int, default=16)
ap.add_argument("--videos", default="")
ap.add_argument("--video_list", default="")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--out_tag", default="v2")
ap.add_argument("--det_calib", default="")
args = ap.parse_args()
OUT = f"{R}/geo/out/tracks_{args.split}_{args.out_tag}"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

from transformers import (AutoProcessor, GroundingDinoForObjectDetection, Sam2VideoModel, Sam2VideoProcessor,
                          Owlv2ForObjectDetection, Owlv2Processor)
gp = AutoProcessor.from_pretrained(f"{R}/models/grounding-dino-base")
gm = GroundingDinoForObjectDetection.from_pretrained(f"{R}/models/grounding-dino-base").to(dev).eval()
sp = Sam2VideoProcessor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
sm = Sam2VideoModel.from_pretrained(f"{R}/models/sam2.1-hiera-large").to(dev, dtype=torch.bfloat16).eval()
op = Owlv2Processor.from_pretrained(f"{R}/models/owlv2-base-patch16-ensemble")
om = Owlv2ForObjectDetection.from_pretrained(f"{R}/models/owlv2-base-patch16-ensemble").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

import track_common as tc

COLOR_WORDS = tc.COLOR_WORDS
CONTRAST = {"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray",
            "silver", "big", "small", "large", "little", "left", "right", "upper", "lower", "top", "bottom", "front",
            "back", "near", "far", "tall", "short", "first", "second", "third", "middle", "center"}


@torch.no_grad()
def gdino(frames, text, thr=0.15, chunk=4):
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
    res = gp.post_process_grounded_object_detection(out, tok.input_ids, threshold=thr, text_threshold=thr, target_sizes=[(H, W)] * K)
    dets = []
    for r in res:
        b = r["boxes"].cpu().numpy().astype(float)
        sc = r["scores"].cpu().numpy().astype(float)
        keep = tc.nms(b, sc, 0.6)
        dets.append([(b[i].tolist(), float(sc[i])) for i in keep])
    return dets


@torch.no_grad()
def owl(frames, phrases, thr=0.05, chunk=4, topk=25):
    H, W = frames[0].shape[:2]
    S = max(H, W)
    out_all = []
    for i in range(0, len(frames), chunk):
        fr = frames[i:i + chunk]
        inp = op(text=[phrases] * len(fr), images=list(fr), return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = om(**inp)
        logits = o.logits.float().sigmoid().cpu().numpy()
        boxes = o.pred_boxes.float().cpu().numpy()
        for b in range(len(fr)):
            bx = boxes[b]
            xyxy = np.stack([(bx[:, 0] - bx[:, 2] / 2) * S, (bx[:, 1] - bx[:, 3] / 2) * S,
                             (bx[:, 0] + bx[:, 2] / 2) * S, (bx[:, 1] + bx[:, 3] / 2) * S], 1)
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
            d = {}
            for qi, ph in enumerate(phrases):
                sc = logits[b, :, qi]
                idx = np.argsort(-sc)[:topk]
                idx = idx[sc[idx] >= thr]
                keep = tc.nms(xyxy[idx], sc[idx], 0.5) if len(idx) else []
                d[ph] = [(xyxy[idx[k]].tolist(), float(sc[idx[k]])) for k in keep]
            out_all.append(d)
    return out_all


def owl_support(box, owl_dets):
    return max([s for b, s in owl_dets if tc.iou(b, box) >= 0.5] + [0.0])


def vote(s, o):
    return s + o


if args.det_calib:
    import det_calib as DC
    vote = DC.load_vote(args.det_calib)


def heads(p):
    t = re.findall(r"[a-z]+", p or "")
    return tc.sing(t[-1]) if t else ""


def contrastive(qa, qb):
    return tc.contrastive_v2(qa["phrase"], qa["where"], qb["phrase"], qb["where"])


def rel_pick(cands, idxs, rel, refb):
    if refb is None or not idxs:
        return None
    rc = np.array([(refb[0] + refb[2]) / 2, (refb[1] + refb[3]) / 2])
    best, bd = None, 1e18
    for i in idxs:
        b = cands[i][0]
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        d = float(np.linalg.norm(c - rc))
        if rel == "in":
            ok = refb[0] - 2 <= c[0] <= refb[2] + 2 and refb[1] - 2 <= c[1] <= refb[3] + 2 and tc.iou(b, refb) < 0.8
        elif rel == "right_of":
            ok = c[0] > rc[0]
        elif rel == "left_of":
            ok = c[0] < rc[0]
        elif rel == "below":
            ok = c[1] > rc[1]
        elif rel == "above":
            ok = c[1] < rc[1]
        else:
            ok = tc.iou(b, refb) < 0.8
        if ok and d < bd:
            best, bd = i, d
    return best


def choose(cands, owl_c, img, q, W, H, refb, use_owl, exclude=None):
    if not cands:
        return None
    cands = [(b, s) for b, s in cands if (b[2] - b[0]) * (b[3] - b[1]) < 0.85 * W * H and s >= (0.15 if use_owl else 0.2)]
    if exclude is not None:
        cands = [(b, s) for b, s in cands if tc.iou(b, exclude) < 0.3]
    if not cands:
        return None
    phrase, where = q["phrase"], q["where"]
    cws = [c for c in COLOR_WORDS if re.search(r"\b" + c + r"\b", phrase or "")]
    sup = [owl_support(b, owl_c) for b, s in cands]
    adj = []
    for (b, s), o in zip(cands, sup):
        a = vote(s, o) if use_owl else s
        if cws:
            fr = max(tc.color_frac(img, b, c) for c in cws)
            a = a * (0.35 + 1.3 * min(fr, 0.5))
        adj.append(a)
    adj = np.array(adj)
    best = adj.max()
    good = [i for i in range(len(cands)) if adj[i] >= max(0.15, 0.55 * best)]
    w = [x for x in where if x != "#second"]
    pick = int(np.argmax(adj))
    if w and good:
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
    rel = q.get("rel")
    if rel and refb is not None:
        rp = rel_pick(cands, good, rel[0], refb)
        if rp is not None:
            pick = rp
    if "#second" in where:
        order = [i for i in np.argsort(-adj) if i in good]
        if len(order) < 2:
            return None
        b0 = cands[order[0]][0]
        others = [i for i in order[1:] if tc.iou(cands[i][0], b0) < 0.3]
        if not others:
            return None
        pick = others[0]
    sc = adj[pick] * (1.0 if (not w or len(good) >= 2 or exclude is not None) else 0.6)
    return cands[pick][0], float(sc), len(good), float(cands[pick][1]), float(sup[pick])


def queries_for(recs):
    qs = {}
    def add(ent, role):
        if not ent or not ent.get("phrase"):
            return None
        rel = ent.get("rel")
        k = ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")
        if k not in qs:
            qs[k] = dict(key=k, phrase=ent["phrase"], where=list(ent.get("where") or []), rel=rel, roles=[])
        if role not in qs[k]["roles"]:
            qs[k]["roles"].append(role)
        if rel:
            add(dict(phrase=rel[1], where=[]), "ref")
        return k
    for r in recs:
        add(r["target"], "dist" if r["q"]["kind"] == "dist2" else "target")
        if r.get("target2"):
            add(r["target2"], "dist" if r["q"]["kind"] == "dist2" else "target2")
        if r["prior_obj"] and r["prior_obj"].get("phrase"):
            add(r["prior_obj"], "prior")
    return list(qs.values())


def select_anchors(qs, di, dimgs, dets_by_phrase, owl_by_frame, W, H, use_owl):
    ref_best = {}
    for q in qs:
        ph = q["phrase"]
        if ph not in ref_best:
            ref_best[ph] = [max(d, key=lambda x: x[1])[0] if d else None for d in dets_by_phrase[ph]]
    per_q = []
    for q in qs:
        refb = ref_best.get(q["rel"][1]) if q.get("rel") else None
        per = []
        for k in range(len(di)):
            per.append(choose(dets_by_phrase[q["phrase"]][k], owl_by_frame[k].get(q["phrase"], []), dimgs[k], q, W, H,
                              refb[k] if refb else None, use_owl))
        per_q.append(per)
    info = []
    for qi, q in enumerate(qs):
        per = per_q[qi]
        scores = [c[1] if c else 0.0 for c in per]
        kb = int(np.argmax(scores))
        info.append(dict(per=per, scores=scores, kbest=kb))
    resolved = []
    for a in range(len(qs)):
        for b in range(a + 1, len(qs)):
            ia, ib = info[a], info[b]
            if not ia["per"][ia["kbest"]] or not ib["per"][ib["kbest"]]:
                continue
            if "ref" in qs[a]["roles"] and len(qs[a]["roles"]) == 1 or "ref" in qs[b]["roles"] and len(qs[b]["roles"]) == 1:
                continue
            if not contrastive(qs[a], qs[b]):
                continue
            k = ia["kbest"]
            pa, pb = ia["per"][k], ib["per"][k]
            kk = ib["kbest"]
            pa2, pb2 = ia["per"][kk], ib["per"][kk]
            hit = (pa and pb and tc.iou(pa[0], pb[0]) > 0.7) or (pa2 and pb2 and tc.iou(pa2[0], pb2[0]) > 0.7)
            if not hit:
                continue
            fa = k if (pa and pb and tc.iou(pa[0], pb[0]) > 0.7) else kk
            box = ia["per"][fa][0]
            ev = []
            for qi in (a, b):
                g = max([s for bb, s in dets_by_phrase[qs[qi]["phrase"]][fa] if tc.iou(bb, box) > 0.7] + [0.0])
                o = owl_support(box, owl_by_frame[fa].get(qs[qi]["phrase"], [])) if use_owl else 0.0
                ev.append(vote(g, o) if use_owl else g)
            loser = b if ev[0] >= ev[1] else a
            owner = a if loser == b else b
            refb = ref_best.get(qs[loser]["rel"][1]) if qs[loser].get("rel") else None
            newper = []
            for k2 in range(len(di)):
                ob = info[owner]["per"][k2][0] if info[owner]["per"][k2] else (info[owner]["per"][info[owner]["kbest"]][0] if k2 == info[owner]["kbest"] else None)
                if ob is None:
                    ob = box if k2 == fa else None
                c = choose(dets_by_phrase[qs[loser]["phrase"]][k2], owl_by_frame[k2].get(qs[loser]["phrase"], []), dimgs[k2],
                           qs[loser], W, H, refb[k2] if refb else None, use_owl, exclude=ob)
                newper.append(c)
            sc = [c[1] if c else 0.0 for c in newper]
            info[loser] = dict(per=newper, scores=sc, kbest=int(np.argmax(sc)))
            resolved.append(dict(owner=qs[owner]["key"], loser=qs[loser]["key"], ev=[round(x, 3) for x in ev],
                                 new_best=round(max(sc), 3)))
    anchors = []
    for qi, q in enumerate(qs):
        it = info[qi]
        kb = it["kbest"]
        c = it["per"][kb]
        q_out = dict(det_scores=[round(s, 3) for s in it["scores"]],
                     det_boxes=[[round(x, 1) for x in c_[0]] if c_ else None for c_ in it["per"]],
                     det_gd=[round(c_[3], 3) if c_ else 0.0 for c_ in it["per"]],
                     det_owl=[round(c_[4], 3) if c_ else 0.0 for c_ in it["per"]],
                     det_ngood=[c_[2] if c_ else 0 for c_ in it["per"]],
                     best_score=float(it["scores"][kb]), best_gd=float(c[3]) if c else 0.0, best_owl=float(c[4]) if c else 0.0,
                     anchor=None)
        if c and it["scores"][kb] >= 0.15:
            q_out["anchor"] = [di[kb], [round(x, 1) for x in c[0]]]
            anchors.append((qi, di[kb], c[0]))
        info[qi]["out"] = q_out
    if anchors:
        tot = np.zeros(len(di))
        for qi, _, _ in anchors:
            s = np.array(info[qi]["scores"]); tot += s / max(s.max(), 1e-6)
        kc = int(np.argmax(tot))
        new = []
        for qi, fi, b in anchors:
            s = info[qi]["scores"]; bx = info[qi]["out"]["det_boxes"][kc]
            if s[kc] >= 0.5 * max(s) and s[kc] >= 0.25 and bx is not None:
                new.append((qi, di[kc], bx)); info[qi]["out"]["anchor"] = [di[kc], bx]
            else:
                new.append((qi, fi, b))
        anchors = new
    return [info[qi]["out"] for qi in range(len(qs))], anchors, resolved


@torch.no_grad()
def sam2_track(frames, objs):
    res = {}
    groups = {}
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


def main():
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{args.split}_v2.jsonl")]
    df = load_split(args.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    fps = {str(v).strip(): float(f) for v, f in zip(df.video_id, df.fps)}
    byv = {}
    for r in recs:
        byv.setdefault(r["video_id"], []).append(r)
    vids = sorted(byv)
    if args.videos:
        vids = [v for v in vids if v in args.videos.split(",")]
    if args.video_list:
        keep = set(l.strip() for l in open(args.video_list) if l.strip())
        vids = [v for v in vids if v in keep]
    vids = vids[args.shard::args.nshards]
    t00 = time.time()
    for vi, v in enumerate(vids):
        op_ = f"{OUT}/{v}.json"
        if os.path.exists(op_) and not args.overwrite:
            continue
        t0 = time.time()
        frames_all = tc.load_video(vpath[v])
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
        phrases = sorted(set(q["phrase"] for q in qs))
        dets_by_phrase = {ph: gdino(dimgs, ph.strip(" .") + " .") for ph in phrases}
        owl_by_frame = owl(dimgs, phrases)
        t_det = time.time() - t1
        cfg = {}
        for name, use_owl in (("d", False), ("e", True)):
            outs, anchors, resolved = select_anchors(qs, di, dimgs, dets_by_phrase, owl_by_frame, W, H, use_owl)
            cfg[name] = (outs, anchors, resolved)
        uniq = []
        alias = {}
        for name in ("d", "e"):
            for qi, fi, b in cfg[name][1]:
                hit = next((u for u in uniq if u[1] == fi and tc.iou(u[2], b) > 0.9), None)
                if hit is None:
                    uniq.append((len(uniq), fi, b)); alias[(name, qi)] = uniq[-1][0]
                else:
                    alias[(name, qi)] = hit[0]
        t1 = time.time()
        tr = sam2_track(frames, uniq) if uniq else {}
        t_sam = time.time() - t1
        need_poly = set()
        for name in ("d", "e"):
            for qi, fi, b in cfg[name][1]:
                if "dist" in qs[qi]["roles"]:
                    need_poly.add(alias[(name, qi)])
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
        out = dict(video_id=v, N=N, stride=stride, fidx=fidx, fps=fps[v], H=H, W=W, det_frames=di, cam=tc.camera_motion(frames),
                   objs=objs, version="v2", t_det=round(t_det, 1), t_sam=round(t_sam, 1))
        for name, key in (("e", "queries"), ("d", "queries_d")):
            outs, anchors, resolved = cfg[name]
            ql = []
            for qi, q in enumerate(qs):
                o = dict(key=q["key"], phrase=q["phrase"], where=q["where"], rel=q.get("rel"), roles=q["roles"], **outs[qi])
                o["obj"] = alias.get((name, qi))
                ious = []
                if o["obj"] is not None:
                    fr = objs[str(o["obj"])]
                    for k, fi in enumerate(di):
                        st = fr[fi]["st"]
                        dd = dets_by_phrase[q["phrase"]][k]
                        if st is None or not dd:
                            continue
                        ious.append(max(tc.iou(st["box"], b) for b, s in dd))
                o["det_iou"] = [round(x, 3) for x in ious]
                ql.append(o)
            out[key] = ql
            out["resolved_" + name] = resolved
        out["t_sec"] = round(time.time() - t0, 1)
        json.dump(out, open(op_, "w"))
        print('', flush=True)
    print("DONE", time.time() - t00, flush=True)


main()
