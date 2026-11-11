#!/usr/bin/env python
import os, sys, re, json, time, math, copy, argparse, subprocess

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
OUT = f"{R}/cache/tmp/r5_tool"
LOCK = f"{R}/cache/tmp/r4_gpu.lock"
TRK = os.environ.get("R5_TOOL_TRACKS", OUT)
T5 = os.environ.get("R5_TABLES", f"{R}/cache/tmp/r5")
BABA = os.path.expanduser("~/anaconda3/envs/baba/bin/python")
MODELS = dict(owlv2=f"{R}/models/owlv2-base-patch16-ensemble", gdino=f"{R}/models/grounding-dino-base",
              sam2=f"{R}/models/sam2.1-hiera-large")
CHAIN_ENV = dict(QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1", QP_TSHIFT="1",
                 QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
                 QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
                 QP_NEAREST="1", QP_BELT="1", RL_SELF_GRAV="1", RL_SELF_GRAV_MODE="2d_gself", QP_FPERSON="1", QP_FPERSON_MINN="2")
PROD = dict(M="v2sc", V="v3ntmv", measM="meas_{s}_v2_v3m_rrj5sQ.csv", measV="meas_{s}_v2_v3v_rntrj5sQ.csv",
            final="{s}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")


def slug(p):
    return re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-")


def base_paths(split, video, base=None):
    if base:
        d = f"{TRK}/tracks_{split}_{base}"
        return f"{d}/{video}.json", f"{d}/{video}.v3ntmv.json"
    out = []
    for ver in (PROD["M"], PROD["V"]):
        for p in (f"{R}/geo/out/tracks_{split}_{ver}/{video}.json", f"{R}/geo/out/tracks_{split}_v2/{video}.json",
                  f"{R}/geo/out/tracks_{split}/{video}.json"):
            if os.path.exists(p):
                out.append(p); break
        else:
            raise FileNotFoundError('')
    return tuple(out)


def q_frames(tj, q):
    if tj.get("objs") is not None and tj.get("version") == "v2":
        return tj["objs"].get(str(q.get("obj"))) if q.get("obj") is not None else None
    return q.get("frames")


def det_csv(split, video, phrase, det):
    return f"{OUT}/det/{split}_{video}_{slug(phrase)}_{det}"


def parse_recs(split, video):
    return [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8") if f'"{video}"' in l
            and json.loads(l)["video_id"] == video]


def ent_key(e):
    if not e or not e.get("phrase"):
        return None
    rel = e.get("rel")
    return e["phrase"] + "|" + ",".join(e.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def gpu_reexec():
    if os.environ.get("R5_REGRID_GPU") == "1":
        return
    env = dict(os.environ, R5_REGRID_GPU="1", PYTHONNOUSERSITE="1")
    cmd = ["flock", LOCK, BABA, os.path.abspath(__file__)] + sys.argv[1:]
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, timeout=600).returncode
    print('')
    sys.exit(rc)


def load_frames(split, video, W, H, fidx):
    sys.path.insert(0, f"{R}/geo")
    import track_common as tc
    import cv2
    cands = ([f"{R}/data/test/{video}.mp4", f"{R}/data/test_480p/{video}.mp4"] if split == "test"
             else [f"{R}/data/val/validation_videos/{video}.mp4"])
    best = None
    for p in cands:
        if not os.path.exists(p):
            continue
        fr = tc.load_video(p)
        if fr and fr[0].shape[1] == W and fr[0].shape[0] == H:
            best = (p, fr); break
        if best is None and fr:
            best = (p, fr)
    p, fr = best
    if fr[0].shape[1] != W or fr[0].shape[0] != H:
        fr = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in fr]
    return p, [fr[i] for i in fidx if i < len(fr)]


def run_detector(det, frames, phrase, thr, topk):
    import numpy as np, torch
    import torch.nn.functional as F
    sys.path.insert(0, f"{R}/geo")
    import track_common as tc
    dev = "cuda"
    out = []
    if det == "owlv2":
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        op = Owlv2Processor.from_pretrained(MODELS["owlv2"])
        om = Owlv2ForObjectDetection.from_pretrained(MODELS["owlv2"]).to(dev).eval()
        H, W = frames[0].shape[:2]; S = max(H, W)
        with torch.no_grad():
            for f in frames:
                inp = op(text=[[phrase]], images=[f], return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = om(**inp)
                sc = o.logits.float().sigmoid().cpu().numpy()[0, :, 0]
                bx = o.pred_boxes.float().cpu().numpy()[0]
                xyxy = np.stack([(bx[:, 0] - bx[:, 2] / 2) * S, (bx[:, 1] - bx[:, 3] / 2) * S,
                                 (bx[:, 0] + bx[:, 2] / 2) * S, (bx[:, 1] + bx[:, 3] / 2) * S], 1)
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
                idx = np.argsort(-sc)[:50]; idx = idx[sc[idx] >= thr]
                keep = tc.nms(xyxy[idx], sc[idx], 0.5) if len(idx) else []
                out.append([(xyxy[idx[k]].tolist(), float(sc[idx[k]])) for k in keep][:topk])
    else:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        gp = AutoProcessor.from_pretrained(MODELS["gdino"])
        gm = GroundingDinoForObjectDetection.from_pretrained(MODELS["gdino"]).to(dev).eval()
        MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        text = phrase.strip(" .") + " ."
        with torch.no_grad():
            for f in frames:
                H, W = f.shape[:2]
                s = 800 / min(H, W)
                if max(H, W) * s > 1333:
                    s = 1333 / max(H, W)
                h, w = int(round(H * s)), int(round(W * s))
                x = torch.from_numpy(f[None]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
                x = (x - MEAN) / STD
                tok = gp.tokenizer([text], return_tensors="pt", padding=True).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = gm(pixel_values=x, input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                           token_type_ids=tok.get("token_type_ids"), pixel_mask=torch.ones((1, h, w), dtype=torch.long, device=dev))
                o.logits = o.logits.float(); o.pred_boxes = o.pred_boxes.float()
                r = gp.post_process_grounded_object_detection(o, tok.input_ids, threshold=thr, text_threshold=thr, target_sizes=[(H, W)])[0]
                b = r["boxes"].cpu().numpy().astype(float); sc = r["scores"].cpu().numpy().astype(float)
                keep = tc.nms(b, sc, 0.5) if len(sc) else []
                keep = sorted(keep, key=lambda i: -sc[i])
                out.append([(b[i].tolist(), float(sc[i])) for i in keep][:topk])
    return out


def tile(imgs, ncol, tw):
    import numpy as np, cv2
    th = int(round(imgs[0].shape[0] * tw / imgs[0].shape[1]))
    rs = [cv2.resize(i, (tw, th), interpolation=cv2.INTER_AREA) for i in imgs]
    while len(rs) % ncol:
        rs.append(np.zeros_like(rs[0]))
    rows = [np.concatenate(rs[i:i + ncol], 1) for i in range(0, len(rs), ncol)]
    return np.concatenate(rows, 0)


COLS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255), (0, 220, 220), (255, 128, 0),
        (160, 80, 255), (128, 255, 128), (255, 255, 255)]


def cmd_detect(a):
    gpu_reexec()
    import numpy as np, cv2, pandas as pd
    t0 = time.time()
    pm, _ = base_paths(a.split, a.video)
    tj = json.load(open(pm))
    fidx = tj["fidx"]
    if a.frames:
        want = [int(x) for x in a.frames.split(",")]
        ks = sorted({int(np.argmin([abs(f - w) for f in fidx])) for w in want})
    else:
        ks = sorted(set(np.linspace(0, len(fidx) - 1, 5).round().astype(int).tolist()))
    vp, frames = load_frames(a.split, a.video, tj["W"], tj["H"], fidx)
    sel = [frames[k] for k in ks]
    t1 = time.time()
    thr = a.thr if a.thr is not None else (0.05 if a.det == "owlv2" else 0.15)
    dets = run_detector(a.det, sel, a.phrase, thr, a.topk)
    t2 = time.time()
    sys.path.insert(0, f"{R}/geo")
    import track_common as tc
    rows, ims, crops = [], [], []
    for k, img, dd in zip(ks, sel, dets):
        im = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        for i, (b, s) in enumerate(dd):
            cf = sorted(((tc.color_frac(img, b, c), c) for c in tc.COLOR_WORDS if c != "gray"), reverse=True)
            rows.append(dict(split=a.split, video=a.video, phrase=a.phrase, det=a.det, frame=fidx[k], k=k, idx=i, score=round(s, 4),
                             x0=round(b[0], 1), y0=round(b[1], 1), x1=round(b[2], 1), y1=round(b[3], 1),
                             col1=cf[0][1], col1f=round(cf[0][0], 2), col2=cf[1][1], col2f=round(cf[1][0], 2)))
            bw, bh = b[2] - b[0], b[3] - b[1]; mg = 0.3 * max(bw, bh, 8)
            X0, Y0 = int(max(0, b[0] - mg)), int(max(0, b[1] - mg))
            X1, Y1 = int(min(img.shape[1], b[2] + mg)), int(min(img.shape[0], b[3] + mg))
            if X1 - X0 >= 2 and Y1 - Y0 >= 2:
                cr = cv2.cvtColor(np.ascontiguousarray(img[Y0:Y1, X0:X1]), cv2.COLOR_RGB2BGR)
                sc_ = 150.0 / max(cr.shape[:2]); cr = cv2.resize(cr, (max(1, int(cr.shape[1] * sc_)), max(1, int(cr.shape[0] * sc_))))
                cv2.rectangle(cr, (int((b[0] - X0) * sc_), int((b[1] - Y0) * sc_)), (int((b[2] - X0) * sc_), int((b[3] - Y0) * sc_)), (0, 255, 255), 1)
                pad = np.zeros((172, 150, 3), np.uint8); pad[22:22 + cr.shape[0], :cr.shape[1]] = cr
                cv2.putText(pad, f"F{fidx[k]}:{i} {s:.2f}", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                crops.append(pad)
            c = COLS[i % len(COLS)][::-1]
            p0, p1 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            cv2.rectangle(im, p0, p1, c, 2)
            cv2.putText(im, f"{i}:{s:.2f}", (p0[0] + 2, max(p0[1] + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(im, f"{i}:{s:.2f}", (p0[0] + 2, max(p0[1] + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(im, f"frame {fidx[k]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, f"frame {fidx[k]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ims.append(im)
    os.makedirs(f"{OUT}/det", exist_ok=True)
    stem = det_csv(a.split, a.video, a.phrase, a.det)
    pd.DataFrame(rows).to_csv(stem + ".csv", index=False)
    cv2.imwrite(stem + ".jpg", tile(ims, 3 if len(ims) > 4 else len(ims), 560), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if crops:
        cv2.imwrite(stem + "_crops.jpg", tile(crops, min(8, len(crops)), 150), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('')
    for k in ks:
        print(f"  frame {fidx[k]:4d}: " + "  ".join(f"{r['idx']}:{r['score']:.2f}[{r['x0']:.0f},{r['y0']:.0f},{r['x1']:.0f},{r['y1']:.0f}]{r['col1']}{r['col1f']:.2f}"
                                               for r in rows if r["k"] == k))
    print('')


def frame_rec(m, lg, img, with_col):
    sys.path.insert(0, f"{R}/geo")
    import track_common as tc
    if lg <= 0 or m is None:
        return dict(st=None, lg=round(float(lg), 2))
    st = tc.mask_stats(m)
    e = dict(st=st, lg=round(float(lg), 2))
    if st:
        e["poly"] = tc.mask_poly(m)
        if with_col:
            e["col"] = tc.mask_colors(img, m)
    return e


def cmd_track(a):
    gpu_reexec()
    import numpy as np, cv2, torch, pandas as pd
    t0 = time.time()
    pm, pv = base_paths(a.split, a.video, a.base)
    tjm, tjv = json.load(open(pm)), json.load(open(pv))
    assert tjm["fidx"] == tjv["fidx"]
    fidx = tjm["fidx"]
    D = pd.read_csv(det_csv(a.split, a.video, a.phrase, a.det) + ".csv")
    picks = []
    for tok in a.pick.split(","):
        F_, I_ = [int(x) for x in tok.split(":")]
        r = D[(D.frame == F_) & (D.idx == I_)]
        assert len(r) == 1
        r = r.iloc[0]
        picks.append(dict(frame=int(F_), k=int(r.k), idx=int(I_), score=float(r.score), box=[float(r.x0), float(r.y0), float(r.x1), float(r.y1)]))
    picks.sort(key=lambda p: p["k"])
    vp, frames = load_frames(a.split, a.video, tjm["W"], tjm["H"], fidx)
    assert len(frames) == len(fidx)
    t1 = time.time()
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    dev = "cuda"
    torch.manual_seed(0)
    sp = Sam2VideoProcessor.from_pretrained(MODELS["sam2"])
    sm = Sam2VideoModel.from_pretrained(MODELS["sam2"]).to(dev, dtype=torch.bfloat16).eval()
    t2 = time.time()
    masks, logits = {}, {}
    with torch.no_grad():
        sess = sp.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
        for p in picks:
            sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=p["k"], obj_ids=[1], input_boxes=[[p["box"]]])
        k0 = picks[len(picks) // 2]["k"]
        for rev in (False, True):
            for o in sm.propagate_in_video_iterator(sess, start_frame_idx=k0, reverse=rev):
                if rev and o.frame_idx == k0:
                    continue
                m = sp.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]], binarize=False)[0]
                masks[int(o.frame_idx)] = (m[0, 0] > 0).cpu().numpy()
                logits[int(o.frame_idx)] = float(o.object_score_logits.float().cpu().numpy().ravel()[0])
        del sess
        torch.cuda.empty_cache()
    t3 = time.time()
    pk = {p["k"] for p in picks}
    newm = [frame_rec(masks.get(k), logits.get(k, -99.0), frames[k], k in pk) for k in range(len(fidx))]
    if (tjv["W"], tjv["H"]) != (tjm["W"], tjm["H"]):
        newv = []
        for k in range(len(fidx)):
            mk = masks.get(k)
            mv = None if mk is None else cv2.resize(mk.astype(np.uint8), (tjv["W"], tjv["H"]), interpolation=cv2.INTER_NEAREST) > 0
            img = cv2.resize(frames[k], (tjv["W"], tjv["H"]), interpolation=cv2.INTER_AREA) if k in pk else None
            newv.append(frame_rec(mv, logits.get(k, -99.0), img, k in pk))
    else:
        newv = copy.deepcopy(newm)
    recs = parse_recs(a.split, a.video)
    obj = a.obj.strip().lower()
    keys = set(x for x in (a.keys or "").split(";") if x)
    ents = {}
    for r in recs:
        for role in ("target", "target2", "prior_obj"):
            e = r.get(role)
            if e and (e.get("phrase") or "").strip().lower() == obj:
                ents.setdefault(ent_key(e), (e, role))
    p0 = picks[len(picks) // 2]
    meta = dict(split=a.split, video=a.video, video_file=vp, phrase=a.phrase, det=a.det, obj=a.obj, keys=sorted(keys), picks=picks,
                base=a.base, tag=a.out, models=MODELS, n_frames=len(fidx), n_mask=sum(1 for e in newm if e["st"]))
    changes = {}
    rg = dict(tag=a.out, det=a.det, picks=[f"{p['frame']}:{p['idx']}" for p in picks])
    for name, tj, fr in (("M", tjm, newm), ("V", tjv, newv)):
        v2 = tj.get("objs") is not None and tj.get("version") == "v2"
        oid = str(max([int(x) for x in tj["objs"]] + [-1]) + 1) if v2 else None
        if v2:
            tj["objs"][oid] = fr
        upd = (dict(obj=int(oid), best_gd=0.5, best_owl=0.5) if v2 else dict(frames=fr))
        upd.update(best_score=0.5, det_iou=[1.0], anchor=[p0["k"], p0["box"]], regrid=rg)
        ch = []
        for qk in (("queries", "queries_d") if v2 else ("queries",)):
            hit = False
            for q in tj.get(qk, []):
                if (keys and q.get("key") in keys) or (not keys and (q.get("phrase") or "").strip().lower() == obj):
                    ch.append(dict(list=qk, key=q.get("key"), old_obj=q.get("obj", "v1"), new_obj=oid or "v1"))
                    q.update(copy.deepcopy(upd))
                    hit = True
            if not hit:
                for k_, (e, role) in ents.items():
                    if keys and k_ not in keys:
                        continue
                    nq = dict(key=k_, phrase=e["phrase"], where=list(e.get("where") or []), rel=e.get("rel"), roles=[role],
                              det_scores=[], det_boxes=[], det_gd=[], det_owl=[], det_ngood=[])
                    nq.update(copy.deepcopy(upd)); nq["regrid"] = dict(rg, added=True)
                    tj.setdefault(qk, []).append(nq)
                    ch.append(dict(list=qk, key=k_, old_obj=None, new_obj=oid or "v1", added=True))
        changes[name] = ch
    meta["changes"] = changes
    od = f"{OUT}/tracks_{a.split}_{a.out}"
    os.makedirs(od, exist_ok=True)
    json.dump(tjm, open(f"{od}/{a.video}.json", "w"))
    json.dump(tjv, open(f"{od}/{a.video}.v3ntmv.json", "w"))
    rows = []
    for k, e in enumerate(newm):
        st = e["st"]
        d = dict(k=k, frame=fidx[k], t=round(fidx[k] / float(tjm["fps"]), 4), lg=e["lg"], ok=int(st is not None))
        if st:
            b = st["box"]
            d.update(area=st["area"], x0=b[0], y0=b[1], x1=b[2], y1=b[3], box_w=b[2] - b[0], box_h=b[3] - b[1],
                     rr_long=round(st["rr"][0], 2), rr_short=round(st["rr"][1], 2), rr_angle=round(st["rr"][2], 2),
                     cx=round(st["c"][0], 2), cy=round(st["c"][1], 2), border=st["border"])
        rows.append(d)
    E = pd.DataFrame(rows)
    E.to_csv(f"{od}/{a.video}_extents.csv", index=False)
    tjm0 = json.load(open(pm))
    oldk = next((c["key"] for c in changes["M"] if not c.get("added")), None)
    oldq = next((q for q in tjm0["queries"] if q.get("key") == oldk), None)
    oldfr = q_frames(tjm0, oldq) if oldq else None
    ks = sorted(set(np.linspace(0, len(fidx) - 1, 12).round().astype(int).tolist()) | pk)
    if len(ks) > 12:
        extra = [k for k in ks if k not in pk]
        drop = sorted(extra, key=lambda k: min(abs(k - q) for q in pk))[:len(ks) - 12]
        ks = [k for k in ks if k not in drop]
    ims = []
    for k in ks:
        im = cv2.cvtColor(frames[k].copy(), cv2.COLOR_RGB2BGR)
        if masks.get(k) is not None and logits.get(k, -1) > 0:
            ov = im.copy(); ov[masks[k]] = (0.45 * np.array([0, 255, 0]) + 0.55 * ov[masks[k]]).astype(np.uint8); im = ov
            b = newm[k]["st"]["box"] if newm[k]["st"] else None
            if b:
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        if oldfr and k < len(oldfr) and oldfr[k] and oldfr[k].get("st"):
            b = oldfr[k]["st"]["box"]; cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        lab = f"F{fidx[k]} lg{logits.get(k, -99):.1f}" + (" PROMPT" if k in pk else "")
        cv2.putText(im, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(im, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        ims.append(im)
    cv2.imwrite(f"{od}/{a.video}_sheet.jpg", tile(ims, 4, 420), [cv2.IMWRITE_JPEG_QUALITY, 80])
    meta.update(t_decode=round(t1 - t0, 1), t_load=round(t2 - t1, 1), t_sam=round(t3 - t2, 1), t_total=round(time.time() - t0, 1))
    json.dump(meta, open(f"{od}/{a.video}_meta.json", "w"), indent=1)
    ok = E[E.ok == 1]
    nb = ok[ok.border == 0] if len(ok) else ok
    print('')
    if len(nb):
        print('')
    for n_, ch in changes.items():
        print(f"  {n_} ({PROD[n_] if not a.base else 'base ' + a.base}): " + ("; ".join(f"{c['list']} {c['key']!r} obj {c['old_obj']}->{c['new_obj']}" + (" (added)" if c.get("added") else "") for c in ch) or "NO QUERY CHANGED"))
    print('')


def mv_setup(split, video):
    for k, v in CHAIN_ENV.items():
        os.environ[k] = v
    sys.path.insert(0, f"{R}/geo")
    import numpy as np
    import measure_v3 as MV
    MV.FIX = set("abcdef"); MV.MOGE_OPTS = {"fbrej"}; MV.HALF_V, MV.HALF_A = 0.25, 0.4
    MV.FOCAL_MIX = ""; MV.PARSE_TAG = ""; MV.AGG3D_P = None; MV.STATIC_FRAC = 0.25; MV.MEASURE_SCENE = False
    MV.WINGED_LEN_SHORT = False
    p = f"{R}/geo/out/moge_{split}/{video}.npz"
    if os.path.exists(p):
        z = np.load(p)
        MV.MOGE[video] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
    fn_ = f"{R}/geo/out/scene_ruler_{split}.json"
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
    fp_ = f"{R}/geo/out/person_focal_{split}.json"
    if os.path.exists(fp_):
        MV.FPERSON.update(json.load(open(fp_)))
    MV.M1.SCALE_DEPTH = True
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))["T_owl"]
    fmed = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
    return MV, fmed


def mv_records(MV, split, video):
    sh = f"{OUT}/shadow/{split}_{video}"
    os.makedirs(f"{sh}/geo/out/tracks_{split}_r5orig", exist_ok=True)
    with open(f"{sh}/geo/out/parse_{split}_v2.jsonl", "w", encoding="utf-8") as fo:
        for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
            if f'"{video}"' in l and json.loads(l)["video_id"] == video:
                fo.write(l)
    import shutil
    shutil.copy(base_paths(split, video)[0], f"{sh}/geo/out/tracks_{split}_r5orig/{video}.json")
    R0, V0 = MV.R, MV.TRACK_VER
    try:
        MV.R, MV.TRACK_VER = sh, "r5orig"
        byv, vjs = MV.load_split_tracks(split, "e")
    finally:
        MV.R, MV.TRACK_VER = R0, V0
    return byv[video]


def mv_measure(MV, fmed, rs, vj, obj=None):
    import numpy as np
    vj = copy.deepcopy(vj)
    note = ""
    if obj:
        new = [q for q in vj["queries"] if (q.get("regrid") or {}).get("tag")]
        if new:
            have = {q["key"] for q in vj["queries"]}
            for r in rs:
                for role in ("target", "target2", "prior_obj"):
                    e = r.get(role); k = ent_key(e)
                    if k and k not in have and (e.get("phrase") or "").strip().lower() == obj:
                        vj["queries"].append(dict(new[0], key=k, phrase=e["phrase"], where=list(e.get("where") or []), rel=e.get("rel")))
                        have.add(k); note += 't81'
    allf = float(np.median(list(fmed.values())))
    fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * vj["W"] for r in rs}
    o, V = MV.measure_video(copy.deepcopy(rs), vj, fdd, "e")
    out = {}
    for x, r in zip(o, rs):
        u = r["q"]["unit"]
        val = x["val_si"] / MV.UNIT_SI[u] if x.get("ok") else float("nan")
        why = x.get("why")
        if x.get("ok"):
            kk = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(u, 2 if "^2" in u else 1)
            lo, hi = MV.PLAUS[kk]
            if not (lo <= x["val_si"] <= hi):
                val, why = float("nan"), f"implausible({x['val_si']:.3g})"
        out[r["id"]] = dict(value=val, mode=x.get("mode"), why=why, f=x.get("f"))
    return out, note


def same(a, b, tol=1e-6):
    import numpy as np
    if not (np.isfinite(a) or np.isfinite(b)):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(b)))


def cmd_measure(a):
    import numpy as np, pandas as pd
    MV, fmed = mv_setup(a.split, a.video)
    rs = mv_records(MV, a.split, a.video)
    pm, pv = base_paths(a.split, a.video)
    res = {"origM": mv_measure(MV, fmed, rs, json.load(open(pm)))[0], "origV": mv_measure(MV, fmed, rs, json.load(open(pv)))[0]}
    notes = {}
    obj = None
    if a.tracks and a.tracks != "orig":
        nm, nv = base_paths(a.split, a.video, a.tracks)
        meta = json.load(open(f"{TRK}/tracks_{a.split}_{a.tracks}/{a.video}_meta.json"))
        obj = meta["obj"].strip().lower()
        res["newM"], notes["M"] = mv_measure(MV, fmed, rs, json.load(open(nm)), obj)
        res["newV"], notes["V"] = mv_measure(MV, fmed, rs, json.load(open(nv)), obj)
    fin = pd.read_csv(f"{R}/preds/" + PROD["final"].format(s=a.split)).set_index("Unnamed: 0")
    mM = pd.read_csv(f"{R}/geo/out/" + PROD["measM"].format(s=a.split)).set_index("id")
    mVv = pd.read_csv(f"{R}/geo/out/" + PROD["measV"].format(s=a.split)).set_index("id")
    want = set(int(x) for x in a.rows.split(",")) if a.rows else None
    rows = []
    for r in rs:
        i = r["id"]
        if want and i not in want:
            continue
        d = dict(id0=i, question=r["question"], unit=r["q"]["unit"], prod=float(fin.geo_value.get(i, np.nan)),
                 prodM=float(mM.value.get(i, np.nan)), prodV=float(mVv.value.get(i, np.nan)))
        for k_, o in res.items():
            d[k_] = o[i]["value"]; d[k_ + "_mode"] = o[i]["mode"]; d[k_ + "_why"] = o[i]["why"]
        fM, fV, fP = (bool(np.isfinite(d[c])) for c in ("prodM", "prodV", "prod"))
        d["follows"] = ("M" if fM and same(d["prod"], d["prodM"]) else "V" if fV and not fM and same(d["prod"], d["prodV"])
                        else "Vx" if fV and same(d["prod"], d["prodV"]) else "nan" if not fP else "post")
        if "newM" in res:
            nM, nV = d["newM"], d["newV"]
            dvw = nM if np.isfinite(nM) else nV
            d["chain_new"] = nV if d["follows"] == "Vx" and np.isfinite(nV) else dvw
            d["chain_rule"] = ('t80' if d["follows"] == "Vx" and np.isfinite(nV)
                               else "d_vw" + ('t82' if d["follows"] == "post" else ""))
        rows.append(d)
    D = pd.DataFrame(rows)
    of = f"{OUT}/measure_{a.split}_{a.video}_{a.tracks or 'orig'}.csv"
    D.to_csv(of, index=False)




def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("detect", "track", "measure"):
        s = sub.add_parser(n)
        s.add_argument("--split", required=True, choices=["test", "val"])
        s.add_argument("--video", required=True)
        if n in ("detect", "track"):
            s.add_argument("--phrase", required=True)
            s.add_argument("--det", default="owlv2", choices=["owlv2", "gdino"])
        if n == "detect":
            s.add_argument("--frames", default="")
            s.add_argument("--topk", type=int, default=8)
            s.add_argument("--thr", type=float, default=None)
        if n == "track":
            s.add_argument("--pick", required=True)
            s.add_argument("--obj", required=True)
            s.add_argument("--out", required=True)
            s.add_argument("--keys", default="")
            s.add_argument("--base", default=None)
        if n == "measure":
            s.add_argument("--tracks", default="orig")
            s.add_argument("--rows", default="")
    a = ap.parse_args()
    dict(detect=cmd_detect, track=cmd_track, measure=cmd_measure)[a.cmd](a)


if __name__ == "__main__":
    main()
