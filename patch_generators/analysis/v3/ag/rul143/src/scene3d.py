"""Scene-level 3D measurement for scene-geometry questions (plan A): heights above surfaces, clear heights, distances
between things, widths on the ground.

The question names the parts (rules in plan()). A VLM (Qwen3-VL-32B from the team model folder) boxes each part on the
frame the question refers to, and the objects with stated distances on their frames; SAM 2 turns the boxes into masks;
MoGe-2 gives the point map (with the prior-calibrated focal). Metres come from the stated distances (an object's stated
distance / its MoGe distance), up from the floor plane or Manhattan surfaces (as in prim_fit2), and the quantity is
measured on the metric points.

  CUDA_VISIBLE_DEVICES=<H100> envs/quantiphy/bin/python scripts/ours/scene3d.py ground   -> geo/out/scene3d_val/ground.json
  CUDA_VISIBLE_DEVICES=<H100> envs/quantiphy/bin/python scripts/ours/scene3d.py geom     -> geo/out/scene3d_val/<qid>.npz
  envs/quantiphy/bin/python scripts/ours/scene3d.py measure                              -> geo/out/scene3d_val.csv
labels are read only for the printed evaluation
"""
import argparse
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # bit-reproducible cuBLAS (greedy VLM boxes flipped digits between identical calls without it)

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["jobs", "ground", "geom", "measure"])
ap.add_argument("--split", default="val")
ap.add_argument("--rows", default="")
ap.add_argument("--vlm", default=os.path.join(os.environ.get("QP_ROOT", "."), "models/Qwen3-VL-32B-Instruct"))
ap.add_argument("--fov", default="moge", choices=["calib", "moge"], help="MoGe gets the prior-calibrated FOV, or estimates its own")
ap.add_argument("--tag", default="")
ap.add_argument("--max_spread", type=float, default=1.35, help="defer when the stated distances disagree on the scale by more")
ap.add_argument("--sizes", type=int, default=1, help="also measure every object-size question (VLM end points, 3D)")
ap.add_argument("--scale", default="v2", choices=["v2", "v3"],
                help="stated-distance scale: v2 median of the agreeing entries (the reported runs); v3 area-weighted (hurt 1128, 3049-3051)")
a = ap.parse_args()
S = a.split
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = f"{R}/geo/out/scene3d_{S}"
GEO = f"{R}/geo/out/scene3d_{S}{a.tag}"
os.makedirs(GEO, exist_ok=True)
SURF = r"(?:the )?(?:water surface|water|ground|floor|road|road surface|sidewalk|table|desk|tabletop)"
EXTENDED = r"\b(wall|pier|house|building|door|fence|shelf|cabin|bridge|stairs|pillar|post|tree|road|lane)\b"
TH = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def plan(question):
    """measurement type and the parts it needs, from the question text"""
    ql = question.lower().strip()
    m = re.search(r"height of (?:the )?(.+?) above (?:the )?(.+?)(?: in | at |\?|$)", ql)
    if m:
        return {"type": "h_above", "x": m.group(1), "y": m.group(2)}
    m = re.search(r"clear height from (?:the )?(.+?) to the underside of (?:the )?(.+?)(?: \(| in |\?|$)", ql)
    if m:
        return {"type": "h_under", "x": m.group(2), "y": m.group(1)}
    m = re.search(r"minimum distance from (?:the )?(.+?) to (?:the )?(.+?)(?: in |\?|$)", ql)
    if m:
        return {"type": "d_min", "x": m.group(1), "y": m.group(2)}
    horiz = "horizontal" in ql
    m = re.search(r"distance between (?:the )?two (.+?)(?: in the .+?)?(?: at | in |\?|$)", ql)
    if m:                                   # two instances of one kind: centre to centre
        return {"type": "d_two", "x": "two " + m.group(1), "y": None, "horiz": horiz}
    m = re.search(r"distance(?:\s*\([^)]*\))? (between|from) (?:the )?(.+?) (?:and|to) (?:the )?(.+?)(?: at | in |\?|$)", ql)
    if m:
        how, x, y = m.group(1), m.group(2), m.group(3)
        if re.fullmatch(SURF, y):
            return {"type": "h_gap", "x": x, "y": y}
        if "camera" in y:
            return None
        # "from X to Y" or a large scene element: closest points; "between X and Y" of things: centres
        ext = re.search(EXTENDED, x) or re.search(EXTENDED, y)
        return {"type": "d_min" if how == "from" or ext else "d_ctr", "x": x, "y": y, "horiz": horiz}
    m = re.search(r"width of (?:the )?([\w\s-]*?\b(?:lane|road|street|path|sidewalk)s?\b[\w\s-]*?)(?: in |\?|$)", ql)   # \b: not "plane\"
    if m:
        return {"type": "w_ground", "x": m.group(1).strip(), "y": None}
    m = re.search(r"side length of (?:the )?(.+?tiles?)(?: in |\?|$)", ql)
    if m:
        return {"type": "tile", "x": "one " + re.sub(r"tiles$", "tile", m.group(1)), "y": None}
    return None


def video_path(vid):
    for p in (f"{R}/data/{S}/validation_videos/{vid}.mp4", f"{R}/data/{S}/validation_videos/ {vid}.mp4",
              f"{R}/data/test_480p/{vid}.mp4", f"{R}/data/test/{vid}.mp4"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(vid)


STRIP = r"\b(?:lanes?|roads?|streets?|paths?|sidewalks?|crosswalks?|tracks?|lines?|stripes?)\b"


def geo_table():
    """the chain's per-row table (meas_mpp): the core variant on val; on test, which has no core-variant run, the recipe's own"""
    p = f"{R}/preds/{S}_geo_v4ntrj5sQ2csrngsr_d_vw_raw.csv"
    if not os.path.exists(p):
        p = f"{R}/preds/{S}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv"
    return pd.read_csv(p).set_index("Unnamed: 0")


def frame_at(vid, t):
    cap = cv2.VideoCapture(video_path(vid))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = int(np.clip(round(t * fps), 0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    return (cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None), i, fps, n


def key_phrase(k):
    k = re.sub(r"_+", " ", k).strip()
    k = re.sub(r"^(the )?distance ", "", k)
    return k


def jobs():
    """per question: plan, query time, and the (time, phrase) pairs to ground"""
    P = {json.loads(x)["id"]: json.loads(x) for x in open(f"{R}/geo/out/parse_{S}_v2.jsonl")}
    geo = geo_table()
    rows_only = {int(x) for x in a.rows.split(",") if x}
    out = []
    for qid, r in sorted(P.items()):
        if rows_only and qid not in rows_only:
            continue
        pl = plan(r["question"])
        if pl is None and r["q"].get("kind") == "size" and (r.get("target") or {}).get("phrase") and a.sizes:
            pl = {"type": "size", "x": r["target"]["phrase"], "y": None, "dim": r["q"].get("dim") or "size"}   # any object size
        if pl is None or (r["q"]["kind"] == "size" and pl["type"] not in ("h_under", "w_ground", "tile", "size")):
            continue
        cap = cv2.VideoCapture(video_path(r["video_id"]))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tend = (n - 1) / fps
        tq, when = r["q"].get("t"), r["q"].get("when")
        tq = min(float(tq), tend) if tq is not None else (0.0 if when == "start" else tend if when == "end" else 0.5 * tend)
        want = {(round(tq, 3), p) for p in (pl["x"], pl["y"]) if p}
        size_ts = [round(tq, 3)]
        if pl["type"] == "size":                    # an object's size: measured on three frames, median
            size_ts = sorted({round(0.25 * tend, 3), round(tq, 3), round(0.75 * tend, 3)})
            want |= {(t_, pl["x"]) for t_ in size_ts}
        for e in r.get("depth") or []:
            if e.get("d") is None:
                continue
            te = round(min(float(e["t"]), tend), 3) if e.get("t") is not None else round(tq, 3)
            want.add((te, key_phrase(e["key"])))
        # no stated distance (2D videos): the prior gives the metres - its own size, or the teammates' metres per pixel
        pr = (r.get("priors") or [{}])[0]
        pph = (r.get("prior_obj") or {}).get("phrase")
        if not r.get("depth") and pph:
            want.add((round(tq, 3), pph))
        mpp = geo.meas_mpp.get(qid)
        out.append({"id": qid, "video": r["video_id"], "vt": r["video_type"][1] + "D", "question": r["question"], "plan": pl, "tq": round(tq, 3),
                    "size_ts": size_ts,
                    "t_asked": r["q"].get("t"), "tend": tend, "want": sorted(want), "unit": r["q"].get("unit"),
                    "depth": [{"phrase": key_phrase(e["key"]), "t": e.get("t"), "d": e["d"]} for e in r.get("depth") or [] if e.get("d") is not None],
                    "prior": {"phrase": pph, "kind": pr.get("kind"), "value": pr.get("value_si"), "dim": pr.get("dim"),
                              "raw": pr.get("raw"), "mpp": float(mpp) if mpp is not None and np.isfinite(mpp) else None}})
    return out


DIMTXT = {"height": "its highest point and its lowest point (top and bottom)",
          "length": "its two ends along its length",
          "width": "its two sides across its width",
          "diameter": "two opposite points on its outline, through its centre",
          "thickness": "its two faces across its thickness",
          "wingspan": "the tips of its two wings",
          "side": "the two ends of one side"}


def span_of(dim, x):
    """the end points the VLM marks for an object dimension"""
    return f"the {dim} of the {x}: {DIMTXT.get(dim, 'the two ends of exactly the length the question asks about')}"


PROMPT = (
    "This image is the frame at t={t:.2f} s of a short video. The video comes with this physics question:\n"
    "\"{context}\"\n"
    "Locate the {what} in this image. If several things could match, choose the one the question means. "
    "For a surface or region (water, ground, floor, a lane), box the part of it that the question is about. "
    "Output only JSON: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<short name>\"}}]"
)


PLAIN_JSON = 'Output only JSON: [{"point_2d": [x, y], "label": "end 1"}, {"point_2d": [x, y], "label": "end 2"}]'
PROMPT_PTS = (
    "This image is the frame at t={t:.2f} s of a short video. The video comes with this physics question:\n"
    "\"{context}\"\n"
    "To answer it, the length to measure is {what}. Mark the two end points of that length in this image. "
    "Output only JSON: [{{\"point_2d\": [x, y], \"label\": \"end 1\"}}, {{\"point_2d\": [x, y], \"label\": \"end 2\"}}]"
)
SPAN = {"w_ground": "the width of the {x}: straight across it, from its left edge to its right edge (lane lines, curbs or road edges), "
                    "both points on the road surface at the same distance from the camera",
        "tile": "one side of {x}: two neighbouring corners of one single tile, both on the floor"}


def parse_points(text, w, h):
    pts = []
    for mm in re.finditer(r"\"point_2d\"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", text or ""):
        pts.append([float(mm.group(1)) * w / 1000, float(mm.group(2)) * h / 1000])
    return pts[:2] if len(pts) >= 2 else None


def parse_boxes(text, w, h):
    m = re.search(r"\[\s*\{.*\}\s*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4:
            x1, y1, x2, y2 = (float(v) for v in b)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            box = [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                out.append(box)
    return out


def parse_box(text, w, h):
    b = parse_boxes(text, w, h)
    return b[0] if b else None


def stage_ground():
    import torch
    torch.use_deterministic_algorithms(True)   # strict: with warn_only=True the VLM answers still varied between runs
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    J = jobs()
    proc = AutoProcessor.from_pretrained(a.vlm)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    res = {}
    for j in J:
        res[j["id"]] = {}
        for t, ph in j["want"]:
            img, fi, _, _ = frame_at(j["video"], t)
            if img is None:
                continue
            h, w = img.shape[:2]
            two = ph.startswith("two ")
            what = f"two {ph[4:]} (both of them, one box each)" if two else ph
            text = PROMPT.format(t=t, context=j["question"], what=what)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[prompt], images=[Image.fromarray(img)], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = model.generate(**inp, max_new_tokens=200, do_sample=False)
            reply = proc.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if two:                             # one entry per instance: "<phrase>#0", "<phrase>#1"
                for bi, b in enumerate(parse_boxes(reply, w, h)[:2]):
                    res[j["id"]][f"{t}|{ph}#{bi}"] = {"t": t, "frame": fi, "phrase": f"{ph}#{bi}", "box": b, "reply": reply}
                print(j["id"], t, ph, "->", [v["box"] for k, v in res[j["id"]].items() if k.startswith(f"{t}|{ph}#")], flush=True)
            else:
                res[j["id"]][f"{t}|{ph}"] = {"t": t, "frame": fi, "phrase": ph, "box": parse_box(reply, w, h), "reply": reply}
                print(j["id"], t, ph, "->", res[j["id"]][f"{t}|{ph}"]["box"], flush=True)
        spans = []                                  # (key, time, what, plain): lengths the VLM marks by their two end points
        if j["plan"]["type"] in SPAN:
            spans.append(("points", j["tq"], SPAN[j["plan"]["type"]].format(x=j["plan"]["x"]), None))
        elif j["plan"]["type"] == "size":
            for t_ in j.get("size_ts") or [j["tq"]]:
                spans.append(("points" if t_ == j["tq"] else f"points@{t_}", t_, span_of(j["plan"]["dim"], j["plan"]["x"]),
                              f"the {j['plan']['dim']} of the {j['plan']['x']}"))
        pr = j.get("prior") or {}
        if not j["depth"] and pr.get("kind") == "size" and pr.get("phrase"):    # 2D: the prior's own stated length
            spans.append(("prior_points", j["tq"], f"{span_of(pr.get('dim') or 'size', pr['phrase'])} (the prior says: {pr.get('raw')})",
                          f"the {pr.get('dim') or 'size'} of the {pr['phrase']}"))
        for key, t, what, plain in spans:
            img, fi, _, _ = frame_at(j["video"], t)
            h, w = img.shape[:2]
            pts, reply = None, ""
            texts_ = [PROMPT_PTS.format(t=t, context=j["question"], what=what)]
            if plain:                               # retry: a plain request without the question context
                texts_.append(f"Mark the two end points of {plain} in this image. " + PLAIN_JSON)
            for txt_ in texts_:
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt_}]}]
                prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = proc(text=[prompt], images=[Image.fromarray(img)], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    o = model.generate(**inp, max_new_tokens=120, do_sample=False)
                reply = proc.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                pts = parse_points(reply, w, h)
                if pts:
                    break
            res[j["id"]][key] = {"t": t, "frame": fi, "points": pts, "reply": reply}
            print(j["id"], t, key, "->", pts, flush=True)
    if a.rows and os.path.exists(f"{OUT}/ground.json"):    # partial rerun: merge into the existing file
        old = json.load(open(f"{OUT}/ground.json"))
        keep = {jj["id"] for jj in J}
        J = [jj for jj in old["jobs"] if jj["id"] not in keep] + J
        res = {**{k: v for k, v in old["boxes"].items() if int(k) not in keep}, **{str(k): v for k, v in res.items()}}
    json.dump({"jobs": J, "boxes": res}, open(f"{OUT}/ground.json", "w"), indent=1)


def stage_geom():
    import torch
    torch.use_deterministic_algorithms(True)   # strict: with warn_only=True the VLM answers still varied between runs
    sys.path.insert(0, f"{R}/third_party/MoGe")
    sys.path.insert(0, f"{R}/third_party/utils3d-moge")
    from moge.model.v2 import MoGeModel
    from transformers import Sam2Model, Sam2Processor
    G = json.load(open(f"{OUT}/ground.json"))
    geo = geo_table()
    moge = MoGeModel.from_pretrained(f"{R}/models/moge-2-vitl-normal/model.pt").to("cuda").eval()
    sp = Sam2Processor.from_pretrained(f"{R}/models/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{R}/models/sam2.1-hiera-large").to("cuda").eval()
    for j in G["jobs"]:
        bx = G["boxes"].get(str(j["id"]), {})
        fm = geo.meas_f.get(j["id"])
        vid_rows = geo[geo.video_id == j["video"]]
        if fm is None or not np.isfinite(fm):
            fs = pd.to_numeric(vid_rows.meas_f, errors="coerce").dropna()
            fm = float(fs.median()) if len(fs) else None
        fov = float(np.degrees(2 * np.arctan(1 / (2 * fm)))) if fm and a.fov == "calib" else None
        store = {}
        for t in sorted({v["t"] for v in bx.values()}):
            img, fi, fps, n = frame_at(j["video"], t)
            H, W = img.shape[:2]
            with torch.no_grad():
                o = moge.infer(torch.tensor(img / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1), use_fp16=True, fov_x=fov)
            d = o["depth"].float().cpu().numpy()
            store[f"depth_{t}"] = np.where(np.isfinite(d), d, np.nan).astype(np.float32)
            store[f"normal_{t}"] = o["normal"].float().cpu().numpy().astype(np.float16)
            store[f"K_{t}"] = o["intrinsics"].float().cpu().numpy()
            store[f"gray_{t}"] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            items = [(k, v["box"]) for k, v in bx.items() if v["t"] == t and v.get("box")]
            if items:
                with torch.no_grad():
                    inp = sp(images=img, input_boxes=[[b for _, b in items]], return_tensors="pt").to("cuda")
                    out = sm(**inp, multimask_output=False)
                mk = sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][:, 0].numpy() > 0
                for (k, _), m_ in zip(items, mk):
                    store[f"mask_{k}"] = np.packbits(m_, axis=1)
                    store[f"maskw_{k}"] = np.array(W)
        np.savez_compressed(f"{GEO}/{j['id']}.npz", fov=np.array(fov if fov else np.nan), **store)
        print("geom", j["id"], j["video"], "frames", len({v['t'] for v in bx.values()}), "fov", fov, flush=True)


# ---------------------------------------------------------------- measurement (CPU)
def points(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u + 0.5 - cx) * depth / fx, (v + 0.5 - cy) * depth / fy, depth], -1)


def gravity(P, N, gray, objs):
    """floor plane on the lower image part (RANSAC), else Manhattan on textured pixels (prim_fit2), else level camera"""
    H, W = gray.shape
    g = gray.astype(np.float32)
    m1, m2 = cv2.blur(g, (7, 7)), cv2.blur(g * g, (7, 7))
    tex = np.sqrt(np.maximum(m2 - m1 * m1, 0)) > 6.0
    ok = np.isfinite(P).all(-1) & np.isfinite(N).all(-1) & ~objs
    sel = ok.copy()
    sel[: int(0.4 * H)] = False
    Pp = P[sel]
    rng = np.random.default_rng(0)
    if len(Pp) >= 500:
        Pp = Pp[rng.choice(len(Pp), min(len(Pp), 15000), replace=False)]
        thr, best = 0.015 * float(np.median(Pp[:, 2])), 0
        for _ in range(300):
            p0, p1, p2 = Pp[rng.choice(len(Pp), 3, replace=False)]
            n = np.cross(p1 - p0, p2 - p0)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = int((np.abs((Pp - p0) @ n) < thr).sum())
            if inl > best:
                best, bn, bp = inl, n, p0
        if best >= 0.2 * len(Pp):
            if np.dot(bn, -bp) < 0:
                bn = -bn
            if bn[1] < -0.3:
                return bn, "floor"
    idx = np.nonzero((ok & tex).reshape(-1))[0]
    if len(idx) >= 500:
        idx = rng.choice(idx, min(len(idx), 20000), replace=False)
        Nn, Pn = N.reshape(-1, 3)[idx].astype(np.float32), P.reshape(-1, 3)[idx]
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9
        U = []
        for ph in np.radians(np.arange(-20, 75.1, 2.5)):
            for ro in np.radians(np.arange(-6, 6.1, 2.0)):
                x, y, z = 0.0, -math.cos(ph), -math.sin(ph)
                U.append([x * math.cos(ro) - y * math.sin(ro), x * math.sin(ro) + y * math.cos(ro), z])
        U = np.array(U)
        c = Nn @ U.T
        hor = (c > math.cos(math.radians(15))) & (Pn @ U.T < 0)
        k = int(np.argmax(hor.sum(0) + 0.5 * (np.abs(c) < math.sin(math.radians(8))).sum(0)))
        if hor[:, k].sum() >= max(150, 0.02 * len(Nn)):
            u = Nn[hor[:, k]].mean(0)
            return u / np.linalg.norm(u), "manhattan"
    return np.array([0.0, -math.cos(math.radians(10)), -math.sin(math.radians(10))]), "assumed_level"


def snap3d(P, u, v, m=None, r=4):
    """3D point for image point (u, v): moved onto the object's mask when it falls just outside it (an end point on a
    silhouette edge otherwise picks up the background's depth), then the median of the mask's points around it"""
    H, W = P.shape[:2]
    u, v = int(round(min(max(u, 0), W - 1))), int(round(min(max(v, 0), H - 1)))
    if m is not None and m.any():
        if not m[v, u]:
            ys, xs = np.nonzero(m)
            k = int(np.argmin((xs - u) ** 2 + (ys - v) ** 2))
            if (xs[k] - u) ** 2 + (ys[k] - v) ** 2 > 30 ** 2:
                return None
            u, v = int(xs[k]), int(ys[k])
        sl = (slice(max(0, v - r), v + r + 1), slice(max(0, u - r), u + r + 1))
        q = P[sl][m[sl]]
    else:
        q = P[max(0, v - 2):v + 3, max(0, u - 2):u + 3].reshape(-1, 3)
    q = q[np.isfinite(q).all(1)]
    return np.median(q, 0) if len(q) else None


def stage_measure():
    G = json.load(open(f"{OUT}/ground.json"))
    val = pd.read_csv(f"{R}/data/val/validation_dataset.csv").set_index("Unnamed: 0") if S == "val" else None
    core = pd.read_csv(f"{R}/submissions/core_moge_focal_valdetail.csv").set_index("Unnamed: 0").answer if S == "val" else None
    rows = []
    for j in G["jobs"]:
        j["plan"] = plan(j["question"]) or j["plan"]
        rec = {"id": j["id"], "video": j["video"], "vt": j.get("vt"), "type": j["plan"]["type"], "x": j["plan"]["x"], "y": j["plan"]["y"],
               "value": np.nan, "why": ""}
        try:
            z = np.load(f"{GEO}/{j['id']}.npz")
        except FileNotFoundError:
            rec["why"] = "no geom"; rows.append(rec); continue
        bx = G["boxes"].get(str(j["id"]), {})

        def mask(t, ph):
            k = f"mask_{t}|{ph}"
            if k not in z.files:
                return None
            W = int(z[f"maskw_{t}|{ph}"])
            return np.unpackbits(z[k], axis=1)[:, :W].astype(bool)

        def geom(t):
            D = z[f"depth_{t}"]
            return points(D, z[f"K_{t}"]), z[f"normal_{t}"].astype(np.float32), z[f"gray_{t}"]

        # metres per MoGe unit from every stated distance whose object was found
        sc, wsc = [], []
        if not j["depth"]:                      # 2D videos: the prior object's own size, else the teammates' metres per pixel
            pr, t0, s0, ssrc = j.get("prior") or {}, j["tq"], None, "moge metric"
            mp = mask(t0, pr.get("phrase")) if pr.get("phrase") and f"depth_{t0}" in z.files else None
            if mp is not None and mp.sum() >= 9:
                P0, _, _ = geom(t0)
                fpx = float(z[f"K_{t0}"][0, 0]) * mp.shape[1]
                me = cv2.erode(mp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                zz = P0[me if me.sum() >= 9 else mp][:, 2]
                Zm = float(np.median(zz[np.isfinite(zz)]))
                ppts = (bx.get("prior_points") or {}).get("points") if isinstance(bx.get("prior_points"), dict) else None
                A0 = snap3d(P0, *ppts[0], mp) if ppts else None
                B0 = snap3d(P0, *ppts[1], mp) if ppts else None
                if pr.get("kind") == "size" and pr.get("value") and A0 is not None and B0 is not None and np.linalg.norm(A0 - B0) > 0:
                    s0, ssrc = pr["value"] / float(np.linalg.norm(A0 - B0)), "prior size (end points)"   # the same 3D length as the target
                elif pr.get("kind") == "size" and pr.get("value"):
                    if re.search(r"ball|sphere|coin|wheel|disc|circle", pr["phrase"]):
                        e_px = math.sqrt(4 * mp.sum() / math.pi)
                    else:
                        e_px = max(cv2.minAreaRect(np.argwhere(mp)[:, ::-1].astype(np.float32))[1])
                    s0, ssrc = pr["value"] / (e_px * Zm / fpx), "prior size"
                elif pr.get("mpp"):
                    s0, ssrc = pr["mpp"] * fpx / Zm, "prior metres-per-pixel"
            sc = [s0 if s0 else 1.0]
            rec["scale_src"] = ssrc
            if s0 is None:                      # nothing in the question fixes the metres (MoGe's own scale only): no answer
                rec.update(scale=1.0, why="scale not anchored"); rows.append(rec); continue
        else:
            rec["scale_src"] = "stated distances"
        for e in j["depth"]:
            te = round(min(float(e["t"]), j["tend"]), 3) if e.get("t") is not None else j["tq"]
            m = mask(te, e["phrase"]) if f"depth_{te}" in z.files else None
            if m is None or m.sum() < 9:
                continue
            P, _, _ = geom(te)
            me = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            q = P[me if me.sum() >= 9 else m]
            q = q[np.isfinite(q).all(1)]
            if len(q):
                sc.append(e["d"] / float(np.median(np.linalg.norm(q, axis=1))))
                wsc.append(float(m.sum()))         # bigger, steadier objects carry more weight (flying balls' MoGe depth is off)
        if not sc:
            rec["why"] = "no scale"; rows.append(rec); continue
        if j["depth"] and a.scale == "v2":       # median of the stated distances that agree within 20 %; at least half agree
            med = float(np.median(sc))
            inl = [v for v in sc if abs(math.log(v / med)) < math.log(1.2)]
            rec.update(spread=max(sc) / min(sc), n_agree=len(inl), scales=" ".join(f"{v:.2f}" for v in sc))
            if len(inl) < max(2, math.ceil(0.5 * len(sc))):
                rec.update(scale=med, n_scale=len(sc), why="scale disagrees"); rows.append(rec); continue
            s = float(np.median(inl))
        elif j["depth"]:                        # v3: area-weighted median; the agreeing entries must carry half the weight
            o_ = np.argsort(sc)
            cw_ = np.cumsum(np.array(wsc)[o_]) / np.sum(wsc)
            med = float(np.array(sc)[o_][int(np.searchsorted(cw_, 0.5))])
            agree_ = [i_ for i_, v in enumerate(sc) if abs(math.log(v / med)) < math.log(1.2)]
            wagree = float(np.sum([wsc[i_] for i_ in agree_]) / np.sum(wsc))
            inl = [sc[i_] for i_ in agree_]
            rec.update(spread=max(sc) / min(sc), n_agree=len(inl), w_agree=wagree, scales=" ".join(f"{v:.2f}" for v in sc))
            if wagree < 0.5:
                rec.update(scale=med, n_scale=len(sc), why="scale disagrees"); rows.append(rec); continue
            s = float(np.median(inl))
        else:
            s = float(sc[0])
            rec.update(n_agree=1, scales=f"{s:.2f}")
        t = j["tq"]
        if f"depth_{t}" not in z.files:
            rec["why"] = "no frame"; rows.append(rec); continue
        P, N, gray = geom(t)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        g_ = gray.astype(np.float32)
        tex_frac = float((np.sqrt(np.maximum(cv2.blur(g_ * g_, (7, 7)) - cv2.blur(g_, (7, 7)) ** 2, 0)) > 6.0).mean())
        rec["textured"] = tex_frac
        if tex_frac < 0.15 and j["plan"]["type"] != "size":   # uniform backdrop: MoGe's scene (floor, gravity) is invented;
                                                              # an object's own size needs only the object
            rec["why"] = "no scene texture"; rows.append(rec); continue
        mx, my = mask(t, j["plan"]["x"]), (mask(t, j["plan"]["y"]) if j["plan"]["y"] else None)
        objs = np.zeros(gray.shape, bool)              # things standing in the scene; surfaces (water, floor, a lane) stay in
        for k in bx:
            if bx[k]["t"] == t and bx[k].get("phrase") and not re.fullmatch(SURF + "|lane|road.*", bx[k]["phrase"]):
                mk = mask(t, bx[k]["phrase"])
                if mk is not None and mk.mean() < 0.3:
                    objs |= mk
        up, gsrc = gravity(P, N, gray, objs)
        hgt = P @ up
        fin = np.isfinite(hgt)
        horiz = (N @ up) > math.cos(math.radians(25))
        rec.update(scale=s, n_scale=len(sc), grav=gsrc, pitch=float(np.degrees(math.asin(np.clip(-up[2], -1, 1)))))
        typ = j["plan"]["type"]
        flat = lambda dv: dv - up * (dv @ up) if j["plan"].get("horiz") else dv
        if typ == "d_two":                      # two instances of one kind: centre to centre
            m0, m1 = mask(t, j["plan"]["x"] + "#0"), mask(t, j["plan"]["x"] + "#1")
            if m0 is None or m1 is None or m0.sum() < 9 or m1.sum() < 9:
                rec["why"] = "no two masks"; rows.append(rec); continue
            dv = np.median(P[m0 & fin], 0) - np.median(P[m1 & fin], 0)
            rec["value"] = float(np.linalg.norm(flat(dv))) * s / UNIT.get(j.get("unit"), 1.0)
            rows.append(rec); continue
        pts = (bx.get("points") or {}).get("points") if isinstance(bx.get("points"), dict) else None
        if typ == "size":                       # an object's dimension: the VLM's end points, snapped onto its mask, in 3D
            u_ = UNIT.get(j.get("unit"), 1.0)
            Ls, Le = [], []
            for t_ in j.get("size_ts") or [t]:
                key_ = "points" if t_ == j["tq"] else f"points@{t_}"
                pts_ = (bx.get(key_) or {}).get("points") if isinstance(bx.get(key_), dict) else None
                if f"depth_{t_}" not in z.files:
                    continue
                P_ = P if t_ == t else geom(t_)[0]
                m_ = mx if t_ == t else mask(t_, j["plan"]["x"])
                ok_ = np.isfinite(P_).all(-1)
                if pts_:
                    A_, B_ = snap3d(P_, *pts_[0], m_), snap3d(P_, *pts_[1], m_)
                    if A_ is not None and B_ is not None and np.linalg.norm(A_ - B_) > 0:
                        Ls.append(float(np.linalg.norm(A_ - B_)))
                        if m_ is not None and (m_ & ok_).sum() >= 20:
                            proj = P_[m_ & ok_] @ ((B_ - A_) / Ls[-1])
                            Le.append(float(np.percentile(proj, 97) - np.percentile(proj, 3)))
                elif m_ is not None and (m_ & ok_).sum() >= 20:     # no end points: the mask's longest 3D extent
                    q_ = P_[m_ & ok_]
                    q_ = q_ - q_.mean(0)
                    _, V_ = np.linalg.eigh(np.cov(q_.T))
                    pr_ = q_ @ V_[:, -1]
                    Le.append(float(np.percentile(pr_, 97) - np.percentile(pr_, 3)))
            if not Ls and not Le:
                rec["why"] = "no end points"; rows.append(rec); continue
            rec.update(value=(float(np.median(Ls)) if Ls else float(np.median(Le))) * s / u_,
                       mode=f"end points x{len(Ls)}" if Ls else "mask extent", n_frames=len(Ls))
            if Le:
                rec["value_ext"] = float(np.median(Le)) * s / u_
            rows.append(rec); continue
        if typ in SPAN and pts:                 # the VLM's two end points, located in 3D, distance along the ground
            def p3(u, v_):
                u, v_ = int(round(u)), int(round(v_))
                win = P[max(0, v_ - 2):v_ + 3, max(0, u - 2):u + 3].reshape(-1, 3)
                win = win[np.isfinite(win).all(1)]
                return np.median(win, 0) if len(win) else None
            A_, B_ = p3(*pts[0]), p3(*pts[1])
            if A_ is not None and B_ is not None:
                dv = A_ - B_
                dv = dv - up * (dv @ up)
                rec.update(value=float(np.linalg.norm(dv)) * s / UNIT.get(j.get("unit"), 1.0), mode="end points")
                rows.append(rec); continue
        if mx is None or mx.sum() < 9:
            rec["why"] = "no x mask"; rows.append(rec); continue
        v = np.nan
        if typ in ("h_above", "h_under", "h_gap"):
            if my is None or my.sum() < 9:
                rec["why"] = "no y mask"; rows.append(rec); continue
            ysel = my & ~mx & horiz & fin
            if ysel.sum() < 20:
                ysel = my & ~mx & fin
            hy = float(np.median(hgt[ysel]))
            xs = mx & fin
            if typ == "h_above":
                xt = mx & horiz & fin
                hx = float(np.median(hgt[xt])) if xt.sum() >= 20 else float(np.percentile(hgt[xs], 90))
            else:                                   # underside of X / bottom of X
                hx = float(np.percentile(hgt[xs], 3))
            v = (hx - hy) * s
            rec.update(hx=hx * s, hy=hy * s)
        elif typ in ("d_min", "d_ctr"):
            if my is None or my.sum() < 9:
                rec["why"] = "no y mask"; rows.append(rec); continue
            wx_, wy_ = re.search(r"\bwalls?\b", j["plan"]["x"] or ""), re.search(r"\bwalls?\b", j["plan"]["y"] or "")
            if wx_ or wy_:                      # a "wall" mask that is mostly horizontal surface is floor; walls: distance on the ground
                wm_ = my if wy_ else mx
                flat_frac = float((wm_ & fin & horiz).sum()) / max(int((wm_ & fin).sum()), 1)
                rec["wall_flat"] = flat_frac
                if flat_frac > 0.5:
                    rec["why"] = "wall mask is a floor"; rows.append(rec); continue
            px, py = P[mx & fin], P[my & fin]
            if wx_ or wy_:
                px, py = px - np.outer(px @ up, up), py - np.outer(py @ up, up)
            rng = np.random.default_rng(0)
            px = px[rng.choice(len(px), min(len(px), 2500), replace=False)]
            py = py[rng.choice(len(py), min(len(py), 2500), replace=False)]
            if typ == "d_min":
                dd = np.linalg.norm(px[:, None, :] - py[None, :, :], axis=-1)
                v = float(np.percentile(dd, 0.5)) * s
            else:
                v = float(np.linalg.norm(flat(np.median(px, 0) - np.median(py, 0)))) * s
        elif typ == "tile":                     # one tile on the floor: mean of its two side lengths on the floor plane
            gsel = mx & fin
            e0 = np.array([1.0, 0, 0]) - up * up[0]
            e0 /= np.linalg.norm(e0)
            e1 = np.cross(up, e0)
            q2 = np.stack([P[gsel] @ e0, P[gsel] @ e1], 1)
            q2 = q2 - q2.mean(0)
            _, V_ = np.linalg.eigh(np.cov(q2.T))
            sides = [float(np.percentile(q2 @ V_[:, i], 97) - np.percentile(q2 @ V_[:, i], 3)) for i in (0, 1)]
            v = float(np.mean(sides)) * s
        elif typ == "w_ground":
            gsel = mx & horiz & fin
            if gsel.sum() < 50:
                rec["why"] = "no ground points"; rows.append(rec); continue
            e0 = np.array([1.0, 0, 0]) - up * up[0]
            e0 /= np.linalg.norm(e0)
            e1 = np.cross(up, e0)
            q2 = np.stack([P[gsel] @ e0, P[gsel] @ e1], 1)
            q2 = q2 - q2.mean(0)
            w_, V_ = np.linalg.eigh(np.cov(q2.T))
            minor = q2 @ V_[:, 0]
            v = float(np.percentile(minor, 97) - np.percentile(minor, 3)) * s
        u_ = UNIT.get(j.get("unit"), 1.0)
        rec["value"] = v / u_ if np.isfinite(v) else np.nan
        rows.append(rec)
    d = pd.DataFrame(rows).set_index("id")
    d.to_csv(f"{R}/geo/out/scene3d_{S}{a.tag}.csv")
    pd.set_option("display.width", 250)
    if val is not None:
        y = val.ground_truth_posterior.reindex(d.index).astype(float)
        d["truth"], d["core"] = y, core.reindex(d.index)
        mra = lambda p, t_: float((abs(p - t_) / t_ < 1 - TH).mean()) if np.isfinite(p) else 0.0
        d["m_val"] = [mra(p, t_) for p, t_ in zip(d.value, y)]
        d["m_core"] = [mra(p, t_) for p, t_ in zip(d.core, y)]
        src_ = pd.read_csv(f"{R}/submissions/core_moge_focal_valdetail.csv").set_index("Unnamed: 0").answer_source
        d["core_src"] = src_.reindex(d.index)
        # routing: scene3d stands in only where the fused answer is CaW's alone (no geometric measurement behind it)
        d["routed"] = d.value.notna() & d.core_src.isin(["caw_nogeo", "d3size_caw"])
        d["final"] = np.where(d.routed, d.value, d.core)
        d["m_final"] = [mra(p, t_) for p, t_ in zip(d.final, y)]
        cols = ["video", "type", "x", "y", "scale_src", "scale", "why", "value", "truth", "core", "core_src", "routed", "m_val", "m_core", "m_final"]
        print(d[[c for c in cols if c in d]].round(3).to_string())
        if "value_ext" in d:
            d["m_ext"] = [mra(p, t_) for p, t_ in zip(d.value_ext, y)]
        d["grp"] = np.where(d.type == "size", "size", "scene") + " " + d.vt.astype(str)
        for g_, gg in d.groupby("grp"):
            an = gg.value.notna()
            print("  %-9s n %3d answered %3d | on answered: scene3d %.3f%s core %.3f | routed %d: scene3d %.3f core %.3f" % (
                g_, len(gg), an.sum(), gg.m_val[an].mean(), (" (mask extent %.3f)" % gg.m_ext[an].mean()) if "m_ext" in gg else "",
                gg.m_core[an].mean(), gg.routed.sum(), gg.m_val[gg.routed].mean() if gg.routed.any() else float("nan"),
                gg.m_core[gg.routed].mean() if gg.routed.any() else float("nan")))
        print("why not answered:", d.loc[~d.value.notna(), "why"].value_counts().to_dict())
        ans = d.value.notna()
        print("answered %d of %d | MRA on answered: scene3d %.3f core %.3f | routed %d: scene3d %.3f core %.3f | all rows: final %.3f core %.3f" % (
            ans.sum(), len(d), d.m_val[ans].mean(), d.m_core[ans].mean(), d.routed.sum(), d.m_val[d.routed].mean(), d.m_core[d.routed].mean(),
            d.m_final.mean(), d.m_core.mean()))
        d.to_csv(f"{R}/geo/out/scene3d_{S}{a.tag}.csv")
    else:
        print(d.round(3).to_string())


if __name__ == "__main__":
    {"jobs": lambda: [print(j["id"], j["video"], j["plan"], "tq", j["tq"], "asked", j["t_asked"], "end %.2f" % j["tend"], j["want"]) for j in jobs()],
     "ground": stage_ground, "geom": stage_geom, "measure": stage_measure}[a.stage]()
