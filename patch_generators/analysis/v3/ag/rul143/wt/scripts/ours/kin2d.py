#!/usr/bin/env python
"""kin2d: planar (2D) measurements anchored on a dynamic prior - a named object's speed / acceleration, or gravity.

The chain's detector tracks miss small or unusual objects (a bubble inside a drop, pedestrians seen from a drone, moon
models in an orrery), so a D2 row gets no scale and the core answer is CaW's guess. Here Qwen3-VL-32B points at each
object on sampled frames; a small object is followed in a zoomed crop around its previous position. The points form a
trajectory; the prior's quantity read on the prior object's trajectory (px/s or px/s^2) gives m/px, and the question's
quantity is read in px with the same tools (trajectories for motion, two end points for sizes and distances).

stages:   locate (GPU: VLM points and boxes)  |  measure (CPU)
outputs:  geo/out/kin2d_<split><tag>/<video>.json   geo/out/kin2d_<split><tag>.csv
usage:    python scripts/ours/kin2d.py --split val --stage locate [--videos a,b] [--tag _v1]
"""
import argparse
import json
import math
import os
import re

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # bit-reproducible cuBLAS (greedy VLM boxes flipped digits between identical calls without it)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--stage", required=True, choices=["locate", "measure"])
ap.add_argument("--tag", default="")
ap.add_argument("--videos", default="")
ap.add_argument("--nsamp", type=int, default=40, help="sampled frames per object track")
ap.add_argument("--reuse", default="", help="tag of a locate run whose targets / ends / pairs are reused")
ap.add_argument("--rows", default="", help="only these question ids: a comma list or a file holding one")
ap.add_argument("--vlm", default=os.path.join(os.environ.get("QP_ROOT", "."), "models/Qwen3-VL-32B-Instruct"))
a = ap.parse_args()
S = a.split
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = f"{R}/geo/out/kin2d_{S}{a.tag}"
TH = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
MOTION = ("speed", "avgspeed", "accel", "hvel", "disp", "path", "orbitdiam")
G = 9.8
DIMTXT = {"height": "its highest point and its lowest point (top and bottom)",
          "length": "its two ends along its length",
          "width": "its two sides across its width",
          "diameter": "two opposite points on its outline, through its centre",
          "thickness": "its two faces across its thickness",
          "wingspan": "the tips of its two wings",
          "side": "the two ends of one side"}
POINT = ("This image is {where} the frame at t={t:.2f} s of a short video. The video comes with this physics question:\n"
         "\"{context}\"\n"
         "Point to {what}.{extra} Output only JSON: [{{\"point_2d\": [x, y], \"label\": \"<short name>\"}}], "
         "or [] if it is not visible.")
BOX = ("This image is the frame at t={t:.2f} s of a short video. The video comes with this physics question:\n"
       "\"{context}\"\n"
       "Locate {what} in this image. If several things could match, choose the one the question means. "
       "Output only JSON: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<short name>\"}}]")
ENDS = ("This image is {where} the frame at t={t:.2f} s of a short video. The video comes with this physics question:\n"
        "\"{context}\"\n"
        "To answer it, the length to measure is the {dim} of {what}: {span}. Mark the two end points of that length in "
        "this image. Output only JSON: [{{\"point_2d\": [x, y], \"label\": \"end 1\"}}, {{\"point_2d\": [x, y], \"label\": \"end 2\"}}]")
GRAV_OBJ = "the object that is flying, thrown, falling or bouncing through the air (for example a ball)"


def video_path(vid):
    for p in (f"{R}/data/{S}/validation_videos/{vid}.mp4", f"{R}/data/test_480p/{vid}.mp4", f"{R}/data/test/{vid}.mp4"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(vid)


def read_video(vid):
    cap = cv2.VideoCapture(video_path(vid))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr, cap.get(cv2.CAP_PROP_FPS)


SCENE_WORDS = (r"\b(lanes?|road|street|court|tiles?|floor|ceiling|room|wall|field|track|pool|river|sidewalk|crosswalk|ground|sky|"
               r"water|ruler|calibre|window frame|markings?|gap|opening|surface|deck|lattice|frame of)\b")     # geo/measure.py SCENE
STRIP = r"\b(?:lanes?|roads?|streets?|paths?|sidewalks?|crosswalks?|tracks?|lines?|stripes?)\b"


def side_fix(r):
    """the teammates' QP_SIDE_FIX (geo/measure_v3.py load_split_tracks) on a parse row: "the long side of X" -> target X with dim
    length (short side -> width); "width (longer side) of X" -> dim length. A locative tail ("on the ground") is cut first."""
    q = r.get("q") or {}
    if q.get("kind") != "size":
        return r
    ql = r["question"].lower()
    tp = ((r.get("target") or {}).get("phrase") or "").strip().lower()
    m_ = re.match(r"^(?:the\s+)?(longer|shorter|long|short|longest|shortest)\s+(?:side|edge)$", tp)
    mo = re.search(r"\b(?:longer|shorter|long|short|longest|shortest)\s+(?:side|edge)\s+of\s+(?:the\s+|a\s+|an\s+)?"
                   r"([a-z][a-z \-]*?)\s*(?:\bin\b|,|\?|$)", ql)
    if m_ and mo:
        obj = re.split(r"\s+(?:on|in|at|near|of)\s+(?:the|a|an)\s+", mo.group(1))[0].strip()
        if obj and not re.search(SCENE_WORDS, obj):
            return dict(r, target=dict(r["target"], phrase=obj, raw=obj), side_fix="target",
                        q=dict(q, dim="length" if m_.group(1).startswith("long") else "width"))
    elif re.search(r"\bwidth\s*\(\s*(?:the\s+)?long(?:er|est)?\s+side\s*\)", ql) and q.get("dim") == "width":
        return dict(r, q=dict(q, dim="length"), side_fix="dim")
    return r


def row_filter():
    if not a.rows:
        return None
    txt = open(a.rows).read() if os.path.exists(a.rows) else a.rows
    return {int(x) for x in txt.replace("\n", ",").split(",") if x.strip()}


def rows():
    """D-prior rows of planar videos with the parse's question plan (only --rows when given)"""
    keep = row_filter()
    out = []
    for line in open(f"{R}/geo/out/parse_{S}_v2.jsonl"):
        r = json.loads(line)
        pr = (r.get("priors") or [None])[0]
        if r["video_type"][1] != "2" or not pr or pr.get("kind") not in ("speed", "accel") or not pr.get("ok"):
            continue
        if keep is not None and r["id"] not in keep:
            continue
        out.append(side_fix(r))
    return out


def phrase(ent, fallback=None):
    if not ent or not ent.get("phrase"):
        return fallback
    w = [x for x in (ent.get("where") or []) if not x.startswith("#")]
    return "the " + ent["phrase"] + (f" ({', '.join(w)})" if w else "")


def raw_object(raw):
    """the object a dynamic prior is about, in the prior's own words: "speed of the bubble in the drop = ..." -> "bubble in
    the drop", "pedestrian walking speed ~1.1 m/s" -> "pedestrian walking", "Callisto's model speed = ..." -> "callisto's model"
    (the parse keeps only a head noun: "bubble", "person")"""
    s = re.split(r"[=~\u2248:]|\b(?:is|are|was|equals)\b", raw.lower())[0].strip()
    m = re.search(r"\b(?:speed|velocity|acceleration|acc)\s+of\s+(?:the\s+|a\s+|an\s+)?(.+)$", s)
    s = m.group(1) if m else re.sub(r"\b(?:average\s+|mean\s+|constant\s+|initial\s+)?(?:speed|velocity|acceleration|acc)\b.*$", "", s)
    s = re.sub(r"\s+(?:at|after|before|from|during)\s+(?:t\s*=\s*)?[\d.]+\s*s?\b.*$", "", s.strip())
    s = re.sub(r"^(?:the|a|an)\s+", "", s.strip(" ,.")).strip()
    return s if 2 <= len(s) <= 60 else ""


def prior_object(r):
    pr = r["priors"][0]
    if pr.get("gravity"):
        return GRAV_OBJ
    ro = raw_object(pr.get("raw") or "")
    if re.fullmatch(r"[a-z]+ing", ro) and pr.get("obj"):      # "walking velocity = 1.25 m/s": add the parse's noun ("person")
        ro = f"{ro} {pr['obj']}"
    if ro:
        return "the " + ro
    return phrase(r.get("prior_obj"), ("the " + pr["obj"]) if pr.get("obj") else None)


def plan(r):
    """objects to follow, lengths to mark and object pairs to point at for one row"""
    q = r["q"]
    k = q.get("kind")
    objs, ends, pairs = [], [], []
    if k in MOTION and phrase(r.get("target")):
        objs.append(phrase(r["target"]))
    elif k == "dist2" and phrase(r.get("target")) and phrase(r.get("target2")):
        pairs.append((phrase(r["target"]), phrase(r["target2"])))
    elif k == "size" and phrase(r.get("target")):
        ends.append((phrase(r["target"]), q.get("dim") or "size"))
    return objs, ends, pairs


def span_text(what, dim):
    if re.search(r"\b(?:lane|road|street|path|track|crosswalk|sidewalk)\b", what) and dim == "width":
        return "straight across it, from one edge (lane line, curb or road edge) to the other, perpendicular to its direction"
    return DIMTXT.get(dim, "the two ends of exactly the length the question asks about")


# ----------------------------------------------------------------------------- locate (GPU)
class VLM:
    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.use_deterministic_algorithms(True)   # strict: with warn_only=True the VLM answers still varied between runs
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    def ask(self, img, text, n_tok=120):
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[Image.fromarray(img)], return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            o = self.model.generate(**inp, max_new_tokens=n_tok, do_sample=False)
        return self.proc.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_pts(text, w, h):
    return [[float(m.group(1)) * w / 1000, float(m.group(2)) * h / 1000]
            for m in re.finditer(r"\"point_2d\"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", text or "")]


def parse_box(text, w, h):
    m = re.search(r"\"bbox_2d\"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", text or "")
    if not m:
        return None
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    b = [min(x1, x2) * w / 1000, min(y1, y2) * h / 1000, max(x1, x2) * w / 1000, max(y1, y2) * h / 1000]
    return b if b[2] - b[0] >= 1 and b[3] - b[1] >= 1 else None


def crop(img, c, side, out=640):
    """square crop of side `side` px centred at c (shifted to stay inside), resized to out x out; returns crop, x0, y0, zoom"""
    h, w = img.shape[:2]
    side = int(max(8, min(side, w, h)))
    x0 = int(np.clip(round(c[0] - side / 2), 0, w - side))
    y0 = int(np.clip(round(c[1] - side / 2), 0, h - side))
    z = out / side
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (out, out), interpolation=cv2.INTER_CUBIC), x0, y0, z


CROP_EXTRA = " If several match, point to the one closest to the centre of the crop."
# a crop around where the object was boxed or last seen: the VLM is told that it should be there (it may be small) - with
# an easy "[] if not visible" exit it gives up on a 9 px bubble inside a drop that it finds when told the bubble is there
POINT_IN = ("This image is a zoomed-in crop (x{z:.1f}) of the frame at t={t:.2f} s of a short video, centred where {what} was "
            "last seen. The video comes with this physics question:\n\"{context}\"\n"
            "{what} should be in this crop, possibly small. Point to it. If several match, point to the one closest to the "
            "centre of the crop. Output only JSON: [{{\"point_2d\": [x, y], \"label\": \"<short name>\"}}]. "
            "Only if it is clearly not in the crop, output [].")


def ask_crop_point(vlm, img, c, side, t, what, context, expect=True):
    im, x0, y0, z = crop(img, c, side)
    if expect:
        txt = POINT_IN.format(z=z, t=t, what=what, context=context)
    else:
        txt = POINT.format(where=f"a zoomed-in crop (x{z:.1f}) of", t=t, context=context, what=what, extra=CROP_EXTRA)
    q = parse_pts(vlm.ask(im, txt, 80), 640, 640)
    return [x0 + q[0][0] / z, y0 + q[0][1] / z] if q else None


def find(vlm, img, t, what, context, box=None, side=96.0, tiles=True):
    """find an object on one frame: the full frame, then a zoomed crop around its nearest known box, then a grid of zoomed
    square tiles (side ~ 40 % of the short image side, 20 % overlap) - a pedestrian seen from a drone, a bubble inside a
    drop that the full frame does not resolve"""
    h, w = img.shape[:2]
    q = parse_pts(vlm.ask(img, POINT.format(where="", t=t, context=context, what=what, extra=""), 80), w, h)
    if q:
        return q[0], "full"
    if box is not None:
        p = ask_crop_point(vlm, img, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                           max(64.0, 1.5 * max(box[2] - box[0], box[3] - box[1])), t, what, context)
        if p is not None:
            return p, "box crop"
    if tiles:
        sd = max(128.0, 0.4 * min(h, w))
        nx, ny = int(math.ceil((w - sd) / (0.8 * sd))) + 1, int(math.ceil((h - sd) / (0.8 * sd))) + 1
        for cy in np.linspace(sd / 2, h - sd / 2, ny):
            for cx in np.linspace(sd / 2, w - sd / 2, nx):
                p = ask_crop_point(vlm, img, [cx, cy], sd, t, what, context, expect=False)
                if p is not None:
                    return p, "tile"
    return None, None


def follow(vlm, frames, fps, what, context, nsamp):
    """the object's image position on sampled frames. Full-frame boxes on three frames give its size; a small object (box
    side < 20 % of the short image side, or no box at all) is followed in a zoomed crop (3x its size, >= 64 px) around its
    previous position; when lost it is searched again (full frame, crop around its nearest box, zoomed tiles at most every
    5th sampled frame)"""
    n = len(frames)
    h, w = frames[0].shape[:2]
    boxes = {}
    for k in sorted({int(round(f * (n - 1))) for f in (0.15, 0.5, 0.85)}):
        boxes[k] = parse_box(vlm.ask(frames[k], BOX.format(t=k / fps, context=context, what=what)), w, h)
    sz = [max(b[2] - b[0], b[3] - b[1]) for b in boxes.values() if b]
    size = float(np.median(sz)) if sz else None
    small = size is None or size < 0.2 * min(w, h)
    side = max(3.0 * size, 64.0) if size else 96.0
    ks = sorted(set(np.linspace(0, n - 1, min(n, nsamp)).round().astype(int).tolist()))
    pts, src, prev, miss, last_tiles = {}, {}, None, 0, -99
    for ii, k in enumerate(ks):
        t = k / fps
        p = None
        if small and prev is not None and miss < 2:
            p = ask_crop_point(vlm, frames[k], prev, side, t, what, context)
            if p is not None:
                src[k] = "crop"
        if p is None:
            near = sorted((abs(kk - k), kk) for kk, b in boxes.items() if b)
            bx = boxes[near[0][1]] if near else None
            tiles = small and ii - last_tiles >= 5
            if tiles:
                last_tiles = ii
            p, how = find(vlm, frames[k], t, what, context, box=bx if small else None, side=side, tiles=tiles)
            if p is not None:
                src[k] = how
                if small and how == "full":          # refine on a crop around the full-frame point
                    p2 = ask_crop_point(vlm, frames[k], p, side, t, what, context)
                    if p2 is not None:
                        p, src[k] = p2, "full+crop"
        miss = 0 if p is not None else miss + 1
        if p is not None:
            pts[k], prev = p, p
    dropped = {}
    if small and len(pts) >= 3:
        keep = chain(pts, src, side)
        dropped = {str(k): pts[k] for k in pts if k not in keep}
        pts = {k: pts[k] for k in keep}
    return {"what": what, "size": size, "small": small, "side": side, "boxes": {str(k): b for k, b in boxes.items()},
            "pts": {str(k): v for k, v in pts.items()}, "src": {str(k): v for k, v in src.items()}, "dropped": dropped}


def chain(pts, src, side):
    """a small object's points split where a re-found point (full frame, box crop, tile) jumps more than half a follow window
    from the previous one: the search picked another object (a static highlight on the dropper instead of the bubble in the
    drop). Kept: the chain with the most points, ties to the one with more box-crop / full-frame points (tiles are the
    weakest evidence)."""
    ks = sorted(pts)
    chains, cur = [], [ks[0]]
    for a_, b_ in zip(ks, ks[1:]):
        jump = math.hypot(pts[b_][0] - pts[a_][0], pts[b_][1] - pts[a_][1])
        if src.get(b_) != "crop" and jump > 0.5 * side:
            chains.append(cur)
            cur = []
        cur.append(b_)
    chains.append(cur)
    score = lambda c: (len(c), sum(src.get(k) in ("box crop", "full", "full+crop") for k in c))
    return max(chains, key=score)


def ct_refine(ct, frames, o):
    """CoTracker3 (offline) tracks seeded on 3 x 3 grids (spacing 0.15 x the object's size, >= 2 px) around up to 12 of the
    VLM points; each grid gives a per-frame median track, a grid whose track stays within max(0.5 x size, 6 px) of the VLM
    points (median over the sampled frames) is kept, and the per-frame median of the kept tracks is a dense, sub-pixel
    trajectory of the same object"""
    import torch
    ks = sorted(int(k) for k in o["pts"])
    if len(ks) < 3:
        return None
    n = len(frames)
    step = max(1, int(math.ceil(n / 300)))
    idx = list(range(0, n, step))
    size = o.get("size") or 20.0
    sel = sorted({ks[i] for i in np.linspace(0, len(ks) - 1, min(12, len(ks))).round().astype(int)})
    sp = max(0.15 * size, 2.0)
    q = []
    for k in sel:
        x, y = o["pts"][str(k)]
        for dx in (-sp, 0.0, sp):
            for dy in (-sp, 0.0, sp):
                q.append([int(round(k / step)), x + dx, y + dy])
    vt = torch.from_numpy(np.stack([frames[i] for i in idx])).permute(0, 3, 1, 2)[None].float().cuda()
    with torch.no_grad():
        tr, _ = ct(vt, queries=torch.tensor(q, dtype=torch.float32)[None].cuda(), backward_tracking=True)
    tr = tr[0].cpu().numpy()
    del vt
    torch.cuda.empty_cache()
    G = len(sel)
    gtr = np.median(tr.reshape(len(idx), G, 9, 2), axis=2)
    kpos = [int(np.argmin(np.abs(np.array(idx) - k))) for k in ks]
    V = np.array([o["pts"][str(k)] for k in ks])
    dev = [float(np.median(np.linalg.norm(gtr[kpos, g] - V, axis=1))) for g in range(G)]
    tol = max(0.5 * size, 6.0)
    good = np.array(dev) <= tol
    if good.sum() < 3:
        return {"ok": False, "why": "few grids agree", "dev": [round(d, 2) for d in dev], "tol": tol, "n_good": int(good.sum())}
    P = np.median(gtr[:, good], axis=1)
    # net motion over the VLM track's span: CoTracker must reproduce it (a transparent drop's refraction pattern moves less
    # than the bubble inside it)
    TT = np.array(ks, float)
    cv = [np.polyfit(TT - TT.mean(), V[:, d], 1) for d in range(2)]
    d_vlm = np.array([cv[0][0], cv[1][0]]) * (TT[-1] - TT[0])
    d_ct = P[kpos[-1]] - P[kpos[0]]
    mis = float(np.linalg.norm(d_ct - d_vlm))
    res = {"frames": idx, "pts": P.round(2).tolist(), "dev": [round(d, 2) for d in dev], "tol": tol, "n_good": int(good.sum()),
           "d_vlm": d_vlm.round(2).tolist(), "d_ct": d_ct.round(2).tolist()}
    if mis > 0.25 * float(np.linalg.norm(d_vlm)) + 3.0:
        return dict(res, ok=False, why=f"net motion differs ({mis:.1f}px)")
    return dict(res, ok=True)


def orient_span(context, what, dim):
    ql = context.lower()
    if re.search(r"\bhorizontal(?:ly)?\b", ql):
        return "its leftmost and rightmost points: a horizontal span, left to right in the image"
    if re.search(r"\bvertical(?:ly)?\b", ql):
        return "its topmost and bottommost points: a vertical span, top to bottom in the image"
    return span_text(what, dim)


def mark_ends(vlm, frames, fps, what, dim, context, t):
    """two end points of an object dimension at time t; a small object is marked on a zoomed crop around its box"""
    n = len(frames)
    k = int(np.clip(round((t if t is not None else 0.5 * (n - 1) / fps) * fps), 0, n - 1))
    img = frames[k]
    h, w = img.shape[:2]
    span = orient_span(context, what, dim)
    box = parse_box(vlm.ask(img, BOX.format(t=k / fps, context=context, what=what)), w, h)
    res = {"frame": k, "box": box, "span": span}
    if box is not None and max(box[2] - box[0], box[3] - box[1]) < 0.35 * min(w, h):
        side = max(1.6 * max(box[2] - box[0], box[3] - box[1]), 64.0)
        im, x0, y0, z = crop(img, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], side)
        rep = vlm.ask(im, ENDS.format(where=f"a zoomed-in crop (x{z:.1f}) of", t=k / fps, context=context, what=what, dim=dim, span=span))
        q = parse_pts(rep, 640, 640)
        if len(q) >= 2 and np.hypot(q[0][0] - q[1][0], q[0][1] - q[1][1]) > 2:
            res.update(points=[[x0 + p[0] / z, y0 + p[1] / z] for p in q[:2]], how="crop", zoom=z, reply=rep[:300])
            return res
    rep = vlm.ask(img, ENDS.format(where="", t=k / fps, context=context, what=what, dim=dim, span=span))
    q = parse_pts(rep, w, h)
    ok = len(q) >= 2 and np.hypot(q[0][0] - q[1][0], q[0][1] - q[1][1]) > 1
    res.update(points=q[:2] if ok else None, how="full", reply=rep[:300])
    return res


def parse_boxes_all(text, w, h):
    out = []
    for m in re.finditer(r"\"bbox_2d\"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", text or ""):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        out.append([min(x1, x2) * w / 1000, min(y1, y2) * h / 1000, max(x1, x2) * w / 1000, max(y1, y2) * h / 1000])
    return out


def mark_pair(vlm, frames, fps, w1, w2, context, t):
    """the two objects of a distance question at time t (their box centres); two instances of one class are boxed together"""
    n = len(frames)
    k = int(np.clip(round((t or 0.0) * fps), 0, n - 1))
    img = frames[k]
    h, w = img.shape[:2]
    if w1 == w2:
        rep = vlm.ask(img, BOX.format(t=k / fps, context=context, what=f"each of the two {w1[4:]} the question is about (one box per object)"), 200)
        bs = parse_boxes_all(rep, w, h)
        if len(bs) >= 2:
            return {"frame": k, "points": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in bs[:2]], "boxes": bs[:2], "how": "pair boxes", "reply": rep[:300]}
        return {"frame": k, "points": None, "how": "pair boxes", "reply": rep[:300]}
    out = []
    for wh in (w1, w2):
        b = parse_box(vlm.ask(img, BOX.format(t=k / fps, context=context, what=wh)), w, h)
        if b is not None:
            out.append(([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], b, "box"))
            continue
        p, how = find(vlm, img, k / fps, wh, context, tiles=True)
        out.append((p, None, how))
    pts = [o[0] for o in out]
    return {"frame": k, "points": pts if all(p is not None for p in pts) else None, "boxes": [o[1] for o in out], "how": "two " + "/".join(str(o[2]) for o in out)}


def stage_locate():
    import sys
    sys.path.insert(0, os.path.join(os.environ.get("QP_ROOT", "."), "third_party/co-tracker"))
    from cotracker.predictor import CoTrackerPredictor
    os.makedirs(OUT, exist_ok=True)
    byv = {}
    for r in rows():
        byv.setdefault(r["video_id"], []).append(r)
    vids = sorted(byv) if not a.videos else [v for v in a.videos.split(",") if v in byv]
    vlm = VLM(a.vlm)
    ct = CoTrackerPredictor(checkpoint=os.path.join(os.environ.get("QP_ROOT", "."), "models/cotracker3/scaled_offline.pth"), offline=True, window_len=60).to("cuda").eval()
    for vid in vids:
        op = f"{OUT}/{vid}.json"
        if os.path.exists(op):
            continue
        frames, fps = read_video(vid)
        rs = byv[vid]
        res = {"video": vid, "fps": fps, "n": len(frames), "H": frames[0].shape[0], "W": frames[0].shape[1], "objs": {}, "ends": {}, "pairs": {}}
        pobj = prior_object(rs[0])
        todo = {pobj: rs[0]["priors"][0]["raw"]} if pobj else {}
        old = f"{R}/geo/out/kin2d_{S}{a.reuse}/{vid}.json" if a.reuse else None
        if old and os.path.exists(old):           # targets as before; only the prior object is followed again
            J0 = json.load(open(old))
            res.update(ends=J0["ends"], pairs=J0["pairs"], objs={k_: v_ for k_, v_ in J0["objs"].items() if k_ != J0.get("prior_obj")})
            if pobj in res["objs"]:
                todo = {}
            rs = []
        for r in rs:
            objs, ends, pairs = plan(r)
            for o in objs:
                todo.setdefault(o, r["question"])
            for what, dim in ends:
                res["ends"][str(r["id"])] = mark_ends(vlm, frames, fps, what, dim, r["question"], r["q"].get("t"))
                res["ends"][str(r["id"])].update(what=what, dim=dim)
                print(vid, r["id"], "ends", what, dim, "->", res["ends"][str(r["id"])].get("points"), flush=True)
            for w1, w2 in pairs:
                res["pairs"][str(r["id"])] = mark_pair(vlm, frames, fps, w1, w2, r["question"], r["q"].get("t"))
                print(vid, r["id"], "pair", w1, w2, "->", res["pairs"][str(r["id"])].get("points"), res["pairs"][str(r["id"])].get("how"), flush=True)
        for o, ctx in todo.items():
            res["objs"][o] = follow(vlm, frames, fps, o, ctx, a.nsamp)
            res["objs"][o]["ct"] = ct_refine(ct, frames, res["objs"][o])
            cr = res["objs"][o]["ct"] or {}
            print(vid, "follow", o, "size", res["objs"][o]["size"], "small", res["objs"][o]["small"], "pts", len(res["objs"][o]["pts"]),
                  "src", sorted(set(res["objs"][o]["src"].values())), "ct", cr.get("ok"), cr.get("n_good"), flush=True)
        res["prior_obj"] = pobj
        json.dump(res, open(op, "w"))


# ----------------------------------------------------------------------------- measure (CPU)
def ct_agrees(c):
    """CoTracker's net motion over the VLM track's span must match the VLM points' (ratio 0.8-1.25): a tracker locked on
    texture that is not the object (the refraction pattern of a transparent drop) moves less than the bubble inside it"""
    if "d_vlm" not in c:                            # older runs: the grid-agreement test only
        return True
    dv, dc = np.array(c["d_vlm"], float), np.array(c["d_ct"], float)
    nv = float(np.linalg.norm(dv))
    if nv < 5.0:
        return float(np.linalg.norm(dc - dv)) <= 3.0
    return 0.8 <= float(np.linalg.norm(dc)) / nv <= 1.25 and float(np.linalg.norm(dc - dv)) <= 0.3 * nv + 2.0


def traj(o, fps):
    """(times, positions, dense): CoTracker's dense trajectory where the VLM confirmed the object (between its first and last
    kept point) when the two trackers agree, else the VLM points"""
    c = o.get("ct") or {}
    ks = sorted(int(k) for k in o["pts"])
    if c.get("ok") and ks and ct_agrees(c):
        F, P = np.array(c["frames"]), np.array(c["pts"], float)
        m = (F >= ks[0]) & (F <= ks[-1])
        if m.sum() >= 4:
            return F[m] / fps, P[m], True
    return np.array(ks) / fps, np.array([o["pts"][str(k)] for k in ks], float), False


def local_fit(T, X, t0, half, deg=2):
    """weighted polynomial fit of X(t) around t0 (Gaussian weights, robust reweighting); returns coefficients per axis"""
    m = np.abs(T - t0) <= half
    if m.sum() < deg + 2:
        idx = np.argsort(np.abs(T - t0))[:deg + 3]
        m = np.zeros(len(T), bool)
        m[idx] = True
    tt, xx = T[m] - t0, X[m]
    wgt = np.exp(-0.5 * (tt / max(half, 1e-3)) ** 2)
    cs = []
    for d in range(2):
        ww = wgt.copy()
        for _ in range(3):                          # robust: down-weight points far from the fit
            c = np.polyfit(tt, xx[:, d], deg, w=np.sqrt(ww))
            r = np.abs(xx[:, d] - np.polyval(c, tt))
            s = 1.4826 * np.median(r) + 1e-6
            ww = wgt / (1 + (r / (2.5 * s)) ** 2)
        cs.append(c)
    return cs


def smooth(T, X, half):
    return np.array([[np.polyval(c, 0.0) for c in local_fit(T, X, t, half)] for t in T])


def vel_at(T, X, t0, half=0.3):
    cs = local_fit(T, X, t0, half)
    return np.array([np.polyval(np.polyder(c), 0.0) for c in cs])


def acc_at(T, X, t0, half=0.45):
    cs = local_fit(T, X, t0, half)
    return np.array([2 * c[0] for c in cs])


def poly_curve(T, X, max_deg=3):
    """robust (Cauchy-reweighted) polynomial of degree 1-3 per axis, the degree by BIC, sampled on 200 times"""
    tt = T - T.mean()
    n = len(T)
    best = None
    for deg in range(1, max_deg + 1):
        if n < deg + 3:
            break
        fits, rss = [], 0.0
        for d in range(2):
            wt = np.ones(n)
            for _ in range(4):
                c = np.polyfit(tt, X[:, d], deg, w=np.sqrt(wt))
                r = X[:, d] - np.polyval(c, tt)
                s = 1.4826 * np.median(np.abs(r)) + 1e-6
                wt = 1 / (1 + (r / (2.5 * s)) ** 2)
            fits.append(c)
            rss += float(np.sum(wt * r ** 2))
        bic = n * math.log(max(rss / n, 1e-9)) + 2 * (deg + 1) * math.log(n)
        if best is None or bic < best[0]:
            best = (bic, fits)
    g = np.linspace(tt[0], tt[-1], 200)
    return np.c_[np.polyval(best[1][0], g), np.polyval(best[1][1], g)]


def avg_speed(T, X, ta=None, tb=None, dense=False):
    """path length / time: a dense sub-pixel (CoTracker) track lightly smoothed (+-0.1 s); sparse VLM points through a
    robust low-order polynomial (clips <= 4 s) or a +-0.5 s local fit, so that pointing jitter does not add length"""
    m = np.ones(len(T), bool)
    if ta is not None:
        m &= T >= ta - 1e-6
    if tb is not None:
        m &= T <= tb + 1e-6
    if m.sum() < 4:
        return None
    Tm, Xm = T[m], X[m]
    dur = float(Tm[-1] - Tm[0])
    if dense:
        Y = smooth(Tm, Xm, 0.1)
    elif dur <= 4.0:
        Y = poly_curve(Tm, Xm)
    else:
        Y = smooth(Tm, Xm, 0.5)
    return float(np.linalg.norm(np.diff(Y, axis=0), axis=1).sum()) / max(dur, 1e-6)


def circle_fit(X):
    A = np.c_[2 * X, np.ones(len(X))]
    b = (X ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0.0))
    ang = np.unwrap(np.arctan2(X[:, 1] - c[1], X[:, 0] - c[0]))
    return r, float(np.ptp(ang))


def quantity(kind, T, X, q, dense=False):
    """the question's (or the prior's) kinematic quantity in px units"""
    t, t1, t2 = q.get("t"), q.get("t1"), q.get("t2")
    if len(T) < 4:
        return None, "few points"
    if kind == "speed":
        if t is None:
            return avg_speed(T, X, dense=dense), "avg"
        if t < T[0] - 0.2 or t > T[-1] + 0.2:
            return None, "time outside the track"
        return float(np.linalg.norm(vel_at(T, X, t))), "vel@t"
    if kind == "avgspeed":
        return avg_speed(T, X, t1, t2, dense=dense), "avg"
    if kind == "hvel":
        Y = smooth(T, X, 0.3)
        ia = 0 if t1 is None else int(np.argmin(np.abs(T - t1)))
        ib = len(T) - 1 if t2 is None else int(np.argmin(np.abs(T - t2)))
        return abs(Y[ib, 0] - Y[ia, 0]) / max(T[ib] - T[ia], 1e-6), "hvel"
    if kind == "accel":
        if t is not None:
            return float(np.linalg.norm(acc_at(T, X, t))), "acc@t"
        cs = [np.polyfit(T - T.mean(), X[:, d], 2) for d in range(2)]
        return float(np.hypot(2 * cs[0][0], 2 * cs[1][0])), "acc whole"
    if kind == "disp":
        Y = smooth(T, X, 0.3)
        ia = 0 if t1 is None else int(np.argmin(np.abs(T - t1)))
        ib = len(T) - 1 if t2 is None else int(np.argmin(np.abs(T - t2)))
        return float(np.linalg.norm(Y[ib] - Y[ia])), "disp"
    if kind == "path":
        s = avg_speed(T, X, t1, t2, dense=dense)
        return (None, "path") if s is None else (s * ((t2 if t2 is not None else T[-1]) - (t1 if t1 is not None else T[0])), "path")
    if kind == "orbitdiam":
        r, arc = circle_fit(smooth(T, X, 0.3))
        return 2 * r, f"circle arc {math.degrees(arc):.0f}deg"
    return None, "kind"


def gravity_mpp(T, X):
    """m/px from a free flight: the vertical px acceleration of the best quadratic segment (>= 0.4 s, R^2 >= 0.95)"""
    best = None
    if len(T) > 60:                                 # dense tracks: 60 samples are enough for the segment search
        sel = np.linspace(0, len(T) - 1, 60).round().astype(int)
        T, X = T[sel], X[sel]
    n = len(T)
    for i in range(n):
        for j in range(i + 6, n + 1):
            tt, yy = T[i:j], X[i:j, 1]
            if tt[-1] - tt[0] < 0.4:
                continue
            c = np.polyfit(tt - tt.mean(), yy, 2)
            res = yy - np.polyval(c, tt - tt.mean())
            r2 = 1 - (res ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-9)
            if c[0] > 0 and r2 >= 0.95 and (best is None or tt[-1] - tt[0] > best[0]):
                best = (tt[-1] - tt[0], 2 * c[0], r2)
    return (G / best[1], best) if best else (None, None)


def stage_measure():
    TR = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/geo/out/parse_{S}_v2.jsonl")}
    truth = {}
    if S == "val":
        v = pd.read_csv(f"{R}/data/val/validation_dataset.csv").set_index("Unnamed: 0")
        truth = v.ground_truth_posterior.to_dict()
    out = []
    for r in rows():
        p = f"{OUT}/{r['video_id']}.json"
        if not os.path.exists(p):
            continue
        J = json.load(open(p))
        fps = J["fps"]
        pr = r["priors"][0]
        rec = {"id": r["id"], "video": r["video_id"], "kind": r["q"].get("kind"), "question": r["question"][:90], "prior": pr["raw"],
               "truth": truth.get(r["id"])}
        # scale from the prior object
        po = J["objs"].get(J.get("prior_obj") or "")
        mpp, why = None, ""
        if po is None or len(po["pts"]) < 4:
            why = "prior not followed"
        else:
            T, X, dense = traj(po, fps)
            rec["prior_dense"] = dense
            Ys = smooth(T, X, 0.3)
            rng = float(np.linalg.norm(Ys.max(0) - Ys.min(0)))
            rec["prior_range_px"] = rng
            if rng < max(4.0, 0.15 * (po.get("size") or 0.0)):      # the prior object hardly moves: a wrong pick
                rec.update(mpp=None, scale_why=f"prior static ({rng:.1f}px)", why=f"prior static ({rng:.1f}px)"); out.append(rec); continue
            if pr.get("gravity"):
                mpp, info = gravity_mpp(T, X)
                why = "no free flight" if mpp is None else f"g flight {info[0]:.2f}s r2 {info[2]:.3f}"
            else:
                kind = "speed" if pr["kind"] == "speed" else "accel"
                v_px, how = quantity(kind, T, X, pr, dense)
                if v_px is None or v_px <= 0:
                    why = f"prior px quantity: {how}"
                else:
                    mpp, why = pr["value_si"] / v_px, f"prior {how} {v_px:.2f}px"
        rec.update(mpp=mpp, scale_why=why)
        if mpp is None:
            rec["why"] = why; out.append(rec); continue
        q = r["q"]
        k = q.get("kind")
        u = UNIT.get(q.get("unit") or "m", 1.0)
        objs, ends, pairs = plan(r)
        val, how = None, ""
        if ends:
            e = J["ends"].get(str(r["id"])) or {}
            if e.get("points"):
                (x1, y1), (x2, y2) = e["points"]
                val, how = float(np.hypot(x2 - x1, y2 - y1)), "ends " + e.get("how", "")
                ql_, b_ = r["question"].lower(), e.get("box")
                if b_ is not None:              # the end points must lie on the boxed object; a strip's width under half its box
                    bw_, bh_ = b_[2] - b_[0], b_[3] - b_[1]
                    pad_ = max(0.15 * max(bw_, bh_), 8.0)   # 15 %: every val object end point lay inside its box
                    on_ = lambda x_, y_: b_[0] - pad_ <= x_ <= b_[2] + pad_ and b_[1] - pad_ <= y_ <= b_[3] + pad_
                    # objects only: a scene element (lane, ceiling, room, floor ...) may be measured on another instance of it
                    # or beyond its box (a ceiling height runs from the ceiling band to the floor)
                    if not re.search(SCENE_WORDS, e.get("what") or "") and not (on_(x1, y1) and on_(x2, y2)):
                        rec["why"] = "end points off the box"; out.append(rec); continue
                    if re.search(STRIP, e.get("what") or "") and e.get("dim") in ("width", "thickness") and val > 0.5 * max(bw_, bh_):
                        rec["why"] = "strip width spans its box"; out.append(rec); continue
                # "horizontal width" / "vertical height": the extent along that image axis - the marked span's projection
                # (the VLM marks rotated text along its length), or the box extent when the span hardly runs that way
                if re.search(r"\bhorizontal(?:ly)?\b", ql_):
                    dx = abs(x2 - x1)
                    val, how = (dx, how + " x-proj") if b_ is None or dx >= 0.5 * (b_[2] - b_[0]) else (b_[2] - b_[0], how + " box w")
                elif re.search(r"\bvertical(?:ly)?\b", ql_):
                    dy = abs(y2 - y1)
                    val, how = (dy, how + " y-proj") if b_ is None or dy >= 0.5 * (b_[3] - b_[1]) else (b_[3] - b_[1], how + " box h")
        elif pairs:
            e = J["pairs"].get(str(r["id"])) or {}
            if e.get("points"):
                val, how = float(np.linalg.norm(np.subtract(*e["points"]))), "pair " + e.get("how", "")
        elif objs:
            o = J["objs"].get(objs[0])
            if o and len(o["pts"]) >= 4:
                T, X, dense = traj(o, fps)
                val, how = quantity(k, T, X, q, dense)
                how += " ct" if dense else " vlm"
        if val is None:
            rec["why"] = how or "no measurement"; out.append(rec); continue
        rec.update(px=val, value=val * mpp / u, how=how)
        if rec["truth"] is not None:
            y = float(rec["truth"])
            rec["mra"] = float((abs(rec["value"] - y) / y < 1 - TH).mean())
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(f"{R}/geo/out/kin2d_{S}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["id", "video", "kind", "truth", "value", "mra", "mpp", "how", "scale_why", "why", "question"] if c in df]
    print(df[cols].to_string(index=False, max_colwidth=50))


if a.stage == "locate":
    stage_locate()
else:
    stage_measure()
