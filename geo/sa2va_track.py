#!/usr/bin/env python
import os, sys, json, time, argparse
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

SIDE = {"left", "right", "top", "bottom", "front", "back", "upper left", "upper right", "lower left", "lower right"}


def describe(phrase, where):
    w = [x for x in (where or []) if not str(x).startswith("#")]
    if not w:
        return f"the {phrase}"
    w = w[0]
    if w in SIDE:
        return f"the {phrase} on the {w}"
    if w in ("center", "middle", "foreground", "background"):
        return f"the {phrase} in the {w}"
    return f"the {w} {phrase}"


def track_file(split, tracks, v):
    for sub in (f"_{tracks}", "_v2", ""):
        p = f"{R}/geo/out/tracks_{split}{sub}/{v}.json"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--tracks", default="v2mv")
    ap.add_argument("--keys", default="spread", choices=["spread", "first"])
    ap.add_argument("--nkeys", type=int, default=5)
    ap.add_argument("--videos", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--filter", default="", choices=["", "scene_weak"])
    a = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    from qwen_vl_utils import process_vision_info
    MP = f"{R}/models/Sa2VA-Qwen3-VL-4B-SAM3"
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_sa2va_qwen.Sa2VAChatModelQwen", MP)
    _orig_init = cls.__init__

    def _init(self, *args, **kw):
        _orig_init(self, *args, **kw)
        if "all_tied_weights_keys" not in self.__dict__:
            try:
                self.post_init()
            except Exception as e:
                print("post_init failed:", repr(e)[:200], flush=True)
                self.all_tied_weights_keys = {}
    cls.__init__ = _init
    model = cls.from_pretrained(MP, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
    mod = sys.modules[type(model).__module__]
    seg_id = proc.tokenizer.convert_tokens_to_ids("[SEG]")
    smod = sys.modules[type(model.grounding_encoder).__module__]
    fresh = smod.SAM3()
    lr_ = fresh.load_state_dict(model.grounding_encoder.state_dict(), strict=False)
    print("SAM3 rebuilt: missing", len(lr_.missing_keys), "unexpected", len(lr_.unexpected_keys), flush=True)
    model.grounding_encoder = fresh.to(device="cuda", dtype=torch.bfloat16).eval()
    ge = model.grounding_encoder

    @torch.no_grad()
    def segment(frames, desc):
        N = len(frames)
        if a.keys == "first":
            key_idx = list(range(min(a.nkeys, N)))
        else:
            key_idx = sorted(set(np.linspace(0, N - 1, min(a.nkeys, N)).round().astype(int).tolist()))
        pil = [Image.fromarray(f) for f in frames]
        content = [{"type": "image", "image": pil[k]} for k in key_idx] + [{"type": "text", "text": f"Please segment {desc}."}]
        messages = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm = proc(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
                  min_pixels=model.min_pixels, max_pixels=model.max_pixels).to(model.device)
        gen = model.model.generate(**mm, max_new_tokens=48, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
        out_ids = gen.sequences[0][mm.input_ids.shape[1]:]
        pred = proc.batch_decode([out_ids], skip_special_tokens=False)[0].strip()
        last = torch.cat([item[-1][0] for item in gen.hidden_states], dim=0)
        seg_h = mod.get_seg_hidden_states(last, gen.sequences[0][:-1], seg_id=seg_id)
        if len(seg_h) == 0:
            return pred, 0, [None] * N
        seg = model.text_hidden_fcs(seg_h)[0][None]
        g = torch.stack([ge.preprocess_image(torch.from_numpy(model.extra_image_processor.apply_image(f)).permute(2, 0, 1).contiguous())
                         for f in frames]).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = ge.get_sam2_embeddings(g)
            for k in key_idx:
                ge.sam2_model.add_language_embd(state, int(k), 100, seg[0][None][None], inference=True)
            outs = {}
            for o in ge.sam2_model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=state["num_frames"],
                                                      reverse=False, propagate_preflight=True):
                outs[int(o[0])] = o[3]
        h, w = frames[0].shape[:2]
        st = []
        for j in range(N):
            m = outs.get(j)
            if m is None:
                st.append(None); continue
            m = F.interpolate(m.float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            m = (m.sigmoid() > 0.5).cpu().numpy()
            st.append(tc.mask_stats(m))
        del state, outs, g
        torch.cuda.empty_cache()
        return pred, int(len(seg_h)), st

    os.environ.setdefault("QP_SCENE_HEAD", "1")
    from measure_v3 import scene_phrase
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    t_owl = json.load(open(tp))["T_owl"] if os.path.exists(tp) else 0.2

    def weak_q(q, tj):
        N = len(tj["fidx"])
        fr = tj["objs"].get(str(q.get("obj"))) if (tj.get("objs") and q.get("obj") is not None) else None
        if fr is None or len(fr) != N or sum(1 for o in fr if o and o.get("st")) < 6:
            return True
        gd, owl = q.get("best_gd") or 0, q.get("best_owl") or 0
        if not (gd >= 0.3 or (gd >= 0.15 and owl >= t_owl)):
            return True
        di = q.get("det_iou") or []
        return bool(di and float(np.mean(di)) < 0.5)

    df = load_split(a.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    vids = sorted(set(vpath)) if not a.videos else a.videos.split(",")
    plan_regions = {}
    if a.plan:
        for ln in open(a.plan, encoding="utf-8"):
            x = json.loads(ln)
            for ph in ((x.get("program") or {}).get("regions") or {}).values():
                if isinstance(ph, str) and ph.strip():
                    plan_regions.setdefault(x["video_id"], set()).add(ph.strip())
        vids = [v for v in vids if v in plan_regions]
    outp = f"{R}/geo/out/sa2va_{a.split}{a.out}.json"
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    t0 = time.time()
    for vi, v in enumerate(vids):
        if v in res:
            continue
        tj = track_file(a.split, a.tracks, v)
        if tj is None:
            continue
        keys = {}
        if a.plan:
            for ph in sorted(plan_regions.get(v, ())):
                keys[ph] = (ph, ["#plan"])
        for qk in (() if a.plan else ("queries", "queries_d")):
            for q in tj.get(qk, []):
                if q.get("key") and q.get("phrase"):
                    if a.filter == "scene_weak" and not (scene_phrase(q["phrase"]) or weak_q(q, tj)):
                        continue
                    keys.setdefault(q["key"], (q["phrase"], q.get("where") or []))
        if not keys:
            continue
        frames_all = tc.load_video(vpath[v])
        frames = [frames_all[min(i, len(frames_all) - 1)] for i in tj["fidx"]]
        del frames_all
        rv = {}
        for key, (ph, wh) in keys.items():
            desc = ph if wh == ["#plan"] else describe(ph, wh)
            try:
                pred, nseg, st = segment(frames, desc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pred, nseg, st = "OOM", 0, [None] * len(frames)
            rv[key] = dict(desc=desc, text=pred[:200], nseg=nseg, st=st)
            cov = sum(1 for s in st if s) / max(1, len(st))
            print(f"  {v} {desc!r}: seg {nseg} cover {cov:.2f} '{pred[:60]}'", flush=True)
        res[v] = rv
        if vi % 10 == 0 or vi == len(vids) - 1:
            json.dump(res, open(outp, "w"))
        print(f"[{vi + 1}/{len(vids)}] {v} keys {len(keys)} N {len(frames)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(outp, "w"))
    print("DONE", len(res), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
