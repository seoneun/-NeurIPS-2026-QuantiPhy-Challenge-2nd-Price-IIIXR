#!/usr/bin/env python
import argparse, json, os, sys, time
from pathlib import Path

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/caw/code-as-world")
sys.path.insert(0, f"{R}/caw")
import code_as_world.evaluation as E
import qwen_vl_utils.vision_process as VP
import torch
from jinja2 import Template
from vllm import SamplingParams
from run_caw import video_input

_SHIFT = {"v": 0.0}


class _TorchProxy:
    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def linspace(start, end, steps, *a, **kw):
        s = _SHIFT["v"]
        if s == 0 or steps < 2:
            return torch.linspace(start, end, steps, *a, **kw)
        step = (float(end) - float(start)) / (steps - 1)
        return torch.linspace(float(start) + s * step, float(end) + s * step, steps, *a, **kw).clamp(max=float(end))


VP.torch = _TorchProxy()

_RANGE = {"v": None}
_orig_process_video = E._process_video


def _process_video_range(video):
    if _RANGE["v"] is None:
        return _orig_process_video(video)
    vs, ve = _RANGE["v"]
    vision_info = {"video": video, "min_pixels": E.MIN_PIXELS, "max_pixels": E.MAX_PIXELS, "video_fps": E.VIDEO_FPS,
                   "nframes": E.VIDEO_NFRAMES, "video_start": vs, "video_end": ve}
    try:
        return E.fetch_video(vision_info, return_video_sample_fps=True, return_video_metadata=True)
    except ValueError as exc:
        m = E.NFRAMES_INTERVAL_PATTERN.search(str(exc))
        if m is None:
            raise
        lo, hi, req = (int(x) for x in m.groups())
        if hi < lo:
            raise
        vision_info["nframes"] = max(2, (hi // 2) * 2)
        return E.fetch_video(vision_info, return_video_sample_fps=True, return_video_metadata=True)


E._process_video = _process_video_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--model-path", default=f"{R}/models/Code-as-World-VL-9B")
    ap.add_argument("--engine-key", default="9b")
    ap.add_argument("--chunk-videos", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--time-map", default="")
    a = ap.parse_args()
    tmap = {int(k): v for k, v in json.load(open(a.time_map)).items()} if a.time_map else {}
    cfgs = []
    for c in a.configs.split(","):
        f = c.split(":")
        name, t, p, k, n, s = f[:6]
        nf = int(f[6]) if len(f) > 6 else E.VIDEO_NFRAMES
        mp = int(f[7]) if len(f) > 7 else E.MAX_PIXELS
        tz = int(f[8]) if len(f) > 8 else 0
        think = len(f) > 9 and f[9] == "think"
        cfgs.append(dict(name=name, temperature=float(t), top_p=float(p), top_k=int(k), n=int(n), shift=float(s), nframes=nf,
                         max_pixels=mp, tz=tz, think=think))
    records, _, _ = E._load_csv(Path(a.input_csv), Path(a.video_dir), ".mp4")
    if a.max_rows:
        records = records[:a.max_rows]
    fmt = Template(E.FORMAT_PROMPT.read_text(encoding="utf-8").strip())
    tok, proc = E._load_model_tools(Path(a.model_path))
    E.MODEL_SETTINGS.setdefault("32b", {"gpu_memory_utilization": 0.92, "enable_chunked_prefill": True})
    engine = E._build_engine(a.engine_key, Path(a.model_path))
    os.makedirs(a.out_dir, exist_ok=True)
    order, byvid = [], {}
    for r in records:
        if r["video"] not in byvid:
            byvid[r["video"]] = []
            order.append(r["video"])
        byvid[r["video"]].append(r)
    for cfg in cfgs:
        out = f"{a.out_dir}/{cfg['name']}.jsonl"
        done = set()
        if os.path.exists(out):
            with open(out) as f:
                done = {json.loads(l)["original_id"] for l in f if l.strip()}
        sc = dict(E.SAMPLING_CONFIG)
        sc.update(temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"], n=cfg["n"])
        if cfg.get("think"):
            sc.update(max_tokens=1536)
        sampling = SamplingParams(**sc, seed=E.SEED, detokenize=True, logit_bias=E._logit_bias(proc))
        _SHIFT["v"] = cfg["shift"]
        E.VIDEO_NFRAMES, E.MAX_PIXELS = cfg["nframes"], cfg["max_pixels"]
        print('', flush=True)
        t0, nrow = time.time(), 0
        with open(out, "a", encoding="utf-8") as fo:
            for i in range(0, len(order), a.chunk_videos):
                chunk = [r for v in order[i:i + a.chunk_videos] for r in byvid[v] if r["original_id"] not in done]
                if not chunk:
                    continue
                cache, inputs = {}, []
                for r in chunk:
                    rng = tuple(tmap[r["original_id"]]) if (cfg["tz"] and r["original_id"] in tmap) else None
                    key = (r["video"], r.get("fps"), rng)
                    if key not in cache:
                        _RANGE["v"] = rng
                        try:
                            cache[key] = video_input(r)
                        finally:
                            _RANGE["v"] = None
                    (mm, kw), _ = cache[key]
                    pids = E._build_prompt_ids(r, tok, proc, fmt)
                    if cfg.get("think"):
                        pids = pids + tok.encode("<think>" + chr(10), add_special_tokens=False)
                    inputs.append({"prompt_token_ids": pids,
                                   "multi_modal_data": mm, "mm_processor_kwargs": kw})
                outs = engine.generate(inputs, sampling, use_tqdm=False)
                for r, inp, o in zip(chunk, inputs, outs):
                    vid, meta = inp["multi_modal_data"]["video"][0]
                    rec = {"original_id": r["original_id"], "video_id": r.get("video_id"), "config": cfg["name"],
                           "responses": [c.text for c in o.outputs],
                           "parsed": [E._parse_prediction(c.text.split("</think>")[-1] if cfg.get("think") else c.text) for c in o.outputs],
                           "frames_indices": [int(x) for x in meta["frames_indices"]]}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                nrow += len(chunk)
                print('', flush=True)
    print('', flush=True)


if __name__ == "__main__":
    main()
