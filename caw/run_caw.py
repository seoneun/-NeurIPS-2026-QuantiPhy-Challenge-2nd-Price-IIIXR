#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/caw/code-as-world")
import code_as_world.evaluation as E
from jinja2 import Template
from vllm import SamplingParams


def n_frames(path):
    try:
        from torchcodec.decoders import VideoDecoder
        return int(VideoDecoder(path).metadata.num_frames)
    except ImportError:
        import decord
        return len(decord.VideoReader(path))


def video_input(r):
    try:
        return E._video_input(r), ""
    except Exception as ex:
        n = n_frames(r["video"])
        if n >= E.VIDEO_NFRAMES:
            raise
        nf = max(2, (n // 2) * 2)
        old = E.VIDEO_NFRAMES
        E.VIDEO_NFRAMES = nf
        try:
            out = E._video_input(r)
        finally:
            E.VIDEO_NFRAMES = old
        note = f"{type(ex).__name__}: {str(ex)[:100]} -> nframes {nf} of {n}"
        print('', flush=True)
        return out, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model-path", default=f"{R}/models/Code-as-World-VL-9B")
    ap.add_argument("--chunk-videos", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()

    records, _, _ = E._load_csv(Path(a.input_csv), Path(a.video_dir), ".mp4")
    done = set()
    if os.path.exists(a.out_jsonl):
        with open(a.out_jsonl) as f:
            done = {json.loads(line)["original_id"] for line in f if line.strip()}
    todo = [r for r in records if r["original_id"] not in done]
    if a.max_rows:
        todo = todo[:a.max_rows]
    print('',
          flush=True)
    if not todo:
        return
    fmt = Template(E.FORMAT_PROMPT.read_text(encoding="utf-8").strip())
    tok, proc = E._load_model_tools(Path(a.model_path))
    engine = E._build_engine("9b", Path(a.model_path))
    sampling = SamplingParams(**E.SAMPLING_CONFIG, seed=E.SEED, detokenize=True, logit_bias=E._logit_bias(proc))
    print('', flush=True)

    order, byvid = [], {}
    for r in todo:
        if r["video"] not in byvid:
            byvid[r["video"]] = []
            order.append(r["video"])
        byvid[r["video"]].append(r)
    t0, n = time.time(), 0
    nchunks = (len(order) + a.chunk_videos - 1) // a.chunk_videos
    os.makedirs(os.path.dirname(os.path.abspath(a.out_jsonl)), exist_ok=True)
    with open(a.out_jsonl, "a", encoding="utf-8") as fo:
        for i in range(0, len(order), a.chunk_videos):
            chunk = [r for v in order[i:i + a.chunk_videos] for r in byvid[v]]
            tp = time.time()
            cache, inputs = {}, []
            for r in chunk:
                key = (r["video"], r.get("fps"))
                if key not in cache:
                    cache[key] = video_input(r)
                (mm, kw), _ = cache[key]
                inputs.append({"prompt_token_ids": E._build_prompt_ids(r, tok, proc, fmt),
                               "multi_modal_data": mm, "mm_processor_kwargs": kw})
            tp = time.time() - tp
            tg = time.time()
            outs = engine.generate(inputs, sampling, use_tqdm=False)
            tg = time.time() - tg
            for r, inp, o in zip(chunk, inputs, outs):
                c = o.outputs[0]
                vid, meta = inp["multi_modal_data"]["video"][0]
                note = cache[(r["video"], r.get("fps"))][1]
                rec = dict(E._record_metadata(r, 0))
                rec.update({
                    "seed": E.SEED, "video": r["video"], "ground_truth_prior": r["ground_truth_prior"],
                    "depth_info": r["depth_info"], "fps": r["fps"],
                    "response": c.text, "parsed_value": E._parse_prediction(c.text),
                    "finish_reason": E._normalise_finish_reason(c.finish_reason),
                    "num_generated_tokens": len(c.token_ids or []),
                    "num_prompt_tokens": len(o.prompt_token_ids or []),
                    "video_tensor_shape": list(vid.shape),
                    "frames_indices": [int(x) for x in meta["frames_indices"]],
                    "meta_fps": meta.get("fps"), "meta_total_num_frames": meta.get("total_num_frames"),
                    "video_note": note,
                    "prompt_text": tok.decode(inp["prompt_token_ids"]),
                })
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            n += len(chunk)
            print('', flush=True)
    print('', flush=True)


if __name__ == "__main__":
    main()
