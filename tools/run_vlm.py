import os, sys, time, json, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--mode", default="video", choices=["video", "blind"])
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=f"{R}/models/Qwen3-VL-8B-Instruct")
ap.add_argument("--max_frames", type=int, default=64)
ap.add_argument("--short_side", type=int, default=480)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--chunk_videos", type=int, default=40)
ap.add_argument("--max_tokens", type=int, default=48)
args = ap.parse_args()

t0 = time.time()
df = load_split(args.split)
vids = sorted(df.video_id.unique())
vids = vids[args.shard::args.nshards]
df = df[df.video_id.isin(vids)].copy()
if args.limit:
    df = df.head(args.limit)
    vids = sorted(df.video_id.unique())
print('', flush=True)

from vllm import LLM, SamplingParams

llm = LLM(model=args.model, max_model_len=20480, gpu_memory_utilization=0.85, seed=0,
          limit_mm_per_prompt={"video": 1 if args.mode == "video" else 0, "image": 0},
          enable_prefix_caching=True, max_num_seqs=32)
greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
retry = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=args.max_tokens, seed=0)
t_load = time.time() - t0


def load_video(path, fps_meta):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = min(h, w)
        if s > args.short_side:
            sc = args.short_side / s
            fr = cv2.resize(fr, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    n = len(frames)
    k = min(n, args.max_frames)
    if k % 2 and k > 1:
        k -= 1
    idx = np.linspace(0, n - 1, k).round().astype(int).tolist()
    arr = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])
    del frames
    meta = {"fps": float(fps_meta), "duration": n / float(fps_meta), "total_num_frames": n,
            "frames_indices": idx, "video_backend": "opencv", "do_sample_frames": False}
    return arr, meta, n


def user_text(row, n_frames=None, k=None):
    ctx = build_context_prefix(row["ground_truth_prior"], depth_for(row))
    vinfo = ""
    if args.mode == "video":
        fps = float(row["fps"])
        vinfo = (f"The video is {n_frames / fps:.2f} seconds long ({n_frames} frames at {fps:g} fps); "
                 f"{k} frames are sampled uniformly from it and each carries its timestamp. ")
    return f"{vinfo}{ctx}{row['question']}\n\nPlease answer the question with numbers and units ONLY. No explanation needed."


def prompt_str(text):
    vis = "<|vision_start|><|video_pad|><|vision_end|>" if args.mode == "video" else ""
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{vis}{text}<|im_end|>\n<|im_start|>assistant\n")


rows_out = []
t_gen = 0.0
for c0 in range(0, len(vids), args.chunk_videos):
    cv = vids[c0:c0 + args.chunk_videos]
    sub = df[df.video_id.isin(cv)]
    reqs, recs = [], []
    vcache = {}
    for _, row in sub.iterrows():
        if args.mode == "video":
            if row.video_id not in vcache:
                vcache[row.video_id] = load_video(row.video_path, row.fps)
            arr, meta, n = vcache[row.video_id]
            text = user_text(row, n, arr.shape[0])
            reqs.append({"prompt": prompt_str(text), "multi_modal_data": {"video": [(arr, meta)]}})
        else:
            text = user_text(row)
            reqs.append({"prompt": prompt_str(text)})
        recs.append((row, text))
    tg = time.time()
    outs = llm.generate(reqs, greedy, use_tqdm=False)
    resp = [o.outputs[0].text for o in outs]
    vals = [parse_number(r) for r in resp]
    bad = [i for i, v in enumerate(vals) if np.isnan(v) or v == 0]
    n_retry = 0
    for attempt in range(3):
        if not bad:
            break
        o2 = llm.generate([reqs[i] for i in bad], retry, use_tqdm=False)
        n_retry += len(bad)
        nb = []
        for i, o in zip(bad, o2):
            v = parse_number(o.outputs[0].text)
            if not (np.isnan(v) or v == 0):
                resp[i], vals[i] = o.outputs[0].text, v
            else:
                nb.append(i)
        bad = nb
    t_gen += time.time() - tg
    for (row, text), r, v in zip(recs, resp, vals):
        d = {c: row[c] for c in OUT_COLS if c in row.index}
        d["parsed_value"] = v
        d["raw_response"] = r
        d["prompt_text"] = text
        d["n_frames_sampled"] = vcache[row.video_id][0].shape[0] if args.mode == "video" else 0
        if args.mode == "video":
            d["frame_hw"] = "x".join(map(str, vcache[row.video_id][0].shape[1:3]))
        rows_out.append(d)
    print('', flush=True)
    del vcache

o = pd.DataFrame(rows_out)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
o[OUT_COLS].to_csv(args.out, index=False)
o.to_csv(args.out.replace(".csv", "_raw.csv"), index=False)
info = {"split": args.split, "mode": args.mode, "rows": len(o), "videos": len(vids),
        "invalid": int(o.parsed_value.isna().sum() + (o.parsed_value == 0).sum()),
        "load_s": round(t_load, 1), "gen_s": round(t_gen, 1), "total_s": round(time.time() - t0, 1),
        "max_frames": args.max_frames, "short_side": args.short_side}
json.dump(info, open(args.out.replace(".csv", "_info.json"), "w"), indent=1)
print("DONE", json.dumps(info), flush=True)
