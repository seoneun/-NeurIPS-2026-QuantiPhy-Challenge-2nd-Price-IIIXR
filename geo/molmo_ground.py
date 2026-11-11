#!/usr/bin/env python
import os, sys, re, json, time, argparse
import numpy as np, pandas as pd
from PIL import Image

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--geo", default="geo_v4ntrj5pbsrng_d_vw")
ap.add_argument("--gpu_mem", type=float, default=0.45)
a = ap.parse_args()
OUT = f"{R}/geo/out/molmoground_{a.split}.jsonl"
COORD_RE = re.compile(r'coords="([0-9\t:;, .]+)"')
PT_RE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
FAIL = ("prior_static_track", "target_static_track", "prior_low_det", "target_low_det", "target_no_track", "prior_no_track",
        "target2_no_track", "target2_low_det", "prior_det_track_iou", "target_det_track_iou")


def main():
    from vllm import LLM, SamplingParams
    raw = pd.read_csv(f"{R}/preds/{a.split}_{a.geo}_raw.csv").set_index("Unnamed: 0")
    parse = {int(json.loads(l)["id"]): json.loads(l) for l in open(f"{R}/geo/out/parse_{a.split}_v2.jsonl")}
    why = raw.meas_why.astype(str).str.extract(r"^([a-z0-9_]+)")[0]
    items = set()
    for i, w in why.items():
        if w not in FAIL:
            continue
        pr = parse.get(int(i))
        if not pr:
            continue
        role = "prior_obj" if w.startswith("prior") else ("target2" if w.startswith("target2") else "target")
        o = pr.get(role)
        if o and o.get("phrase"):
            items.add((pr["video_id"], o["phrase"]))
    items = sorted(items)
    df = load_split(a.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    llm = LLM(model=f"{R}/models/Molmo2-8B", trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=a.gpu_mem,
              seed=0, limit_mm_per_prompt={"image": 1}, max_num_seqs=16)
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    convs, meta = [], []
    byv = {}
    for v, ph in items:
        byv.setdefault(v, []).append(ph)
    for v, phs in byv.items():
        tj = None
        for sub in ("_v2", "_v3nt", ""):
            p = f"{R}/geo/out/tracks_{a.split}{sub}/{v}.json"
            if os.path.exists(p):
                tj = json.load(open(p)); break
        frames = tc.load_video(vpath[v])
        fidx = tj["fidx"] if tj else list(range(len(frames)))
        ks = sorted(set(int(round((len(fidx) - 1) * f)) for f in (0.2, 0.5, 0.8)))
        H, W = frames[0].shape[:2]
        for ph in phs:
            for k in ks:
                if fidx[k] >= len(frames):
                    continue
                convs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": Image.fromarray(frames[fidx[k]])},
                                                           {"type": "text", "text": f"Point to the {ph}."}]}])
                meta.append(dict(video_id=v, phrase=ph, k=int(k), W=W, H=H))
        del frames
    print('', flush=True)
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False)
    with open(OUT, "w", encoding="utf-8") as fo:
        for m, o in zip(meta, outs):
            txt = o.outputs[0].text
            pts = []
            for cm in COORD_RE.finditer(txt):
                for pm in PT_RE.finditer(cm.group(1)):
                    pts.append([float(pm.group(2)) / 1000 * m["W"], float(pm.group(3)) / 1000 * m["H"]])
            m.update(points=[[round(x, 1), round(y, 1)] for x, y in pts], response=txt[:200])
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("[mg] DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
