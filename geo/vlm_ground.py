#!/usr/bin/env python
import os, sys, re, json, math, time, argparse
import numpy as np
from PIL import Image

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools")
sys.path.insert(0, f"{R}/geo")
from qp_common import load_split
import track_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--nframes", type=int, default=4)
ap.add_argument("--max_frames", type=int, default=160)
ap.add_argument("--det_frames", type=int, default=16)
ap.add_argument("--videos", default="")
ap.add_argument("--batch_videos", type=int, default=20)
ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
ap.add_argument("--gpu_mem", type=float, default=0.6)
ap.add_argument("--notext", action="store_true")
ap.add_argument("--out_tag", default="")
a = ap.parse_args()
OUT = f"{R}/geo/out/vlmground_{a.split}{a.out_tag}.jsonl"


def queries_for(recs):
    qs = {}

    def add(ent, role):
        if not ent or not ent.get("phrase"):
            return None
        rel = ent.get("rel")
        k = ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")
        if k not in qs:
            qs[k] = dict(key=k, phrase=ent["phrase"], where=list(ent.get("where") or []), rel=rel, roles=[], raw=set())
        if role not in qs[k]["roles"]:
            qs[k]["roles"].append(role)
        if ent.get("raw"):
            qs[k]["raw"].add(ent["raw"])
        if rel:
            add(dict(phrase=rel[1], where=[], raw=rel[1]), "ref")
        return k
    for r in recs:
        add(r["target"], "dist" if r["q"]["kind"] == "dist2" else "target")
        if r.get("target2"):
            add(r["target2"], "dist" if r["q"]["kind"] == "dist2" else "target2")
        if r["prior_obj"] and r["prior_obj"].get("phrase"):
            add(r["prior_obj"], "prior")
    return list(qs.values())


def expression(q):
    raws = sorted(q["raw"], key=len, reverse=True)
    e = raws[0] if raws else q["phrase"]
    w = [x for x in q["where"] if x != "#second"]
    for x in w:
        if x not in e:
            e = f"{x} {e}"
    e = re.sub(r"\s+", " ", e).strip()
    return e


def prompt_text(q):
    if "#second" in q["where"]:
        return (f'Locate every instance of "{q["phrase"]}" in the image. Report all of their bounding boxes in JSON format, '
                f'as a list [{{"bbox_2d": [x1, y1, x2, y2], "label": "{q["phrase"]}"}}, ...]. If there is none, output [].')
    e = expression(q)
    if a.notext:
        e = e + " (the object itself, not a text label or caption naming it)"
    return (f'Locate the {e} in the image. Report its bounding box in JSON format as '
            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{q["phrase"]}"}}]. If it is not in the image, output [].')


def parse_boxes(text):
    out = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', text):
        out.append([float(m.group(i)) for i in range(1, 5)])
    return out


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    mp = f"{R}/models/{a.model}"
    proc = AutoProcessor.from_pretrained(mp)
    llm = LLM(model=mp, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, limit_mm_per_prompt={"image": 1, "video": 0},
              max_num_seqs=64, enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{a.split}_v2.jsonl")]
    df = load_split(a.split)
    vpath = {str(v).strip(): p for v, p in zip(df.video_id, df.video_path)}
    byv = {}
    for r in recs:
        byv.setdefault(r["video_id"], []).append(r)
    vids = sorted(byv)
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["video_id"] for l in open(OUT)}
    vids = [v for v in vids if v not in done]
    print('', flush=True)
    t00 = time.time()
    for b0 in range(0, len(vids), a.batch_videos):
        reqs, meta = [], []
        for v in vids[b0:b0 + a.batch_videos]:
            frames = tc.load_video(vpath[v])
            N = len(frames)
            stride = max(1, int(math.ceil(N / a.max_frames)))
            fl = list(range(0, N, stride))
            frames = [frames[i] for i in fl]
            di = np.unique(np.linspace(0, len(frames) - 1, min(a.det_frames, len(frames))).round().astype(int)).tolist()
            pick = [di[int(round(x))] for x in np.linspace(0, len(di) - 1, min(a.nframes, len(di)))]
            H, W = frames[0].shape[:2]
            for q in queries_for(byv[v]):
                txt = prompt_text(q)
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}]
                pr = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for j in pick:
                    reqs.append({"prompt": pr, "multi_modal_data": {"image": Image.fromarray(frames[j])}})
                    meta.append(dict(video_id=v, key=q["key"], phrase=q["phrase"], where=q["where"], roles=q["roles"],
                                     expr=expression(q), multi="#second" in q["where"], j=int(j), W=W, H=H, prompt=txt))
            del frames
        outs = llm.generate(reqs, sp, use_tqdm=False)
        with open(OUT, "a", encoding="utf-8") as fo:
            for m, o in zip(meta, outs):
                txt = o.outputs[0].text
                bx = parse_boxes(txt)
                W, H = m["W"], m["H"]
                rel = [[x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H] for x0, y0, x1, y1 in bx]
                m.update(response=txt, boxes_raw=bx, boxes_rel1000=[[round(c, 1) for c in b] for b in rel],
                         n_prompt_tokens=len(o.prompt_token_ids or []))
                fo.write(json.dumps(m, ensure_ascii=False) + "\n")
        print('', flush=True)
    print("[vg] DONE", round(time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
