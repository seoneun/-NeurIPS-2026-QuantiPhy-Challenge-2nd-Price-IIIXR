import os, sys, re, json, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "tools"))
from qp_common import load_split, R
from rules import parse_question, parse_prior, parse_depth, match_depth_keys, clean_text, COLORS

OUT = f"{R}/geo/out"
SPATIAL = ["upper left", "upper right", "lower left", "lower right", "top left", "top right", "bottom left",
           "bottom right", "left", "right", "upper", "lower", "top", "bottom", "middle", "center", "centre",
           "front", "back", "near", "far", "closest", "farthest", "first", "second", "third", "last",
           "largest", "biggest", "smallest", "big", "small", "large", "tall", "short"]
CUT = r"\s+(?:in the|on the|at the|near the|with the|with a|that|which|who|above the|below the|under the|beside the|next to|behind the|in front of|from the|placed|standing|sitting|walking on|moving on|cutting|closest to|on top of|in his|in her|in its|of the|of a|of)\s+"


def det_phrase(p):
    if not p:
        return None, []
    full = p
    where = []
    for w in SPATIAL:
        if re.search(r"\b" + w + r"\b", full) and not any(w in x for x in where):
            where.append(w)
    head = re.split(CUT, p)[0]
    head = re.sub(r"^(?:the|a|an|two|2|both)\s+", "", head).strip()
    for w in ["left", "right", "upper", "lower", "top", "bottom", "middle", "center", "front", "back", "far", "near", "first", "second"]:
        head = re.sub(r"\b" + w + r"\b", " ", head)
    head = re.sub(r"\s+", " ", head).strip(" ,'")
    return (head or None), where


def build_prompt(q, prior, keys):
    ks = ", ".join(keys) if keys else "none"
    return f"""You turn a physics question about a video into a JSON record for an object detector. Reply with one JSON object only.

Fields:
"target": the single visible thing whose size or motion is asked, as a short noun phrase (1-4 words) you would type into an object detector. Keep colour/size words, drop location clauses and time. If a part is asked (e.g. the tire of a car), name the part ("tire").
"target_where": position words for the target given in the question (e.g. "left", "upper left", "middle", "front"), or "none".
"target2": for a distance between two things, the second thing as a short noun phrase, else "".
"target_whole": true if the answer is the extent of the whole target, or its motion, or the centre-to-centre distance between target and target2; false if it is a part of something not separately visible, a gap, clearance, height above the floor/ground/water, scene layout (lane, road, court, tiles, room), an orbit, or anything else.
"prior_object": short noun phrase for the thing the given fact is about; "gravity" if it is gravitational acceleration with no object named; "person" for walking speed.
"prior_whole": same meaning as target_whole, for the given fact.
"target_depth_key", "target2_depth_key", "prior_depth_key": the depth key that refers to that thing (exactly as listed), or "none".

Example 1
Question: What is the speed of the bycicle on the upper left conrner at 3.0s in m/s?
Given fact: pedestrian walking speed ~1.1 m/s
Depth keys: none
{{"target": "bicycle", "target_where": "upper left", "target2": "", "target_whole": true, "prior_object": "person", "prior_whole": true, "target_depth_key": "none", "target2_depth_key": "none", "prior_depth_key": "none"}}

Example 2
Question: What is the height of the pier deck above the water surface in meters?
Given fact: speed of the boat = 1.5 m/s
Depth keys: boat, pier, stairs
{{"target": "pier", "target_where": "none", "target2": "", "target_whole": false, "prior_object": "boat", "prior_whole": true, "target_depth_key": "pier", "target2_depth_key": "none", "prior_depth_key": "boat"}}

Now this one
Question: {q}
Given fact: {prior}
Depth keys: {ks}
"""


def run_llm(prompts):
    from vllm import LLM, SamplingParams
    llm = LLM(model=f"{R}/models/Qwen3-VL-8B-Instruct", max_model_len=4096, gpu_memory_utilization=0.35, seed=0,
              limit_mm_per_prompt={"video": 0, "image": 0}, enable_prefix_caching=True, max_num_seqs=128)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    texts = [f"<|im_start|>system\nYou are a precise information extraction assistant.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" for p in prompts]
    outs = llm.generate(texts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        s2 = s.replace("True", "true").replace("False", "false")
        try:
            return json.loads(s2)
        except Exception:
            return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--no_llm", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = load_split(a.split)
    recs = []
    for _, row in df.iterrows():
        q = parse_question(row.question, row.inference_type)
        pr = parse_prior(row.ground_truth_prior)
        dep = parse_depth(row.depth_info) if str(row.video_type)[1] == "3" else []
        keys = sorted(set(d["key"] for d in dep))
        recs.append(dict(id=int(row["Unnamed: 0"]), video_id=str(row.video_id).strip(), video_source=row.video_source,
                         video_type=row.video_type, inference_type=row.inference_type, fps=float(row.fps),
                         cat=row.inference_type[0] + row.video_type[1], question=row.question,
                         prior_text=row.ground_truth_prior, q=q, priors=pr, depth=dep, depth_keys=keys))
    llm_path = f"{OUT}/llm_parse_{a.split}.jsonl"
    if not a.no_llm:
        t0 = time.time()
        prompts = [build_prompt(clean_text(r["question"]), clean_text(r["prior_text"]).replace("\n", "; "), r["depth_keys"]) for r in recs]
        raw = run_llm(prompts)
        with open(llm_path, "w") as f:
            for r, t in zip(recs, raw):
                f.write(json.dumps(dict(id=r["id"], raw=t)) + "\n")
        print('', flush=True)
    llm = {}
    if os.path.exists(llm_path):
        for l in open(llm_path):
            d = json.loads(l); llm[d["id"]] = parse_json(d["raw"])
    n_llm_used = dict(target=0, target2=0, prior=0, tkey=0, pkey=0)
    for r in recs:
        L = llm.get(r["id"], {})
        r["llm"] = L
        q = r["q"]
        tp, where = det_phrase(q["target"])
        src = "rule"
        if (tp is None or len(tp.split()) > 3) and L.get("target"):
            tp, src = str(L["target"]).lower().strip(), "llm"; n_llm_used["target"] += 1
        if not where and L.get("target_where") and str(L["target_where"]).lower() not in ("none", ""):
            where = [w for w in SPATIAL if re.search(r"\b" + w + r"\b", str(L["target_where"]).lower())][:2]
        r["target"] = dict(phrase=tp, where=where, src=src, raw=q["target"])
        t2 = None
        if q["target2"] == "#second":
            t2 = dict(phrase=tp, where=["#second"], src="rule", raw=q["target"])
        elif q["target2"]:
            p2, w2 = det_phrase(q["target2"]); s2 = "rule"
            if (p2 is None or len(p2.split()) > 3) and L.get("target2"):
                p2, s2 = str(L["target2"]).lower().strip(), "llm"; n_llm_used["target2"] += 1
            t2 = dict(phrase=p2, where=w2, src=s2, raw=q["target2"])
        elif q["kind"] == "dist2" and L.get("target2"):
            t2 = dict(phrase=str(L["target2"]).lower().strip(), where=[], src="llm", raw=None); n_llm_used["target2"] += 1
        r["target2"] = t2
        p0 = r["priors"][0] if r["priors"] else {}
        pp, pw = det_phrase(p0.get("obj"))
        psrc = "rule"
        if p0.get("gravity") and not pp:
            pp = None
        elif (pp is None or len(pp.split()) > 3) and L.get("prior_object") and str(L["prior_object"]).lower() != "gravity":
            pp, psrc = str(L["prior_object"]).lower().strip(), "llm"; n_llm_used["prior"] += 1
        r["prior_obj"] = dict(phrase=pp, where=pw, src=psrc, raw=p0.get("obj"))
        def dkey(phrase_raw, phrase, llm_key):
            if not r["depth_keys"]:
                return [], "none"
            ks, sc = match_depth_keys(phrase_raw or phrase, r["depth_keys"])
            if ks and sc >= 0.5:
                return ks, "rule"
            if llm_key and llm_key in r["depth_keys"]:
                return [llm_key], "llm"
            if ks:
                return ks, "rule_weak"
            return [], "none"
        r["target_dkeys"], r["target_dsrc"] = dkey(q["target"], tp, L.get("target_depth_key"))
        if r["target_dsrc"] == "llm": n_llm_used["tkey"] += 1
        r["target2_dkeys"], _ = dkey(q["target2"] if t2 and q["target2"] != "#second" else None, t2["phrase"] if t2 else None, L.get("target2_depth_key")) if t2 else ([], "none")
        r["prior_dkeys"], r["prior_dsrc"] = dkey(p0.get("obj"), pp, L.get("prior_depth_key")) if pp else ([], "none")
        if r["prior_dsrc"] == "llm": n_llm_used["pkey"] += 1
        r["target_whole"] = bool(L.get("target_whole", True)) if L else True
        r["prior_whole"] = bool(L.get("prior_whole", True)) if L else True
    with open(f"{OUT}/parse_{a.split}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    agree = dict(target=0, prior=0, n=0)
    for r in recs:
        L = r["llm"]
        if not L:
            continue
        agree["n"] += 1
        a1 = set(re.findall(r"[a-z]+", (r["target"]["phrase"] or "")))
        b1 = set(re.findall(r"[a-z]+", str(L.get("target", "")).lower()))
        agree["target"] += bool(a1 & b1)
        a2 = set(re.findall(r"[a-z]+", (r["prior_obj"]["phrase"] or "gravity")))
        b2 = set(re.findall(r"[a-z]+", str(L.get("prior_object", "")).lower()))
        agree["prior"] += bool(a2 & b2)
    kinds = pd.Series([r["q"]["kind"] for r in recs]).value_counts().to_dict()
    print(json.dumps(dict(split=a.split, rows=len(recs), kinds=kinds, llm_used=n_llm_used, agree=agree,
                          rule_ok=int(sum(r["q"]["ok"] for r in recs)),
                          whole_false=int(sum(not r["target_whole"] for r in recs)))), flush=True)


if __name__ == "__main__":
    main()
