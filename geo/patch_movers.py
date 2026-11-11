#!/usr/bin/env python
import os, sys, re, json, shutil, argparse
import numpy as np

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--src", default="v2mo")
ap.add_argument("--out", default="v2mv")
a = ap.parse_args()
MT = json.load(open(f"{R}/geo/out/movertracks_{a.split}.json"))
NEED = {v: x["need"] for v, x in json.load(open(f"{R}/geo/out/movers_{a.split}.json")).items()}
ANYINST = {}
for ln in open(f"{R}/geo/out/parse_{a.split}_v2.jsonl", encoding="utf-8"):
    x = json.loads(ln)
    raw = " ".join(str(p_.get("raw") or "") for p_ in (x.get("priors") or []))
    if re.search(r"\b(?:one|each|every|any)\b", raw.lower()) and (x.get("prior_obj") or {}).get("phrase"):
        ANYINST.setdefault(x["video_id"], set()).add(x["prior_obj"]["phrase"])
SRC = f"{R}/geo/out/tracks_{a.split}_{a.src}" if a.src != "v1" else f"{R}/geo/out/tracks_{a.split}"
V1 = f"{R}/geo/out/tracks_{a.split}"
OUT = f"{R}/geo/out/tracks_{a.split}_{a.out}"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(".json"):
        shutil.copy(f"{SRC}/{f}", f"{OUT}/{f}")


def span_of(frames):
    c = np.array([o["st"]["c"] for o in frames if o and o.get("st")], float)
    ar = [o["st"]["area"] for o in frames if o and o.get("st")]
    if len(c) < 3:
        return 0.0, 0.0
    return float(np.linalg.norm(c.max(0) - c.min(0))), float(np.median(ar))


MOVING = r"\b(?:rolling|moving|falling|sliding|bouncing|flying|swinging|spinning|thrown|dropped|dropping|walking|running|swimming)\b"


def head(ph):
    w = re.findall(r"[a-z]+", (ph or "").lower())
    return w[-1] if w else ""


def med_area(frames):
    ar = [o["st"]["box"] for o in frames if o and o.get("st")]
    return float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in ar])) if ar else 0.0


n_ok = 0
for v, info in sorted(MT.items()):
    if v not in NEED or not NEED[v]:
        print(v, "not needed under the current scan rules"); continue
    free = sorted([t for t in info["tracks"] if t["taken"] is None], key=lambda t: -t["cover"] * t["span"])
    need = NEED[v]
    if not free:
        print(v, "no free mover"); continue
    if len(free) > 1 and (len(need) != 1 or free[0]["cover"] * free[0]["span"] < 3 * free[1]["cover"] * free[1]["span"]):
        print(v, "ambiguous movers", [(round(t["cover"], 2), round(t["span"])) for t in free], "need", need); continue
    mv = free[0]
    p = f"{SRC}/{v}.json"
    if not os.path.exists(p):
        p = f"{V1}/{v}.json"
    tj = json.load(open(p))
    N = len(tj["fidx"])
    if len(mv["objs"]) != N:
        print(v, "frame count mismatch", len(mv["objs"]), N); continue
    heads = {head(ph) for ph in need}
    qall = [q for qk in ("queries", "queries_d") for q in tj.get(qk, [])]
    a_mv = med_area(mv["objs"])
    bad = []
    for q in qall:
        if q.get("phrase") in need or head(q.get("phrase")) not in heads or q.get("phrase") not in ANYINST.get(v, set()):
            continue
        fr = (tj["objs"].get(str(q.get("obj"))) if (tj.get("objs") and q.get("obj") is not None) else q.get("frames")) or []
        if len(fr) == N and a_mv > 0 and med_area(fr) > 9 * a_mv:
            bad.append(q.get("key"))
    if bad and os.environ.get("PM_SAMECLASS_SKIP", "1") == "1":
        print(v, "same-class object much larger than the mover", bad); continue
    if bad:
        print(v, "same-class object much larger than the mover (kept: measure re-measures the prior)", bad)
    need = dict(need)
    for q in qall:
        ph = q.get("phrase") or ""
        if ph not in need and head(ph) in heads and re.search(MOVING, ph.lower()):
            need[ph] = "target"
    hit = []
    if "objs" in tj and tj.get("objs") is not None:
        oid = str(max([int(x) for x in tj["objs"]] + [0]) + 1)
        tj["objs"][oid] = mv["objs"]
        for qk in ("queries", "queries_d"):
            for q in tj.get(qk, []):
                if q.get("phrase") not in need:
                    continue
                cur = tj["objs"].get(str(q.get("obj"))) if q.get("obj") is not None else None
                if cur is not None and len(cur) == N:
                    s_, ar_ = span_of(cur)
                    if s_ >= max(8.0, 0.5 * np.sqrt(max(ar_, 1.0))):
                        continue
                q.update(obj=int(oid), best_gd=max(q.get("best_gd") or 0, 0.5), best_owl=max(q.get("best_owl") or 0, 0.5),
                         best_score=max(q.get("best_score") or 0, 0.5), det_iou=[1.0] * max(1, len(q.get("det_iou") or [])),
                         molmo=True, mover=True)
                hit.append((qk, q.get("key")))
    else:
        for q in tj.get("queries", []):
            if q.get("phrase") not in need:
                continue
            cur = q.get("frames")
            if cur is not None and len(cur) == N:
                s_, ar_ = span_of(cur)
                if s_ >= max(8.0, 0.5 * np.sqrt(max(ar_, 1.0))):
                    continue
            q.update(frames=mv["objs"], best_score=max(q.get("best_score") or 0, 0.5),
                     det_iou=[1.0] * max(1, len(q.get("det_iou") or [])), mover=True)
            hit.append(("queries", q.get("key")))
    if hit:
        json.dump(tj, open(f"{OUT}/{v}.json", "w"))
        n_ok += 1
    print(v, "patched" if hit else "no query matched", hit, "mover span", round(mv["span"]), "cover", round(mv["cover"], 2), "from", p.split("/")[-2])
print("DONE", a.src, "->", a.out, "videos patched", n_ok)
