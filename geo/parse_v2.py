import os, sys, re, json, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "tools"))
from qp_common import load_split, R
from rules_v2 import parse_question, parse_prior, parse_depth, match_depth_keys, clean_text, fix_object_phrase
from parse import SPATIAL, CUT, parse_json

OUT = f"{R}/geo/out"
CUT2 = CUT[:-4] + r"|before the|after the|ahead of|following|beside|besides|riding on)\s+"


def det_phrase(p):
    if not p:
        return None, []
    where = []
    for w in SPATIAL:
        if re.search(r"\b" + w + r"\b", p) and not any(w in x for x in where):
            where.append(w)
    head = re.split(CUT2, p)[0]
    head = re.sub(r"^(?:the|a|an|two|2|both|each)\s+", "", head).strip()
    for w in ["left", "right", "upper", "lower", "top", "bottom", "middle", "center", "front", "back", "far", "near",
              "first", "second", "most", "closest", "nearest"]:
        head = re.sub(r"\b" + w + r"\b", " ", head)
    head = re.sub(r"\s+", " ", head).strip(" ,'")
    return (head or None), where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    df = load_split(a.split)
    v1 = {json.loads(l)["id"]: json.loads(l) for l in open(f"{OUT}/parse_{a.split}.jsonl")}
    llm = {}
    for l in open(f"{OUT}/llm_parse_{a.split}.jsonl"):
        d = json.loads(l); llm[d["id"]] = parse_json(d["raw"])
    recs = []
    nfix = {}
    for _, row in df.iterrows():
        rid = int(row["Unnamed: 0"])
        q = parse_question(row.question, row.inference_type)
        pr = parse_prior(row.ground_truth_prior)
        dep = parse_depth(row.depth_info) if str(row.video_type)[1] == "3" else []
        keys = sorted(set(d["key"] for d in dep))
        r = dict(id=rid, video_id=str(row.video_id).strip(), video_source=row.video_source, video_type=row.video_type,
                 inference_type=row.inference_type, fps=float(row.fps), cat=row.inference_type[0] + row.video_type[1],
                 question=row.question, prior_text=row.ground_truth_prior, q=q, priors=pr, depth=dep, depth_keys=keys)
        L = llm.get(rid, {})
        r["llm"] = L
        fixes = list(q.get("fixes", []))
        tp, where = det_phrase(q["target"])
        src = "rule"
        if (tp is None or len(tp.split()) > 3) and L.get("target"):
            tp, src = str(L["target"]).lower().strip(), "llm"
            f = fix_object_phrase(tp)
            if f["changed"]:
                tp = f["phrase"]; fixes += ["llm_target:" + c for c in f["changed"]]
                if f["rel"] and not q.get("target_rel"):
                    q["target_rel"] = [f["rel"], f["ref"]]
        if not where and L.get("target_where") and str(L["target_where"]).lower() not in ("none", ""):
            where = [w for w in SPATIAL if re.search(r"\b" + w + r"\b", str(L["target_where"]).lower())][:2]
        if q.get("target_rel"):
            drop = {"near": ("closest", "nearest", "near", "front", "back"), "right_of": ("right",), "left_of": ("left",),
                    "below": ("lower", "bottom"), "above": ("upper", "top")}.get(q["target_rel"][0], ())
            where = [w for w in where if w not in drop]
        r["target"] = dict(phrase=tp, where=where, src=src, raw=q["target"], rel=q.get("target_rel"))
        t2 = None
        if q["target2"] == "#second":
            t2 = dict(phrase=tp, where=["#second"], src="rule", raw=q["target"], rel=None)
        elif q["target2"]:
            p2, w2 = det_phrase(q["target2"]); s2 = "rule"
            if (p2 is None or len(p2.split()) > 3) and L.get("target2"):
                p2, s2 = str(L["target2"]).lower().strip(), "llm"
            t2 = dict(phrase=p2, where=w2, src=s2, raw=q["target2"], rel=q.get("target2_rel"))
        elif q["kind"] == "dist2" and L.get("target2"):
            t2 = dict(phrase=str(L["target2"]).lower().strip(), where=[], src="llm", raw=None, rel=None)
        r["target2"] = t2
        p0 = pr[0] if pr else {}
        pobj_raw = p0.get("obj")
        prel = None
        if pobj_raw:
            f = fix_object_phrase(pobj_raw)
            if f["changed"]:
                pobj_raw = f["phrase"]; fixes += ["prior:" + c for c in f["changed"]]
                prel = [f["rel"], f["ref"]] if f["rel"] else None
        pp, pw = det_phrase(pobj_raw)
        psrc = "rule"
        if p0.get("gravity") and not pp:
            pp = None
        elif (pp is None or len(pp.split()) > 3) and L.get("prior_object") and str(L["prior_object"]).lower() != "gravity":
            pp, psrc = str(L["prior_object"]).lower().strip(), "llm"
        r["prior_obj"] = dict(phrase=pp, where=pw, src=psrc, raw=p0.get("obj"), rel=prel)

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
        r["target2_dkeys"], _ = dkey(q["target2"] if t2 and q["target2"] != "#second" else None, t2["phrase"] if t2 else None, L.get("target2_depth_key")) if t2 else ([], "none")
        r["prior_dkeys"], r["prior_dsrc"] = dkey(p0.get("obj"), pp, L.get("prior_depth_key")) if pp else ([], "none")
        r["target_whole"] = bool(L.get("target_whole", True)) if L else True
        r["prior_whole"] = bool(L.get("prior_whole", True)) if L else True
        o = v1[rid]
        def sig(e):
            return None if not e else (e.get("phrase"), tuple(e.get("where") or []), tuple(e.get("rel") or []))
        changed = (sig(r["target"]) != sig(o["target"])) or (sig(r["target2"]) != sig(o.get("target2"))) or \
                  (sig(r["prior_obj"]) != sig(o["prior_obj"])) or (q["kind"] != o["q"]["kind"]) or (q.get("dim") != o["q"].get("dim")) or bool(q.get("part"))
        r["v2_changed"] = bool(changed)
        r["v2_fixes"] = fixes
        for f in fixes:
            nfix[f.split(":")[-1]] = nfix.get(f.split(":")[-1], 0) + 1
        recs.append(r)
    with open(f"{OUT}/parse_{a.split}_v2.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ch = [r for r in recs if r["v2_changed"]]
    print(json.dumps(dict(split=a.split, rows=len(recs), changed_rows=len(ch), changed_videos=len(set(r["video_id"] for r in ch)), fixes=nfix)))
    for r in ch:
        o = v1[r["id"]]
        print(f"{r['id']:5d} {r['video_id']:26s} | {o['target']['phrase']}|{','.join(o['target']['where'])} -> {r['target']['phrase']}|{','.join(r['target']['where'])} rel={r['target']['rel']} "
              f"part={r['q'].get('part', False)} dim={o['q'].get('dim')}->{r['q'].get('dim')} | prior {o['prior_obj']['phrase']} -> {r['prior_obj']['phrase']} {r['prior_obj']['rel']} | "
              f"t2 {(o.get('target2') or {}).get('phrase')} -> {(r['target2'] or {}).get('phrase')} | {r['v2_fixes']}")


if __name__ == "__main__":
    main()
