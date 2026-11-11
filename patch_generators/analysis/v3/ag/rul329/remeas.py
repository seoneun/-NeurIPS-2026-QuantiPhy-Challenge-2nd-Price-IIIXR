"""rul329: re-run the pipeline's measure_video (geo/measure_v3.py, imported read-only) for rows whose parse target is a
part word ("longest side", "side", "edge", "height", "moving speed") of a named object X.
Retarget = what QP_SIDE_FIX already does in load_split_tracks (it runs here, env as in the rj5sQ chain).
Why "longest" slipped past QP_SIDE_FIX: the retarget itself works, but the pipeline track file for the video was built
from the original parse, so no query for X exists (internet_0024 has no v2/v2sc tracks at all; the v1 file has no objs)
-> meas_why target_no_track. Track for X, in order: (1) a pipeline query whose phrase is X; (2) the VLM-anchored v3
track file (tracks_{split}_v3nt) whose query 'expr' names X ("longest side of the wallet" -> obj 3)."""
import os, sys, json, copy, glob, re
ENV = dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", QP_TSHIFT="1", QP_FPERSON="1", QP_FPERSON_MINN="2")
for k, v in ENV.items():
    os.environ.setdefault(k, v)
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo")
import numpy as np, pandas as pd
import measure_v3 as MV

PARTW = r"(?:(?:longer|shorter|long|short|longest|shortest)\s+)?(?:side|edge|height|width|length)|moving speed|speed"
PART_TP = re.compile(r"^(?:the\s+)?(?:" + PARTW + r")$")
ART = r"(?:the\s+|a\s+|an\s+)?"
OF_X = re.compile(r"\b(?:" + PARTW + r")\s+of\s+" + ART + r"(?:(?:" + PARTW + r")\s+of\s+" + ART + r")*"
                  r"((?:[a-z]+'s\s+)?(?:\([a-z ]+\)\s+)?[a-z][a-z \-]*?)\s*(?:\bin\b|\bat\b|,|\?|$)")


def setup(split, vids):
    MV.TRACK_VER = "v2sc"
    MV.FIX = set("abcdef")
    MV.MOGE_OPTS = {"fbrej"}
    for p in glob.glob(f"{R}/geo/out/moge_{split}/*.npz"):
        v = os.path.basename(p)[:-4]
        if v not in vids:
            continue
        z = np.load(p)
        MV.MOGE[v] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
    fn_ = f"{R}/geo/out/scene_ruler_{split}.json"
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
    fp_ = f"{R}/geo/out/person_focal_{split}.json"
    if os.path.exists(fp_):
        MV.FPERSON.update(json.load(open(fp_)))
    MV.M1.SCALE_DEPTH = True
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))["T_owl"]
    return json.load(open(f"{R}/geo/out/focal_medians_test.json"))["f_over_W"]


def measure(rs, vj, fmed, cfg="e"):
    allf = float(np.median(list(fmed.values())))
    fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * vj["W"] for r in rs}
    o, V = MV.measure_video(copy.deepcopy(rs), vj, fdd, cfg)
    out = {}
    for x, r in zip(o, rs):
        u = r["q"]["unit"]
        out[r["id"]] = dict(value=(x["val_si"] / MV.UNIT_SI[u]) if x.get("ok") else float("nan"), mode=x.get("mode"), f=x.get("f"), why=x.get("why"))
    return out, V


def candidates(split):
    """rows whose original parse target is a part word; X = the object it belongs to (from the question)"""
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    C = {}
    for r in P:
        tp = ((r.get("target") or {}).get("phrase") or "").strip().lower()
        if not PART_TP.match(tp):
            continue
        ql = r["question"].lower().replace("’", "'")
        m = OF_X.search(ql)
        x = m.group(1).strip() if m else None
        if x:
            x = re.sub(r"^[a-z]+'s\s+", "", x)               # "user's fingertip" -> "fingertip"
            x = re.sub(r"\([a-z ]+\)\s*", "", x).strip()      # "(middle) fingertip" -> "fingertip"
        C[r["id"]] = dict(vid=r["video_id"], tp=tp, x=x, scene=bool(x and re.search(MV.SCENE, x)), q=r["question"])
    return C


def v3_alias(vj3, x):
    """a v3 (VLM-anchored) query whose expr names X"""
    for q in vj3.get("queries", []):
        e = (q.get("expr") or "").lower()
        if q.get("obj") is not None and x and re.search(r"\b" + re.escape(x) + r"\b", e):
            return q
    return None


def run(split, verbose=True):
    C = candidates(split)
    if verbose:
        for i, c in sorted(C.items()):
            print(i, c)
    vids = sorted({c["vid"] for c in C.values()})
    fmed = setup(split, set(vids))
    byv, vjs = MV.load_split_tracks(split, "e")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    res = {}
    for v in vids:
        rs = byv[v]
        ids = [i for i, c in C.items() if c["vid"] == v]
        rmap = {r["id"]: r for r in rs}
        for i in ids:
            r = rmap[i]
            print(f"\n== {i} {v} target after side_fix: {r['target'].get('phrase')!r} dim={r['q'].get('dim')} "
                  f"kind={r['q'].get('kind')} side_fix={r.get('side_fix')}")
        vj = vjs.get(v)
        have = vj is not None and vj.get("version") == "v2"
        have1 = vj is not None and not have and any(q.get("frames") for q in vj.get("queries", []))
        print(v, "pipeline track file:", "v2-format" if have else ("v1-format" if have1 else "none"))
        p3 = f"{R}/geo/out/tracks_{split}_v3nt/{v}.json"
        vj3 = json.load(open(p3)) if os.path.exists(p3) else None
        pipe = vj if (have or have1) else vj3
        base = copy.deepcopy(pipe)
        if base is None:
            print("  no usable track file"); continue
        rs2 = copy.deepcopy(rs)
        todo = []
        for r in rs2:
            if r["id"] not in ids:
                continue
            x = C[r["id"]]["x"]
            if not x or C[r["id"]]["scene"]:
                print("  ", r["id"], "skip: no X or scene X:", x); continue
            if r["q"].get("kind") not in ("size",):
                print("  ", r["id"], "skip: kind", r["q"].get("kind"), "(rule covers size rows only)"); continue
            V0 = MV.Video(base, "e")
            ent = dict(r["target"], phrase=x, raw=x, where=[])
            if V0.get(ent) is None:
                qa = v3_alias(vj3, x) if vj3 is not None else None
                if qa is None:
                    print("  ", r["id"], "no track for", repr(x)); continue
                if (have or have1) and base["fidx"] != vj3["fidx"]:
                    print("  ", r["id"], "frame mismatch v3 vs pipeline"); continue
                k = V0.key(ent)
                qn = dict(qa, key=k, phrase=x, where=[])
                if have:
                    oid = str(10000 + int(qa["obj"])); base["objs"][oid] = vj3["objs"][str(qa["obj"])]; qn["obj"] = int(oid)
                elif have1:                                   # v1 file: the track lives in the query's frames
                    qn["frames"] = vj3["objs"][str(qa["obj"])]; qn.pop("obj", None)
                base["queries"].append(qn)
                if "queries_d" in base:
                    base["queries_d"].append(dict(qn))
                print("  ", r["id"], f"alias {k!r} -> v3 obj {qa['obj']} (expr {qa.get('expr')!r})")
            else:
                print("  ", r["id"], "pipeline track exists for", repr(x))
            r["target"] = ent
            r["q"] = dict(r["q"], target=x)
            todo.append(r["id"])
        if not todo:
            continue
        o, V = measure(rs2, base, fmed)
        o0, _ = measure(rs, pipe, fmed)                 # the unchanged rows must reproduce the pipeline raw
        for i in sorted(o0):
            if i not in todo:
                print(f"   sib {i}: {o0[i]['value']:.4g} {o0[i]['mode']} | raw {raw.geo_value.get(i)} {raw.meas_mode.get(i)} | grafted {o[i]['value']:.4g}")
        for i in todo:
            x = o[i]
            print(f"  {i}: new {x['value']:.4g} {x['mode']} {x['why']} | raw {raw.geo_value.get(i)} {raw.meas_mode.get(i)} {raw.meas_why.get(i)}")
            res[str(i)] = dict(new=x["value"], mode=x["mode"], f=x["f"], why=x["why"], x=C[i]["x"], vid=v)
    json.dump(dict(res=res), open(f"{R}/cache/tmp/ag_partword_meas_{split}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    run(sys.argv[1])
