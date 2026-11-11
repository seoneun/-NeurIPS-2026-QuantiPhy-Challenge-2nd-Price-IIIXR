"""rul194: dist2 rows whose target/target2 parse is truncated (ends in a dangling function word, the rel_near
truncation) AND an endpoint's query key binds to different SAM2 objects under the two detector configs
(queries vs queries_d) with mean per-frame box IoU < 0.3, AND the row is agree_geo_kind with g, cs finite,
positive, within 2x -> hedge to gm sqrt(g*cs). No ids; labels only evaluate.
usage: python rul194.py [SUB]"""
import sys, json, math, re, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz27"
TAG = "rul194"
RAWN = "geo_v4ntrj5sQ2srngsr_d_vw"
TRUNC = re.compile(r"\b(of the|of|the|a|an)$")
WHY = "rul194: truncated part phrase + detector disagreement on dist2 endpoint -> gm"
def key(ent):
    if not ent or not ent.get("phrase"): return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    I = ix * iy; U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I
    return I / U if U > 0 else 0.0
def mean_iou(vj, oa, ob):
    fa, fb = vj["objs"].get(str(oa)) or [], vj["objs"].get(str(ob)) or []
    v = [iou(x["st"]["box"], y["st"]["box"]) for x, y in zip(fa, fb)
         if x and y and x.get("st") and y.get("st")]
    return float(np.mean(v)) if v else 0.0
def disagree(vj, k):
    if k is None: return None
    qa = {q["key"]: q.get("obj") for q in vj.get("queries", [])}
    qb = {q["key"]: q.get("obj") for q in vj.get("queries_d", [])}
    if k not in qa or k not in qb or qa[k] is None or qb[k] is None or qa[k] == qb[k]: return None
    m = mean_iou(vj, qa[k], qb[k])
    return m if m < 0.3 else None
L = labelled(SUB)
res = {}
for split in ("test", "val"):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWN}_raw.csv", index_col=0)
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(ln); P[int(j["id"])] = j
    out = {}; na = nb = 0
    for i in det.index:
        p = P.get(int(i)) or {}; q = p.get("q") or {}
        if q.get("kind") != "dist2": continue
        tr = [r for r in ("target", "target2") if TRUNC.search(str(q.get(r) or "").strip())]
        if not tr: continue
        na += 1
        vid = raw.video_id[i]; mt = raw.meas_tracks[i]
        fn = f"{R}/geo/out/tracks_{split}_{mt}/{vid}.json"
        if not isinstance(mt, str) or not os.path.exists(fn):
            print(split, i, "no tracks", mt); continue
        vj = json.load(open(fn))
        # endpoint keys of the row (both roles; the truncated parse corrupts the pairing of the endpoints)
        ks = [key(p.get("target")), key(p.get("target2"))]
        dis = {k: disagree(vj, k) for k in ks if k}
        hit = {k: v for k, v in dis.items() if v is not None}
        src = det.answer_source[i]; g, cs = float(det.g[i]), float(det.cs[i])
        print(split, i, vid, det.cat[i], "trunc", tr, "keys", ks, "dis", dis, src, round(g, 3) if np.isfinite(g) else g, round(cs, 3) if np.isfinite(cs) else cs, round(float(det.answer[i]), 3))
        if not hit: continue
        nb += 1
        if src != "agree_geo_kind" or not (np.isfinite(g) and np.isfinite(cs) and g > 0 and cs > 0 and max(g/cs, cs/g) <= 2): continue
        out[str(int(i))] = {"value": float(math.sqrt(g * cs)), "why": WHY, "_old": float(det.answer[i]), "_cat": det.cat[i], "_vid": vid}
    print(split, "trunc", na, "trunc+dis", nb, "fire", len(out))
    res[split] = out
    json.dump({k: {"value": v["value"], "why": v["why"]} for k, v in out.items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
for k, v in res["test"].items(): print("TEST", k, v["_vid"], v["_cat"], round(v["_old"], 3), "->", round(v["value"], 3))
# labelled eval
ch = 0
for _, r in L.iterrows():
    pv = res.get(r.split, {}).get(str(int(r.name)) if not hasattr(r, "row_id") else str(int(r.row_id)))
    if pv: ch += 1; print("LAB", r.split, r.name, r.y, r.m, A.row_mra(np.array([pv["value"]]), np.array([r.y])))
print("labelled changed", ch, "val fired", len(res["val"]))
assert len(res["val"]) == 0 and ch == 0
