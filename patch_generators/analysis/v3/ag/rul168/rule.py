"""rul168: lab stated-fact witness for the ping pong ball diameter.
A captured_* video's prior_text states 'diameter of the ping pong ball = 40mm'. A size question (dim diameter/size/None)
about the ping pong ball (or its lab synonym 'white ball') in a captured_* video whose own geo AND current answer lie
within 1.25x of the fact is answered with the fact (when >5% off and the answer is not a prior copy)."""
import json, re, sys, math
import pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp")
SUB = "sub_xt_Nzz27"
FACT = re.compile(r"diameter of the ping ?pong ball\s*=\s*([\d.]+)\s*(mm|cm|m)\b", re.I)
TOM = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
NAMES = {"ping pong ball", "pingpong ball", "white ball"}

parse = {}
for sp in ("test", "val"):
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); parse[(sp, r["id"])] = r
facts = []
for (sp, i), r in parse.items():
    if not str(r["video_id"]).startswith("captured_"): continue
    for m in FACT.finditer(r.get("prior_text") or ""):
        facts.append((sp, r["video_id"], float(m.group(1)) * TOM[m.group(2).lower()]))
vals = sorted({f[2] for f in facts})
print("facts:", len(facts), "videos", len({f[1] for f in facts}), "values(m)", vals)
assert vals and max(vals) / min(vals) <= 1.02, "facts disagree"
v_si = float(np.median([f[2] for f in facts]))

def phrase(r):
    t = r.get("target")
    p = t.get("phrase") if isinstance(t, dict) else None
    p = p or (r.get("q") or {}).get("target") or ""
    return re.sub(r"\s+", " ", p.lower().strip())

out = {}
for sp, det in (("test", f"{R}/submissions/{SUB}_detail.csv"), ("val", f"{R}/submissions/{SUB}_valdetail.csv")):
    D = pd.read_csv(det, index_col=0)
    patch = {}; near = []
    for i, row in D.iterrows():
        r = parse[(sp, int(i))]
        if not str(r["video_id"]).startswith("captured_"): continue
        q = r.get("q") or {}
        if q.get("kind") != "size" or q.get("dim") not in ("diameter", "size", None): continue
        if phrase(r) not in NAMES: continue
        unit = (q.get("unit") or "").lower()
        if unit not in TOM: print("skip unit", sp, i, unit); continue
        v = v_si / TOM[unit]
        g, a, src = row["g"], float(row["answer"]), str(row["answer_source"])
        info = (sp, i, r["video_id"], phrase(r), unit, round(v, 4), g, round(a, 4), src)
        if not (np.isfinite(g) and v / 1.25 <= g <= 1.25 * v): print("gate a geo", info); continue
        if not (v / 1.25 <= a <= 1.25 * v): print("gate a ans", info); continue
        if abs(math.log(a / v)) <= math.log(1.05): near.append(info); continue
        if src.startswith("scene_prior") or src.startswith("xrow_prior") or "scene_prior" in src or "xrow_prior" in src:
            print("gate c", info); continue
        print("HIT", info)
        patch[str(i)] = {"value": round(v, 6), "why": f"rul168: lab fact 'diameter of the ping pong ball = {v_si*1000:g}mm' "
                         f"({len(facts)} stated facts agree); target '{phrase(r)}' own geo {g:.3g} within 1.25x; was {a:.4g} ({src})"}
    for n in near: print("near(unchanged)", n)
    json.dump(patch, open(f"{R}/cache/tmp/ag_c83dkeyppball_patch_{sp}.json", "w"), indent=1)
    out[sp] = patch
    print(sp, "patched", len(patch))

from lab_rows import labelled
import analyze_caw as A
L = labelled(SUB)
for sp in ("test", "val"):
    for k, p in out[sp].items():
        hit = L[(L.split == sp) & (L.index.astype(str) == k)] if "split" in L else L.iloc[0:0]
        if len(hit):
            y = hit.y.values
            print("LAB", sp, k, "y", y, "before", hit.m.values, "after", A.row_mra(np.array([p["value"]]), y))
print("labelled rows hit:", sum(len(L[(L.split == sp) & L.index.astype(str).isin(list(out[sp]))]) for sp in out))
# also report all labelled ping-pong rows for context
