"""rul212 (copy of rul191, answer gate: ans>2.05 and source in dimgm/agree gm set): class-bound cap for passenger-car WIDTH in 3D size rows.
Gates (code only): cat S3/D3; parse q.kind size; question has width/wide and not wheel|road|lane|truck|bus|forklift;
target phrase has car|sedan|taxi|suv|coupe; answer > 2.5 m and geo g > 2.5 m (geo inflated); CaW cs in [1.5, 2.3] m.
Output: cs x 1.05 (d3size_caw scale). _segmented / variant twins (same question, video id sharing the base) copy.
Writes cache/tmp/ag_c89carwcap_patch_{test,val}.json"""
import sys, os, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C98_SUB", "sub_xt_Nzz34")
TAG = "rul212"
SRCOK = {"d3size_dimgm", "d3size_unstable_gm", "agree_gm", "agree_wgm"}
QW = re.compile(r"\bwidth\b|\bwide\b", re.I)
QX = re.compile(r"wheel|road|lane|truck|bus|forklift", re.I)
TC = re.compile(r"\b(car|sedan|taxi|suv|coupe)\b", re.I)
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "ft": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
       "feet": 0.3048, "meter": 1.0, "meters": 1.0, "centimeters": 0.01}


def base(v):
    return re.sub(r"_segmented$", "", str(v))


def run(split):
    D = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv" if split == "test" else f"{R}/submissions/{SUB}_valdetail.csv").set_index("Unnamed: 0")
    print(split, "cols", list(D.columns))
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = j
    out, log = {}, []
    for i, r in D.iterrows():
        if r["cat"] not in ("S3", "D3"): continue
        p = P.get(int(i), {}); q = p.get("q") or {}
        if q.get("kind") != "size": continue
        qs = str(r.get("question", p.get("question", "")))
        if not QW.search(qs) or QX.search(qs): continue
        tp = p.get("target") or q.get("target") or ""
        tp = str(tp.get("phrase") or "") if isinstance(tp, dict) else str(tp)
        if not TC.search(tp): continue
        u = str(q.get("unit") or "m").lower(); f = USI.get(u)
        if f is None: log.append((i, "unit?", u)); continue
        a, g, c = r.answer * f, r.g * f if pd.notna(r.g) else np.nan, r.cs * f if pd.notna(r.cs) else np.nan
        log.append((i, r.video_id, tp, u, round(a, 3), round(g, 3), round(c, 3), r.answer_source))
        if not (a > 2.05 and str(r.answer_source) in SRCOK and g > 2.5 and 1.5 <= c <= 2.3): continue
        v = c * 1.05
        out[str(int(i))] = dict(value=float(v / f), why=f"{TAG}: dimgm car width with inflated geo; ans {a:.3g} m, geo {g:.3g} m > 2.5 (inflated); CaW {c:.3g} in [1.5,2.3]; cs x1.05; was {r.answer:.4g} ({r.answer_source})")
    for k in list(out):
        vid, q = D.video_id[int(k)], str(D.question[int(k)]).strip()
        tw = D[(D.video_id.map(base) == base(vid)) & (D.video_id != vid) & (D.question.astype(str).str.strip() == q)]
        for j in tw.index:
            if str(int(j)) not in out:
                out[str(int(j))] = dict(value=out[k]["value"], why=out[k]["why"] + f" [twin of {k}]")
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    return out, log, D


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
    from lab_rows import labelled
    import analyze_caw as A
    allp = {}
    for s in ("test", "val"):
        o, log, D = run(s)
        print(f"== {s}: {len(o)} changed; candidates passing text gates: {len(log)}")
        for x in log: print("  cand", x)
        for k, v in o.items():
            print("  HIT", s, k, D.video_id[int(k)], D["cat"][int(k)], "|", str(D.question[int(k)])[:110], "|", D.answer[int(k)], "->", round(v["value"], 4))
        allp[s] = o
    L = labelled(SUB)
    print("labelled cols", list(L.columns)[:20], len(L))
    idc = "row_id" if "row_id" in L.columns else None
    ch = 0
    for i, r in L.iterrows():
        rid = str(int(r[idc])) if idc else str(int(i))
        sp = r.split
        if rid in allp.get(sp, {}):
            ch += 1; nv = allp[sp][rid]["value"]
            print("LAB", sp, rid, r.video_id, "y", r.y, "old m", r.m, "new m", A.row_mra(np.array([nv]), np.array([r.y]))[0])
    print("labelled changed", ch); assert ch == 0, "labelled row changed"
    for s, rid in (("val", "1455"), ("test", "1455")):
        print("check 1455", s, rid in allp[s])
