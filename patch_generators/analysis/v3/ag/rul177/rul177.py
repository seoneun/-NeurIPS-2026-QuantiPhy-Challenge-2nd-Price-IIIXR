"""rul177: width capped by the object's own answered length (audit81:D).
Gates: (not simulation or cat S3/D3); q kind=size, dim=width; target not winged (plane/airplane/jet/aircraft/bird/wing/
butterfly/drone/glider); same video has a size/length row with the same normalised target and final answer A_L>0;
cs > 1.2*A_L and 0 < g < A_L; answer_source (mo_guard: stripped) in dis_size_gm, d3size_caw, caw* (never patch:*,
not agree_*); every copy of the recording must pass. Output: sqrt(g*A_L) (CaW replaced by its physical upper bound, the sibling length answer, before the
gm), then log-mean over the fired plain/s/x copies of one recording (split_variant group + question), as lab_pool does.
No ids; labels only evaluate. usage: python rul177.py [SUB]"""
import sys, json, math, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
from variant_keys import split_variant
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz27"
TAG = "rul177"
WING = {"plane", "airplane", "aeroplane", "jet", "aircraft", "bird", "wing", "wings", "butterfly", "drone", "glider"}
OKSRC = {"dis_size_gm", "d3size_caw"}  # agree_gm/agree_wgm dropped: the 3 labelled agree fires (captured_0021 table) each lose 0.1 MRA
def norm(t):
    t = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower())
    return " ".join(w for w in t.split() if w not in ("the", "a", "an", "of"))
L = labelled(SUB)
for split in ("test", "val"):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0, usecols=["Unnamed: 0", "video_id", "question"])
    for c in ("video_id", "question"):
        if c not in det:
            det[c] = raw[c].reindex(det.index)
    Q = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(ln); Q[j["id"]] = j.get("q") or {}
    det["qkind"] = [Q.get(i, {}).get("kind") for i in det.index]
    det["qdim"] = [Q.get(i, {}).get("dim") for i in det.index]
    det["tn"] = [norm(Q.get(i, {}).get("target")) for i in det.index]
    det["src_"] = det.answer_source.astype(str).str.replace("mo_guard:", "", regex=False)
    lens = det[(det.qkind == "size") & (det.qdim == "length") & (det.answer > 0)]
    cand = {}
    for i, r in det[(det.qkind == "size") & (det.qdim == "width")].iterrows():
        if str(r.video_id).startswith("simulation") and r["cat"] not in ("S3", "D3"):
            continue
        if not r.tn or set(r.tn.split()) & WING:
            continue
        s = r.src_
        if s.startswith("patch:") or not (s in OKSRC or s.startswith("caw")):
            continue
        m = lens[(lens.video_id == r.video_id) & (lens.tn == r.tn)]
        if len(m) == 0:
            continue
        AL = float(m.answer.iloc[0]); g, cs = float(r.g), float(r.cs)
        if not (np.isfinite(cs) and np.isfinite(g) and cs > 1.2 * AL and 0 < g < AL):
            continue
        cand[i] = (math.sqrt(g * AL), AL, int(m.index[0]))
    det["key"] = det.video_id.map(lambda z: split_variant(str(z))[0]) + "||" + det.question.astype(str).str.strip().str.lower()
    out = {}
    allk = det.key.value_counts()
    for k, grp in det.loc[list(cand)].groupby("key") if cand else []:
        if len(grp) < allk[k]:   # every plain/s/x copy of the recording must show the out-of-band value (copy-consistent)
            print("  skip partial group", k, list(grp.index), "of", allk[k]); continue
        pv = float(np.exp(np.mean([math.log(cand[i][0]) for i in grp.index])))
        for i in grp.index:
            r = det.loc[i]; new, AL, li = cand[i]
            out[str(i)] = {"value": pv, "why": f"rul177: CaW width {r.cs:.4g} > 1.2x own length answer {AL:.4g} (row {li}); "
                                               f"sqrt(g {r.g:.4g} * L)={new:.4g}; pooled over {len(grp)} copies -> {pv:.4g}"}
            print(split, i, r.video_id, r["cat"], r.question[:60], "|", round(r.answer, 4), r.src_, "g", round(r.g, 3), "cs", r.cs, "L", round(AL, 3), "->", round(new, 3), "pool", round(pv, 3))
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    Ls = L[L.split == split]
    for i in out:
        if int(i) in Ls.index:
            y = Ls.y[int(i)]
            print("  LAB", i, y, A.row_mra(np.array([det.answer[int(i)]]), np.array([y])), "->", A.row_mra(np.array([out[i]["value"]]), np.array([y])))
    print(split, "rows", len(out), "by cat", det.loc[[int(i) for i in out], "cat"].value_counts().to_dict() if out else {}, "labelled hit", sum(int(i) in Ls.index for i in out))
    # near-miss diagnostics for named controls
    if split == "test":
        for i in (136, 232, 997, 1003, 1471, 1476, 1481, 2923, 2931, 2939):
            if i in det.index:
                r = det.loc[i]; print("  ctl", i, r.video_id, r.question[:55], r.src_, "g", r.g, "cs", r.cs, "fired", str(i) in out)
