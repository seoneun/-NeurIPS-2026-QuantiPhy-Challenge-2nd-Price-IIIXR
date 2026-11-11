"""rul011: occlusion-order lower bound for S3/D3 size rows measured at the prior's depth (tdepth=prior).

If the prior object is seen IN FRONT of a static target (it occludes the target's box edge, or its box sits inside the
target box while visible), the target lies at or beyond the prior's depth key.  geo for tdepth=prior rows assumes the
target is AT the prior's depth, so size = px * Z_prior / f is a lower bound on the true size px * Z_target / f.
Final = max(answer, geo).  (Equivalent to bounding the CaW candidate at geo and re-running the fusion:
gm(geo, max(cs, geo)) / max(cs, geo) are both >= geo.)

Usage:  python occdepth.py [--novlm] [--vert] [--diag]
  --novlm : skip vlm_* geo modes (optional extra safety)
  --vert  : also count top/bottom edge retreat (default: left/right only, as scanned)
  --diag  : print relaxed-gate diagnostics on labelled rows
Writes cache/tmp/ag_c20occdepth_patch_{test,val}.json (default variant only, i.e. no flags or as given).
"""
import json, os, re, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
SUB = "sub_xt_Nzn"
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
TAG = "rul011"
NOVLM = "--novlm" in sys.argv
VERT = "--vert" in sys.argv
DIAG = "--diag" in sys.argv
NOWRITE = "--nowrite" in sys.argv
DIMS = {"length", "height", "size", "diameter"}
SUPPORT = {"table", "desk", "floor", "ground", "lane", "road", "street", "sidewalk", "field", "court", "grass",
           "track", "ramp", "slope"}
RETREAT_PX, N_RETREAT, N_INSIDE, STATIC_FRAC, PMOVE = 3.0, 3, 5, 0.03, 0.05

pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 48); pd.set_option("display.max_rows", 500)


def load_tr(split, v):
    for sub in ["_v2sc", "_v2", "_v3nt", ""]:
        p = f"{R}/geo/out/tracks_{split}{sub}/{v}.json"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def head_noun(phrase):
    s = (phrase or "").lower()
    s = re.split(r"\b(?:of|on|in|at|with|under|near|next to|behind|beside)\b", s)[0]
    toks = re.findall(r"[a-z]+", s)
    return toks[-1] if toks else ""


def bx(lst, k):
    x = lst[k] if k < len(lst) else None
    return x["st"]["box"] if x and x.get("st") else None


def evidence(j, T, P):
    lt = j["objs"][str(T)]; lp = j["objs"][str(P)]; N = len(j["fidx"])
    tb = [bx(lt, k) for k in range(N)]; pb = [bx(lp, k) for k in range(N)]
    ok = [k for k in range(N) if tb[k]]
    if len(ok) < 5:
        return None
    A_ = np.array([tb[k] for k in ok], float)
    C = np.c_[(A_[:, 0] + A_[:, 2]) / 2, (A_[:, 1] + A_[:, 3]) / 2]
    static = bool(np.ptp(C[:, 0]) < STATIC_FRAC * j["W"] and np.ptp(C[:, 1]) < STATIC_FRAC * j["H"])
    med = np.median(A_, axis=0)
    ov = inside = retreat = 0
    pin = []                                   # prior centres in the frames where its box is inside the target box
    for k in ok:
        a = tb[k]; b = pb[k]
        if not b:
            continue
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        if ix * iy <= 0:
            continue
        ov += 1
        if b[0] >= a[0] and b[2] <= a[2] and b[1] >= a[1] and b[3] <= a[3]:
            inside += 1; pin.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        r = False
        # prior spans the target's median right / left edge and that edge sits >= 3 px inside its median
        if b[0] < med[2] < b[2] and a[2] < med[2] - RETREAT_PX: r = True
        if b[0] < med[0] < b[2] and a[0] > med[0] + RETREAT_PX: r = True
        if VERT:
            if b[1] < med[3] < b[3] and a[3] < med[3] - RETREAT_PX: r = True
            if b[1] < med[1] < b[3] and a[1] > med[1] + RETREAT_PX: r = True
        retreat += int(r)
    # extra safety for (b): the prior must MOVE across the target while inside its box (a passing occluder), so a prior
    # track stuck on the target's own texture cannot fire
    pmove = float(np.ptp(np.array(pin), axis=0).max() / j["W"]) if len(pin) >= 2 else 0.0
    return dict(static=static, ov=ov, inside=inside, retreat=retreat, pmove=round(pmove, 3), nok=len(ok))


def scan(split):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        jj = json.loads(l); P[int(jj.get("id", jj.get("row_id")))] = jj
    G = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv")
    if "Unnamed: 0" in G.columns:
        G = G.set_index("Unnamed: 0")
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    rows = []
    for i, r in G.iterrows():
        mm = str(r.meas_mode)
        if "tdepth=prior" not in mm:
            continue
        pq = P.get(int(i), {}); q = pq.get("q") or {}
        src = str(r.video_source).lower()
        rec = dict(split=split, id=int(i), vid=r.video_id, cat=pq.get("cat"), kind=q.get("kind"), dim=q.get("dim"),
                   q=str(r.question)[:58], mode=mm, src=src, fails=[])
        lab = src == "lab" or str(r.video_id).startswith("captured")
        if lab: rec["fails"].append("lab")
        if pq.get("cat") not in ("S3", "D3"): rec["fails"].append("cat")
        if q.get("kind") != "size": rec["fails"].append("kind")
        if q.get("dim") not in DIMS: rec["fails"].append("dim")
        if not pq.get("prior_dkeys"): rec["fails"].append("nopriorkey")
        if NOVLM and mm.startswith("vlm"): rec["fails"].append("vlm")
        tgt = q.get("target") or ""
        hn = head_noun((pq.get("target") or {}).get("phrase") or tgt)
        rec["head"] = hn
        if hn in SUPPORT: rec["fails"].append("support")
        j = load_tr(split, r.video_id)
        if j is None:
            continue
        Tq = [x for x in j["queries"] if "target" in (x.get("roles") or []) and x.get("obj") is not None and tgt and x["phrase"] in tgt]
        Pq = [x for x in j["queries"] if "prior" in (x.get("roles") or []) and x.get("obj") is not None]
        if not Tq or not Pq:
            continue
        T = Tq[0]["obj"]; Pp = Pq[0]["obj"]
        if T == Pp:
            rec["fails"].append("sameobj")
            continue
        ev = evidence(j, T, Pp)
        if ev is None:
            continue
        rec.update(ev)
        if not ev["static"]: rec["fails"].append("static")
        if not (ev["retreat"] >= N_RETREAT or (ev["inside"] >= N_INSIDE and ev["pmove"] >= PMOVE)): rec["fails"].append("noocc")
        d = D.loc[i] if i in D.index else None
        g = float(d.g) if d is not None and pd.notna(d.g) else float(r.geo_value) if pd.notna(r.geo_value) else np.nan
        rec.update(geo=g, ans=float(d.answer) if d is not None else np.nan,
                   asrc=d.answer_source if d is not None else "", cs=float(d.cs) if d is not None and pd.notna(d.cs) else np.nan)
        rec["new"] = max(rec["ans"], g) if np.isfinite(g) and g > 0 else rec["ans"]
        rec["fire"] = not rec["fails"]
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import analyze_caw as A
    from lab_rows import labelled
    L = labelled(SUB)
    out = {}
    allx = []
    for split in ("test", "val"):
        X = scan(split)
        if "id" not in X:   # 10-17 clean6: a scan with 0 rows returns a frame without columns (was AttributeError on X.id)
            X = pd.DataFrame({c: pd.Series(dtype=float) for c in ("split", "id", "vid", "cat", "dim", "q", "mode", "static", "ov", "inside",
                              "pmove", "retreat", "geo", "cs", "ans", "asrc", "new")}).assign(fire=pd.Series(dtype=bool), fails=pd.Series(dtype=object))
        # attach labels
        if split == "val":
            ys = L[L.split == "val"].y
        else:
            ys = L[L.split == "test"].y
        X["y"] = X.id.map(ys.to_dict())
        allx.append(X)
        F = X[X.fire & (X.new > X.ans * 1.0000001)]
        patch = {str(int(r.id)): {"value": float(r.new),
                                  "why": f"occdepth: prior in front (retreat {r.retreat}, inside {r.inside}) -> size >= geo {r.geo:.3g} at prior depth; was {r.ans:.4g} ({r.asrc})"}
                 for r in F.itertuples()}
        out[split] = patch
        if not NOWRITE:
            json.dump(patch, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        print(f"== {split}: fire {int(X.fire.sum())}, changed {len(patch)}  (novlm={NOVLM}, vert={VERT})")
        cols = ["split", "id", "vid", "cat", "dim", "q", "mode", "static", "ov", "inside", "pmove", "retreat", "geo", "cs", "ans", "asrc", "new", "y"]
        print(X[X.fire][cols].round(3).to_string())
    X = pd.concat(allx, ignore_index=True)
    lab = X[X.y.notna()].copy()
    lab["y/geo"] = lab.y / lab.geo
    if len(lab):
        lab["m_old"] = A.row_mra(lab.ans.values, lab.y.values); lab["m_new"] = A.row_mra(lab.new.values, lab.y.values)
    print("\n== labelled rows that fire:", int(lab.fire.sum()) if len(lab) else 0)
    if len(lab) and lab.fire.any():
        print(lab[lab.fire][["split", "id", "vid", "q", "ans", "new", "y", "m_old", "m_new"]].round(3).to_string())
    if DIAG and len(lab):
        print("\n== DIAG labelled tdepth=prior rows with any occlusion evidence (retreat>=3 or inside>=5), any gate:")
        E = lab[(lab.retreat >= N_RETREAT) | (lab.inside >= N_INSIDE)]
        print(E[["split", "id", "vid", "cat", "kind", "dim", "head", "q", "static", "ov", "inside", "retreat", "geo", "ans", "y", "y/geo", "fails"]].round(3).to_string())
        print("\n== DIAG labelled S3/D3 non-lab size tdepth=prior rows (all): y/geo quantiles by evidence")
        S = lab[(lab.cat.isin(["S3", "D3"])) & (lab.kind == "size") & (~lab.fails.map(lambda f: "lab" in f))]
        S = S.assign(occ=(S.retreat >= N_RETREAT) | (S.inside >= N_INSIDE))
        for k, g_ in S.groupby("occ"):
            print(" occ", k, "n", len(g_), "y/geo q10/50/90", np.nanquantile(g_["y/geo"], [.1, .5, .9]).round(3),
                  "frac y>=0.95geo", round(float((g_["y/geo"] >= 0.95).mean()), 3))
        print(S[["split", "id", "vid", "dim", "head", "static", "ov", "inside", "retreat", "geo", "ans", "y", "y/geo", "fails"]].round(3).to_string())
