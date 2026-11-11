"""rul202: re-read a border-cut size target's extent from frames uncut on the MEASURED axis.

When the pipeline's fully-uncut frame set (tr.oke) falls back to all frames (too few frames are uncut on every side), the
size extent is the median over frames that may be cut by the image border on the very axis being measured -> the extent is
biased low. Here the extent axis is identified per frame (ext_px matches box w or box h), frames touching the border on that
axis are dropped, and the extent is re-aggregated (median / p75 for diameter) over the remaining frames when they are many
(>= 8), consistent (CV <= 0.05) and the cut share is substantial (>= 0.2). g_new = g * new_px/old_px (1.05 <= r <= 1.4),
re-fused with tedge's fuse(). Inputs only: tracks, raw geo, detail; labels used only for evaluation.
Writes cache/tmp/ag_c95border_axisext_patch_{test,val}.json and ..._diag.csv."""
import os, sys, json, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, R + "/geo")
import measure_v3 as MV
import measure as M1
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
TAG = "rul202"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz29"
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
TOL = 2.0
MODES = {"2d_ratio", "vlm_2d_ratio"}
STRONG = {"2d_ratio"}
SRCS = {"agree_gm", "agree_wgm", "dis_geo", "geo_cawinvalid"}
MATCH, NKEEP, FCUT, CVMAX, RLO, RHI = 0.05, 8, 0.2, 0.05, 1.05, 1.4
BAND = math.log(2.0)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid", "sruler_geo"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}


def fuse(r, gn, kind, strong):   # copied verbatim from analysis/v3/ag/rul174/rule.py (tedge fuse)
    s, a, g = str(r.answer_source), float(r.answer), float(r.g)
    if s.startswith("mo_guard:"):
        if s[9:] in GEO and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    c = float(r.cs) if np.isfinite(r.cs) else np.nan
    in_old = np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if s == "geo_cawinvalid" or s == "sruler_geo":
        return gn, s
    if kind in GEOK:
        if s in AGREE or s in GEO:
            if in_new:
                return gn, f"{s} -> in 2x band, geo kind -> geo (agree_geo_kind)"
            return (gn, f"{s} -> outside 2x band, strong -> geo") if strong else (None, f"{s} -> outside band, weak -> CaW (skip)")
        if s in CAWD and in_new and not in_old:
            return gn, f"{s} entered 2x band, geo kind -> geo (agree_geo_kind)"
        return None, s
    if s in AGREE:
        if not in_new:
            return (gn, "left 2x band, strong -> geo") if strong else (None, "left 2x band, weak -> CaW (skip)")
        lg = math.log(g / c)
        w = math.log(a / c) / lg if abs(lg) > 1e-6 else 0.5
        if not (0.3 <= w <= 0.9):
            w = 0.5
        return a * (gn / g) ** w, f"{s} w{w:.2f}"
    if s in GEO:
        if in_new and s == "dis_geo":
            return math.sqrt(gn * c), "entered 2x band -> agree_gm"
        return gn, s
    if s in CAWD:
        if in_new and not in_old:
            return math.sqrt(gn * c), f"{s} entered 2x band -> agree_gm"
        return None, s
    return None, s


def axis_ext(tr, dim, phrase, W, H):
    okc = np.asarray(tr.okc); oke = np.asarray(tr.oke)
    n = len(okc)
    if n == 0:
        return dict(why="no okc")
    e, _ = tr.ext_px(dim, phrase)
    e = np.asarray(e, float)
    old = M1.agg_ext(e, dim)
    nfu = int(sum(1 for j in okc if (tr.st[j].get("border", 0) or 0) == 0))
    d = dict(n=n, nfu=nfu, noke=len(oke), old=old)
    # gate 1: oke fell back to all frames (too few fully-uncut frames)
    if not (len(oke) == n and len(e) == n and nfu < max(2, 0.3 * n)):
        d["why"] = "g1 no fallback"; return d
    B = np.array([tr.st[j]["box"] for j in okc], float)
    bw, bh = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
    cx = (B[:, 0] <= TOL) | (B[:, 2] >= W - 1 - TOL); cy = (B[:, 1] <= TOL) | (B[:, 3] >= H - 1 - TOL)
    axx = np.abs(e - bw) <= np.abs(e - bh)
    ax = "x" if np.median(axx) >= 0.5 else "y"
    match = float(np.median(np.minimum(np.abs(e - bw), np.abs(e - bh)) / np.maximum(e, 1)))
    d.update(ax=ax, match=match)
    if match > MATCH:
        d["why"] = "g2 axis match"; return d
    cut = cx if ax == "x" else cy
    full = (bw >= W - 1 - 2 * TOL) if ax == "x" else (bh >= H - 1 - 2 * TOL)
    keep = ~cut
    d.update(fcut=float(cut.mean()), nkeep=int(keep.sum()), full=int(full.any()))
    if keep.sum() < NKEEP or cut.mean() < FCUT or full.any():
        d["why"] = "g3 keep/cut/full"; return d
    ek = e[keep]
    cv = float(np.std(ek) / max(np.mean(ek), 1e-6))
    d["cv"] = cv; d["new"] = M1.agg_ext(ek, dim)
    if cv > CVMAX:
        d["why"] = "g3 cv"; return d
    d["why"] = "ok"
    return d


def main():
    diag, patches = [], {}
    for split in ("test", "val"):
        raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv", index_col=0)
        det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv" if split == "test" else f"{R}/submissions/{SUB}_valdetail.csv", index_col=0)
        if "cs" not in det:
            det["cs"] = det["c"] if "c" in det else np.nan
        cand = det[((det.kind == "size") if "kind" in det else True) & det.answer_source.astype(str).isin(SRCS)]
        cand = cand[raw.meas_mode.reindex(cand.index).astype(str).isin(MODES)]
        gr = raw.geo_value.reindex(cand.index).astype(float)
        cand = cand[(np.abs(cand.g / gr - 1) <= 0.01)]
        P = {}
        TV, VC = {}, {}
        for i, r in cand.iterrows():
            rw = raw.loc[i]; ver = str(rw.meas_tracks)
            if ver not in TV:
                MV.TRACK_VER = ver
                TV[ver] = MV.load_split_tracks(split, "e")
            byv, vjs = TV[ver]; v = rw.video_id
            row = dict(split=split, id=i, vid=v, q=r.get("question", ""), src=r.answer_source, mode=rw.meas_mode, g=r.g, cs=r.cs, ans=r.answer)
            recs = [x for x in byv.get(v, []) if x["id"] == i] if v in vjs else []
            if not recs:
                row["why"] = "no rec"; diag.append(row); continue
            rec = recs[0]; vj = vjs[v]; W, H = float(vj["W"]), float(vj["H"])
            if (ver, v) not in VC:
                try:
                    VC[(ver, v)] = MV.Video(vj, "e")
                except Exception:
                    VC[(ver, v)] = None
            V = VC[(ver, v)]
            q = rec.get("q") or {}
            if q.get("kind") != "size":
                row["why"] = "not size"; diag.append(row); continue
            tr = V.get(rec["target"]) if (V is not None and rec.get("target")) else None
            if tr is None:
                row["why"] = "no track"; diag.append(row); continue
            d = axis_ext(tr, q.get("dim"), (rec["target"] or {}).get("phrase"), W, H)
            row.update(d)
            if d.get("why") == "ok":
                mpp = float(rw.meas_mpp) if pd.notna(rw.meas_mpp) else np.nan
                if not (np.isfinite(mpp) and abs(r.g / mpp / d["old"] - 1) <= 0.01):
                    row["why"] = "old_px mismatch"
                else:
                    rr = d["new"] / d["old"]; row["r"] = rr
                    if not (RLO <= rr <= RHI):
                        row["why"] = "r out of range"
                    else:
                        gn = r.g * rr
                        val, why = fuse(r, gn, "size", str(rw.meas_mode) in STRONG)
                        if val is None:
                            row["why"] = "fuse skip " + why
                        else:
                            row["new_ans"] = val
                            P[str(i)] = {"value": float(val), "why": f"{TAG}: {d['ax']}-axis uncut ext {d['old']:.1f}->{d['new']:.1f}px (r{rr:.3f}, keep {d['nkeep']}/{d['n']}), g {r.g:.3f}->{gn:.3f}; {why}"}
            diag.append(row)
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        patches[split] = P
        print(split, "candidates", len(cand), "patched", len(P))
    D = pd.DataFrame(diag)
    D.to_csv(f"{R}/cache/tmp/ag_{TAG}_diag.csv", index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    print(D.why.value_counts().to_string())
    cols = [c for c in ["split", "id", "vid", "src", "mode", "ax", "match", "fcut", "nkeep", "cv", "old", "new", "r", "g", "cs", "ans", "new_ans", "why"] if c in D]
    sel = D.why.isin(["ok", "r out of range", "old_px mismatch", "g3 cv"]) | D.why.astype(str).str.startswith("fuse")
    print(D[sel][cols].to_string())
    for split, P in patches.items():
        for k, x in P.items():
            print(split, k, x)
    # evaluation on labelled rows (labels only evaluate)
    import analyze_caw as A
    from lab_rows import labelled
    L = labelled(SUB)
    for split, P in patches.items():
        ids = [int(k) for k in P]
        hit = L[(L.split == split) & L.index.isin(ids)]
        print("labelled hits", split, len(hit))
        for i, lr in hit.iterrows():
            print(split, i, lr.y, lr.m, A.row_mra(np.array([P[str(i)]["value"]]), np.array([lr.y])))


if __name__ == "__main__":
    main()
