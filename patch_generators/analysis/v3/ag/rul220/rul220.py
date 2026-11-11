#!/usr/bin/env python
"""rul220: 3D size rows (S3 / D3) whose target box is cut by the image border on the measured axis in most frames. The box
extent is then a lower bound on the size, not an estimate, so a fusion that averages it with a larger CaW read (or takes it
alone) lands below the bound. When CaW's crop read is at least 1.3x the cut geo, CaW is the only bound-consistent candidate
and answers. Extends rul096 (2D ratio rows) to the 3D measurement modes. No ids, no labels in the logic.
usage: python rul220.py [write]"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul220"; WRITE = "write" in sys.argv[1:]
HORIZ = {"width", "length", "diameter", "wingspan", "span"}
SRC = {"agree_gm", "agree_wgm", "dis_size_gm", "d3size_dimgm", "d3size_unstable_gm", "dis_geo", "agree_geo_kind"}


def obj_of(tr, phrase):
    """object id of the query whose phrase matches; v2/v3 files list queries with an obj field, v1 files map key -> obj"""
    ph = str(phrase or "").strip().lower()
    qs = tr.get("queries")
    if isinstance(qs, dict):                      # v1 format
        for q in tr.get("queries_d") or []:
            if str(q.get("phrase", "")).strip().lower() == ph and qs.get(q.get("key")) is not None:
                return qs[q["key"]]
        return None
    for qk in ("queries", "queries_d"):
        for q in tr.get(qk) or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("obj") is not None:
                return q["obj"]
    return None


def states_of(tr, phrase):
    """per-frame states of the target: v2/v3 files via the query's obj id, v1 files via the query's own frames list"""
    ph = str(phrase or "").strip().lower()
    if "objs" not in tr:                              # v1 format
        for q in tr.get("queries") or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("frames"):
                return q["frames"]
        return None
    o = obj_of(tr, phrase)
    return (tr["objs"].get(str(o)) or None) if o is not None else None


def boxes(st_list):
    return [(k, s["st"]["box"]) for k, s in enumerate(st_list or []) if s and s.get("st")]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def cut_frac(tr, st_list, axis):
    """share of tracked frames whose box touches the frame border on the given axis (0 = x: left/right, 1 = y: top/bottom)"""
    W, H = tr["W"], tr["H"]; n = cut = 0
    for s in st_list or []:
        if not (s and s.get("st")):
            continue
        b = s["st"]["box"]; n += 1
        if axis == 0:
            cut += int(b[0] <= 1 or b[2] >= W - 1)
        else:
            cut += int(b[1] <= 1 or b[3] >= H - 1)
    return (cut / n if n else 0.0), n


def rule(sp, verbose=False):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
    PP = {p["id"]: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    out, diag = {}, []
    for i, r in A.iterrows():
        p = PP[i]; q = p["q"]
        if not (r["cat"] in ("S3", "D3") and q.get("kind") == "size" and str(r.source) in SRC and np.isfinite(r.g) and r.g > 0 and np.isfinite(r.cs) and r.cs > 0):
            continue
        mode = str(r["mode"])
        if not mode.startswith(("3d", "vlm_3d")):
            continue
        trk = str(raw.meas_tracks[i]) if isinstance(raw.meas_tracks[i], str) else "v2sc"
        trk = {"v2": "v2sc", "v1": ""}.get(trk, trk)
        if mode.startswith("vlm_"):
            trk = "v3ntmv"
        fn = f"{R}/geo/out/tracks_{sp}_{trk}/{r.video}.json" if trk else f"{R}/geo/out/tracks_{sp}/{r.video}.json"
        if not os.path.exists(fn):
            continue
        tr = json.load(open(fn)); ph = (p.get("target") or {}).get("phrase"); st = states_of(tr, ph)
        if not st:
            continue
        axis = 0 if q.get("dim") in HORIZ else 1
        cf, n = cut_frac(tr, st, axis)
        if n < 5 or cf < 0.6:
            continue
        # the measured box must be the same object as the VLM-anchored box (a box on a wrong object is no bound): median IoU >= 0.3
        fv = f"{R}/geo/out/tracks_{sp}_v3ntmv/{r.video}.json"
        if os.path.exists(fv) and trk != "v3ntmv":
            sv = states_of(json.load(open(fv)), ph)
            if sv:
                bm, bv = dict(boxes(st)), dict(boxes(sv)); common = [k for k in bm if k in bv]
                if common and float(np.median([iou(bm[k], bv[k]) for k in common])) < 0.3:
                    continue
        cand = max([x for x in (r.cs, r.cfull) if np.isfinite(x) and x > 0]); ratio = cand / r.g
        diag.append((i, cf, ratio))
        tag = f"target box cut by the border on the {'x' if axis == 0 else 'y'} axis in {cf:.0%} of frames: geo {r.g:.3g} is a lower bound"
        if ratio >= 1.3 and r.answer < cand * 0.98:
            out[i] = (float(cand), tag + f"; CaW {cand:.3g} (max of crop / full read) >= 1.3x geo answers")
        elif cand < r.g and r.answer < r.g * 0.98:
            out[i] = (float(r.g), tag + f"; CaW {cand:.3g} lies below the bound, the fused {r.answer:.3g} rises to the bound")
    if verbose:
        print(f"  [{sp}] rows with a border-cut target box on the measured axis: {len(diag)}; of them CaW >= 1.3x geo: {len(out)}")
    return out


if __name__ == "__main__":
    res = {}
    for sp in ("test", "val"):
        res[sp] = rule(sp, verbose=True); print(sp, len(res[sp]), "rows")
        A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
        for i, (v, why) in res[sp].items():
            lab = f" | y {A.y[i]:.4g} MRA {A.mra[i]:.1f} -> {float(FC.row_mra(np.array([v]), np.array([A.y[i]]))[0]):.1f}" if sp == "val" else ""
            print(f"  {i} {A.video[i]} {A['cat'][i]} {str(A.question[i])[:50]}: {A.answer[i]:.4g} -> {v:.4g} ({A.source[i]}, g {A.g[i]:.3g}){lab}")
    if WRITE:
        for sp in res:
            json.dump({str(i): dict(value=round(float(v), 7), why=f"{TAG}: {why}") for i, (v, why) in res[sp].items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print("written")
