"""rul017: speed rows whose geo measurement stopped at meas_why == 'target=prior_object' (the target query resolved onto
the prior's SAM2 object) although the target query's OWN detections are a separate physical object.

Gate (code conditions on tracks_{split}_v2sc, cfg 'e' queries as used by the rj5sQ2 geo chain):
  * 2D speed row (cat D2/S2, q.kind == 'speed', q.t given), prior = time-less speed v of the prior object;
  * target query det boxes: >= 4 det frames, median det score >= 0.5, and IoU(det box, prior obj tracked box) < 0.2
    in >= 70% of the det frames (the dets sit on a different object than the prior's track).
Measurement:
  * det frames within +-0.75 s of t that are separate from the prior (IoU < 0.2); fall back to all separate det frames if < 4;
  * Theil-Sen slope (px/s) of each box edge (x1, x2, y1, y2), using only frames where that edge is off the image border;
    vx = median of x-edge slopes, vy = median of y-edge slopes, v_det_px = hypot(vx, vy);
  * v_prior_px = hypot of Theil-Sen slopes of the prior obj track centroid over the same time span (all tracked frames);
  * ratio = v_det_px / v_prior_px in [0.5, 2]; geo = v_prior * ratio (mode 2d_ratio).
Final answer follows the pipeline fusion: CaW within 2x -> geometric mean; else geo (strong 2d_ratio mode).
"""
import json, sys, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C21_SUB", "sub_xt_Nzn")
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
TR = "v2sc"
WIN = 0.75
IOU_SEP, SEP_FRAC, MIN_SCORE, MIN_DET = 0.2, 0.7, 0.5, 4
RLO, RHI = 0.5, 2.0
ALL = "--all" in sys.argv          # diagnostics: every 2D speed row with a time-less speed prior, not only target=prior_object
TARGET_IS_PRIOR = "--self" in sys.argv   # diagnostics: measure the prior query's own track edges (sanity of the method)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def theil_sen(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    if len(t) < 3:
        return np.nan
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]; ok = np.abs(dt) > 1e-9
    return float(np.median((y[j] - y[i])[ok] / dt[ok])) if ok.any() else np.nan


def qkey(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def measure(vj, P, fps, prior_v, sep_required=True):
    qs = {q["key"]: q for q in vj["queries"]}
    kt, kp = qkey(P.get("target")), qkey(P.get("prior_obj"))
    qt, qp = qs.get(kt), qs.get(kp)
    if qt is None or qp is None or qp.get("obj") is None:
        return dict(ok=False, why=f"no query ({kt}/{kp})")
    W, H = vj["W"], vj["H"]
    fidx = list(vj["fidx"]); pos = {f: k for k, f in enumerate(fidx)}
    ptrack = vj["objs"][str(qp["obj"])]
    dets = []
    for k, f in enumerate(vj["det_frames"]):
        b = qt["det_boxes"][k] if k < len(qt["det_boxes"]) else None
        if b is None:
            continue
        sc = qt["det_scores"][k]
        pb = ptrack[pos[f]] if f in pos and ptrack[pos[f]] is not None and ptrack[pos[f]].get("st") else None
        pi = iou(b, pb["st"]["box"]) if pb else np.nan
        dets.append(dict(f=f, t=f / fps, b=[float(x) for x in b], s=float(sc), iou=pi))
    n = len(dets)
    out = dict(kt=kt, kp=kp, obj_t=qt.get("obj"), obj_p=qp.get("obj"), n_det=n)
    if n == 0:
        return dict(out, ok=False, why="no dets")
    ious = np.array([d["iou"] for d in dets]); okiou = np.isfinite(ious)
    out["sep_frac"] = float(np.mean(ious[okiou] < IOU_SEP)) if okiou.any() else np.nan
    out["med_iou"] = float(np.nanmedian(ious)) if okiou.any() else np.nan
    out["med_score"] = float(np.median([d["s"] for d in dets]))
    t0 = P["q"]["t"]
    sep = [d for d in dets if np.isfinite(d["iou"]) and d["iou"] < IOU_SEP] if sep_required else list(dets)
    win = [d for d in sep if abs(d["t"] - t0) <= WIN + 1e-9]
    out["n_win"] = len(win)
    if len(win) < MIN_DET:
        win = sep
        out["win_fallback"] = True
    if len(win) < 3:
        return dict(out, ok=False, why="too few separate dets")
    tt = np.array([d["t"] for d in win]); B = np.array([d["b"] for d in win])
    border = [B[:, 0] <= 2, B[:, 2] >= W - 2, B[:, 1] <= 2, B[:, 3] >= H - 2]
    sl = []
    for e, col in enumerate((0, 2, 1, 3)):          # x1, x2, y1, y2 (box = [x1, y1, x2, y2])
        m = ~border[e]
        sl.append(theil_sen(tt[m], B[m, col]) if m.sum() >= 3 else np.nan)
    sx = [s for s in sl[:2] if np.isfinite(s)]; sy = [s for s in sl[2:] if np.isfinite(s)]
    out["edge_slopes"] = [None if not np.isfinite(s) else round(s, 2) for s in sl]
    if not sx or not sy:
        return dict(out, ok=False, why="no usable edge on an axis")
    vdet = float(np.hypot(np.median(sx), np.median(sy)))
    ta, tb = tt.min(), tt.max()
    pt, pc = [], []
    for k, f in enumerate(fidx):
        x = ptrack[k]
        if x is None or not x.get("st") or f / fps < ta - 1e-9 or f / fps > tb + 1e-9:
            continue
        pt.append(f / fps); pc.append(x["st"]["c"])
    if len(pt) < 3:
        return dict(out, ok=False, why="prior track too short in window")
    pc = np.array(pc, float)
    vpri = float(np.hypot(theil_sen(pt, pc[:, 0]), theil_sen(pt, pc[:, 1])))
    out.update(v_det_px=vdet, v_prior_px=vpri, win_t=(round(ta, 3), round(tb, 3)))
    if not (vpri > 0):
        return dict(out, ok=False, why="static prior")
    ratio = vdet / vpri
    out["ratio"] = ratio
    out["geo_new"] = prior_v * ratio
    why = []
    if n < MIN_DET: why.append("n_det")
    if out["med_score"] < MIN_SCORE: why.append("score")
    if not (np.isfinite(out["sep_frac"]) and out["sep_frac"] >= SEP_FRAC): why.append("iou")
    if not (RLO <= ratio <= RHI): why.append("ratio")
    return dict(out, ok=not why, why="pass" if not why else "gate:" + ",".join(why))


def fuse(g, cs):
    if np.isfinite(cs) and cs > 0 and abs(np.log10(g / cs)) <= np.log10(2):
        return float(np.sqrt(g * cs)), f"agree_gm with CaW {cs:.4g}"
    return float(g), f"dis_geo (CaW {cs:.4g})"


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = j
    patch, rows = {}, []
    for i, r in raw.iterrows():
        p = P.get(int(i))
        if p is None:
            continue
        if not ALL and str(r.meas_why) != "target=prior_object":
            continue
        q = p.get("q") or {}
        if p.get("cat") not in ("D2", "S2") or q.get("kind") != "speed" or q.get("t") is None:
            continue
        pr = [x for x in (p.get("priors") or []) if x.get("kind") == "speed" and x.get("t") is None and x.get("t1") is None
              and x.get("ok") and x.get("value_si")]
        if len(pr) != 1:
            continue
        vid = str(r.video_id)
        fp = f"{R}/geo/out/tracks_{split}_{TR}/{vid}.json"
        if not os.path.exists(fp):
            continue
        vj = json.load(open(fp))
        fps = float(r.fps)
        unit = q.get("unit") or "m/s"
        conv = {"m/s": 1.0, "cm/s": 100.0, "km/h": 3.6, "mm/s": 1000.0}.get(unit)
        if conv is None:
            continue
        m = measure(vj, p, fps, pr[0]["value_si"] * conv)
        old = float(det.answer[i]) if i in det.index else np.nan
        cs = float(det.cs[i]) if i in det.index else np.nan
        src = str(det.answer_source[i]) if i in det.index else ""
        y = float(det.y[i]) if ("y" in det.columns and i in det.index) else np.nan
        row = dict(split=split, id=int(i), vid=vid, cat=p["cat"], q=p["question"], prior=p["prior_text"], why_raw=str(r.meas_why),
                   old=old, old_src=src, cs=cs, g_old=float(det.g[i]) if i in det.index else np.nan, y=y, **{k: v for k, v in m.items()})
        if m["ok"] and str(r.meas_why) == "target=prior_object" and not src.startswith("patch"):
            v, fw = fuse(m["geo_new"], cs)
            row["new"] = v
            patch[str(int(i))] = dict(value=v, why=(f"rul017: target dets separate from prior obj (sep {m['sep_frac']:.2f}, "
                                                    f"score {m['med_score']:.2f}); edge speed {m['v_det_px']:.1f} px/s vs prior centroid "
                                                    f"{m['v_prior_px']:.1f} px/s -> geo {m['geo_new']:.4g}; {fw}"))
        rows.append(row)
    return patch, rows


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    allrows = []
    for split in ("test", "val"):
        patch, rows = run(split)
        allrows += rows
        if not ALL:
            json.dump(patch, open(f"{R}/cache/tmp/ag_c21comove_patch_{split}.json", "w"), indent=1)
            print(f"[{split}] patch rows: {len(patch)} -> cache/tmp/ag_c21comove_patch_{split}.json")
            for k, x in patch.items():
                print("  ", split, k, x)
    D = pd.DataFrame(allrows)
    if len(D):
        cols = [c for c in ["split", "id", "vid", "cat", "why_raw", "old", "old_src", "cs", "g_old", "y", "obj_t", "obj_p", "n_det", "sep_frac",
                            "med_iou", "med_score", "n_win", "edge_slopes", "v_det_px", "v_prior_px", "ratio", "geo_new", "new", "ok", "why"]
                if c in D.columns]
        print(D[cols].round(3).to_string())
        D.to_csv(f"{R}/cache/tmp/ag_c21comove_rows{'_all' if ALL else ''}.csv", index=False)
