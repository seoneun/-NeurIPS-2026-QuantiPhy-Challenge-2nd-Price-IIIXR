#!/usr/bin/env python
"""B2 'bind a size prior to the only mover' (rule rul006).
Trigger (per row, computed from tracks / parse / geo only):
  * the row's prior is a size prior (diameter / length / ...) and the detector chain (v2sc, meas v3m) failed on the prior:
    prior_no_track, prior_low_det, or (diameter prior) the prior box is not round (median max/min side > 1.4);
  * the video has exactly ONE distinct mover over the v2sc + v3nt tracks (cover >= 50 %, centre span >= max(30 px, 1.5 x size));
  * the prior phrase is the target of a motion question in the same video (so the prior object IS the mover);
  * for a diameter prior the mover is round (median max/min side <= 1.2).
Bind: prior px = median mean(w, h) of the mover; mpp_bind = prior / px. If the row's geo was measured with a scale within 15 %
of mpp_bind (the VLM chain already bound the prior to the mover; meas_v3v mpp), the geo value stays and only its mode is
promoted from weak (vlm_2d_ratio) to strong 2d_ratio; rows whose geo scale differs by more than 15 % are skipped (scale_mismatch:
the geo is not verified against the mover, and a rescale path has no evidence behind it).
Fusion recomputed like the pipeline for strong 2D rows: within 2x of CaW -> size g^w cs^(1-w) (w 0.65 S, 0.7 D),
dist2 / avgspeed geo alone, other kinds gm; outside 2x -> geo.
Skipped: patch* rows, rows outside the plain fusion sources, rows whose TARGET track is the mover although the target phrase is
not the prior phrase (a second phrase grounded on the mover, e.g. 'floor mirror' -> caught by C1 already), 3D videos.
Writes cache/tmp/ag_bindmover_patch_{test,val}.json."""
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3/ag/rul006")
from scan import load_tracks, obj_stats, unique_movers, qobj, MOTION
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nza"
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"; M3 = "_v3m_rrj5sQ"; M3V = "_v3v_rntrj5sQ"
PLAIN = {"agree_gm", "agree_wgm", "dis_size_gm", "dis_caw_routed", "dis_geo", "agree_geo_kind"}
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)


def fuse_strong(g, cs, kind, cat):
    if not (cs > 0):
        return g, "geo_only"
    if abs(np.log10(g / cs)) <= np.log10(2.0):
        if kind in ("dist2", "avgspeed"):
            return g, "agree_geo_kind"
        if kind == "size":
            w = 0.7 if cat.startswith("D") else 0.65
            return g ** w * cs ** (1 - w), "agree_wgm"
        return float(np.sqrt(g * cs)), "agree_gm"
    return g, "dis_geo"


allrows = []
for split in ("test", "val"):
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln); P[int(x["id"])] = x
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
    m3 = pd.read_csv(f"{R}/geo/out/meas_{split}_v2{M3}.csv").set_index("id")
    m3v = pd.read_csv(f"{R}/geo/out/meas_{split}_v2{M3V}.csv").set_index("id")
    motion_t = {}
    for i, x in P.items():
        if (x.get("q") or {}).get("kind") in MOTION and (x.get("target") or {}).get("phrase"):
            motion_t.setdefault(x["video_id"], set()).add(x["target"]["phrase"].strip())
    vinfo = {}
    for i, x in P.items():
        pr = (x.get("priors") or [{}])[0]
        vid = x["video_id"]
        if pr.get("kind") != "size" or not pr.get("value_si"):
            continue
        if str(x.get("video_type", ""))[1:2] != "2":
            continue
        if vid.startswith("internet"):                 # real footage: hands, people, camera shake -> no "only mover" (mover_scan rule)
            continue
        pph = ((x.get("prior_obj") or {}).get("phrase") or "").strip()
        t2 = load_tracks(split, "_v2sc", vid); t3 = load_tracks(split, "_v3nt", vid)
        if t2 is None or t3 is None or not pph or "objs" not in t2 or "objs" not in t3:
            continue
        o2, q2 = qobj(t2, pph)
        if q2 is None:
            continue
        s2 = obj_stats(t2["objs"][str(o2)]) if (o2 is not None and str(o2) in t2["objs"]) else None
        notround = pr.get("dim") == "diameter" and s2 is not None and s2["asp"] > 1.4
        weak = s2 is None or float(q2.get("best_score") or 0) < 0.2   # detector found no / only a low-score prior box
        if not (weak or notround):
            continue
        if pph not in motion_t.get(vid, set()):
            continue
        if vid not in vinfo:
            vinfo[vid] = unique_movers(split, vid)
        G = vinfo[vid]
        if len(G) != 1:
            continue
        mst = max(G[0], key=lambda m: m[2]["cover"])[2]
        if pr.get("dim") == "diameter" and mst["asp"] > 1.2:
            continue
        px = 0.5 * (mst["w"] + mst["h"])
        mpp_b = float(pr["value_si"]) / px
        # the row's own geo scale and target track
        tph = ((x.get("target") or {}).get("phrase") or "").strip()
        o3t, _ = qobj(t3, tph)
        tgt_is_mover = any(tv == "_v3nt" and str(o) == str(o3t) for tv, o, _ in G[0])
        r = det.loc[i]
        src = str(r.answer_source); g = float(r.g) if pd.notna(r.g) else np.nan; cs = float(r.cs) if pd.notna(r.cs) else np.nan
        mpp_r = float(m3v.mpp.get(i)) if pd.notna(m3v.mpp.get(i)) else np.nan
        rec = dict(split=split, id=i, trig=("weak" if weak else "notround"), vid=vid, cat=r["cat"], kind=r["kind"], q=x["question"][:60], src=src, mode=r["mode"], g=g, cs=cs, ans=float(r.answer),
                   mpp_row=mpp_r, mpp_bind=mpp_b, px=px, tgt_mover=tgt_is_mover, tph=tph)
        skip = None
        if src.startswith("patch"):
            skip = "patch"
        elif src not in PLAIN:
            skip = "src"
        elif not (g > 0):
            skip = "nogeo"
        elif tgt_is_mover and tph != pph:
            skip = "target_on_mover"
        elif not (mpp_r > 0):
            skip = "no_row_mpp"
        elif abs(np.log(mpp_b / mpp_r)) >= np.log(1.15):
            skip = "scale_mismatch"
        if skip is None:
            ratio = mpp_b / mpp_r
            g2 = g                                        # geo already on the mover scale: value kept, mode promoted
            nv, ns = fuse_strong(g2, cs, r["kind"], r["cat"])
            rec.update(g2=g2, new=nv, nsrc=ns, ratio=ratio)
        rec["skip"] = skip
        allrows.append(rec)

Z = pd.DataFrame(allrows)
print("trigger rows", len(Z), "videos", Z.groupby("split").vid.unique().to_dict() if len(Z) else {})
if len(Z):
    print(Z.drop(columns=["tph"]).round(4).to_string())
for split in ("test", "val"):
    out = {}
    if len(Z):
        for _, r in Z[(Z.split == split) & Z.skip.isna()].iterrows():
            if abs(r.new - r.ans) > 1e-9 * max(1.0, abs(r.ans)):
                out[str(int(r.id))] = dict(value=float(r.new), why=f"B2 rul006: prior bound to the only mover ({r.px:.0f}px, mpp {r.mpp_bind:.5f}); geo {r.g2:.3g} strong -> {r.nsrc}")
    json.dump(out, open(f"{R}/cache/tmp/ag_bindmover_patch_{split}.json", "w"), indent=1)
    print(split, "patch rows", len(out))
    if out:
        ch = Z[(Z.split == split) & Z.id.astype(str).isin(out.keys())]
        print(" by cat", ch.groupby("cat").size().to_dict(), "| by kind", ch.groupby("kind").size().to_dict(), "| src->nsrc", ch.groupby(["src", "nsrc"]).size().to_dict())
