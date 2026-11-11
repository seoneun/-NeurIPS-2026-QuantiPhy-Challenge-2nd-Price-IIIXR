#!/usr/bin/env python
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/analysis/v3/ag/rul006")
from scan import load_tracks, obj_stats, same_traj, qobj, MOTION
SUB = sys.argv[1] if len(sys.argv) > 1 else "pred_t060"; GT = sys.argv[2] if len(sys.argv) > 2 else "c17"
GATE = os.environ.get("BM_GATE", "0") == "1"
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


def own_movers(split, vid, phrases):
    allm = []
    for tv in ("_v2sc", "_v3nt"):
        tj = load_tracks(split, tv, vid)
        if tj is None:
            continue
        objs = {}
        for q in tj.get("queries") or []:
            if (q.get("phrase") or "").strip() in phrases and q.get("obj") is not None:
                objs.setdefault(str(q["obj"]), (q.get("phrase") or "").strip())
        for o, lst in (tj.get("objs") or {}).items():
            if o not in objs: continue
            st = obj_stats(lst)
            if st and st["cover"] >= 0.5 and st["span"] >= max(30.0, 1.5 * st["size"]):
                allm.append((tv, o, st, objs[o]))
    groups = []
    for m in allm:
        for g in groups:
            if same_traj(g[0][2], m[2]):
                g.append(m); break
        else:
            groups.append([m])
    return groups


allrows = []
for split in ("test",):
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln); P[int(x["id"])] = x
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    ownm = {v: pd.read_csv(f"{R}/cache/tmp/ag_ownt_meas_{split}_{v}_ownt.csv").set_index("id") for v in ("m", "v")}
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
        if vid.startswith("internet"):
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
        weak = s2 is None or float(q2.get("best_score") or 0) < 0.2
        if not (weak or notround):
            continue
        if GATE and pph not in motion_t.get(vid, set()):
            continue
        own = {s.strip() for s in (pph, ((x.get("target") or {}).get("phrase") or ""), ((x.get("target2") or {}).get("phrase") or "") if isinstance(x.get("target2"), dict) else (x.get("target2") or "")) if s and s.strip()}
        G = own_movers(split, vid, own)
        if len(G) != 1:
            continue
        if not any(ph_ == pph for _, _, _, ph_ in G[0]):
            continue
        mst = max(G[0], key=lambda m: m[2]["cover"])[2]
        if pr.get("dim") == "diameter" and mst["asp"] > 1.2:
            continue
        px = 0.5 * (mst["w"] + mst["h"])
        mpp_b = float(pr["value_si"]) / px
        tph = ((x.get("target") or {}).get("phrase") or "").strip()
        o3t, _ = qobj(t3, tph)
        tgt_is_mover = any(tv == "_v3nt" and str(o) == str(o3t) for tv, o, _, _ in G[0])
        r = det.loc[i]
        src = str(r.answer_source); g = float(r.g) if pd.notna(r.g) else np.nan; cs = float(r.cs) if pd.notna(r.cs) else np.nan
        mpp_g = float(pd.to_numeric(raw.meas_mpp.get(i), errors="coerce"))
        msrc = str(raw.merge_src.get(i)); var = msrc[-1:] if msrc[-1:] in ("m", "v") else None
        mpp_v = np.nan; own_ok = False
        if var is not None and i in ownm[var].index:
            ov = ownm[var].loc[i]; gv = float(pd.to_numeric(raw.geo_value.get(i), errors="coerce"))
            own_ok = bool(ov.ok) and np.isclose(float(ov.value), gv, rtol=1e-6) if pd.notna(ov.value) and gv > 0 else False
            mpp_v = float(ov.mpp) if own_ok and pd.notna(ov.mpp) else np.nan
        mpp_r = mpp_g if mpp_g > 0 else mpp_v
        rec = dict(split=split, id=i, msrc=msrc, trig=("weak" if weak else "notround"), vid=vid, cat=r["cat"], kind=r["kind"], q=x["question"][:60], src=src, mode=r["mode"], g=g, cs=cs, ans=float(r.answer),
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
        elif not (mpp_g > 0) and not own_ok:
            skip = "own_meas_mismatch"
        elif not (mpp_r > 0):
            skip = "no_row_mpp"
        elif abs(np.log(mpp_b / mpp_r)) >= np.log(1.15):
            skip = "scale_mismatch"
        if skip is None:
            ratio = mpp_b / mpp_r
            g2 = g
            nv, ns = fuse_strong(g2, cs, r["kind"], r["cat"])
            rec.update(g2=g2, new=nv, nsrc=ns, ratio=ratio)
        rec["skip"] = skip
        allrows.append(rec)

Z = pd.DataFrame(allrows)
print("trigger rows", len(Z), "videos", Z.groupby("split").vid.unique().to_dict() if len(Z) else {})
if len(Z):
    print(Z.drop(columns=["tph"]).round(4).to_string())
for split in ("test",):
    out = {}
    if len(Z):
        for _, r in Z[(Z.split == split) & Z.skip.isna()].iterrows():
            if True:
                out[str(int(r.id))] = dict(value=float(r.new), why=f"bindmoverS3: prior bound to the only mover ({r.px:.0f}px, mpp {r.mpp_bind:.5f}); geo {r.g2:.3g} strong -> {r.nsrc}")
    json.dump(out, open(f"{R}/cache/tmp/ag_bindmoverS3_patch_{split}.json", "w"), indent=1)
    print(split, "patch rows", len(out))
    if out:
        ch = Z[(Z.split == split) & Z.id.astype(str).isin(out.keys())]
        print(" by cat", ch.groupby("cat").size().to_dict(), "| by kind", ch.groupby("kind").size().to_dict(), "| src->nsrc", ch.groupby(["src", "nsrc"]).size().to_dict())
