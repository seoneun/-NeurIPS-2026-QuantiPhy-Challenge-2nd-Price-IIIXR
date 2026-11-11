"""rul069: a 3d_calib geo whose single timed depth anchor is provably invalid is fused as a weak geo.
track_depth_fn_v3 (one timed depth key) sets s0 = np.interp(t_key, trk.t, s) on the key-owner track (the query track of the
object the depth key names, prior or target role). When that track has NO border-free box (border flag 0 and box >= 2 px
from every frame edge) within +-0.15 s of t_key, np.interp clamps s0 to a later box (or reads a border-cut sliver), so D(t)
is anchored at the wrong time and is flat (clamped) before the owner's first box: the 3D reconstruction is not a strong
measurement. Such rows that the pipeline routed to geo on a disagreement (dis_geo / dis_motion_geo) get the pipeline's
weak-mode fusion gm(geo, CaW). agree_* rows and CaW-routed rows are left unchanged.
Inputs only: raw meas_mode / meas_tracks, parse depth keys and times, the raw's own tracks file (v2sc for meas_tracks=v2,
the loader fallback tracks_{split}/ for v1, then v3ntmv, v3nt), CSV fps. Labels only evaluate.
writes cache/tmp/ag_c44anchor_patch_{test,val}.json
usage: tools/run_limited.sh --io python analysis/v3/ag/rul069/rul069.py [SUB=sub_xt_Nzz2] [GAP=0.15]"""
import sys, os, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz2"
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 300)
ROUTED = ("dis_geo", "dis_motion_geo")


def tracks_path(split, vid, mt):
    for suf in ("_v2sc", "", "_v3ntmv", "_v3nt"):
        if suf == "_v2sc" and mt == "v1":        # the raw read meas_tracks=v1 videos from the loader fallback tracks_{split}/
            continue
        p = f"{R}/geo/out/tracks_{split}{suf}/{vid}.json"
        if os.path.exists(p):
            return p, suf
    return None, None


def owner_gaps(vj, role, phr, tkey, fps):
    """gap (s) from t_key to the nearest border-free box of every query track of this role + phrase (both configs)"""
    W, H = vj["W"], vj["H"]
    out = []
    for qn in ("queries", "queries_d"):
        for q in vj.get(qn) or []:
            if role not in (q.get("roles") or []) or q.get("phrase") != phr:
                continue
            fr = vj["objs"][str(q["obj"])] if q.get("obj") is not None else q.get("frames")
            if not fr:
                continue
            tb = []
            for k, f in enumerate(fr):
                st = (f or {}).get("st")
                if not st:
                    continue
                x1, y1, x2, y2 = st["box"]
                if st.get("border", 0) == 0 and x1 >= 2 and y1 >= 2 and x2 <= W - 2 and y2 <= H - 2:
                    tb.append(vj["fidx"][k] / fps)
            out.append((qn, q.get("key"), q.get("obj"), min([abs(t - tkey) for t in tb]) if tb else np.inf))
    return out


def scan(split):
    PR = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); PR[int(j["id"])] = j
    raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    VJ, rows = {}, []
    for i, r in raw.iterrows():
        mode = str(r.meas_mode)
        if not mode.startswith("3d_calib"):
            continue
        p = PR[int(i)]
        pk, tk = p.get("prior_dkeys") or [], p.get("target_dkeys") or []
        if pk:
            keys, role, phr = pk, "prior", (p.get("prior_obj") or {}).get("phrase")
        elif tk:
            keys, role, phr = tk, "target", (p.get("target") or {}).get("phrase")
        else:
            continue
        timed = [d for d in p.get("depth") or [] if d["key"] in keys and d.get("t") is not None]
        if len(timed) != 1:
            continue
        tkey = float(timed[0]["t"])
        pth, suf = tracks_path(split, r.video_id, str(r.meas_tracks))
        if pth is None:
            continue
        if pth not in VJ:
            VJ[pth] = json.load(open(pth))
        og = owner_gaps(VJ[pth], role, phr, tkey, float(r.fps))      # CSV fps (never the container fps)
        if not og:
            continue
        gmin = min(x[3] for x in og)
        if not gmin > GAP:
            continue
        rows.append(dict(split=split, id=int(i), vid=r.video_id, cat=det.cat.get(i), role=role, phr=phr, key=timed[0]["key"], tkey=tkey,
                         gap=gmin, trk=suf or "plain", src=det.answer_source.get(i), g=det.g.get(i), cs=det.cs.get(i),
                         ans=det.answer.get(i), mode=mode, q=str(r.question)[:70]))
    return pd.DataFrame(rows), det


out = {}
for split in ("test", "val"):
    F, det = scan(split)
    P = {}
    if len(F):
        F["new"] = np.nan
        for k, r in F.iterrows():
            if r.src not in ROUTED or not (np.isfinite(r.g) and r.g > 0 and np.isfinite(r.cs) and r.cs > 0):
                continue
            v = float(np.sqrt(r.g * r.cs))
            F.loc[k, "new"] = v
            P[str(r.id)] = dict(value=v, why=f"rul069: 3d_calib depth anchor invalid - key '{r.key}' t={r.tkey:g}s has no border-free "
                                             f"box of the {r.role} track '{r.phr}' within {GAP:g}s (nearest {r.gap:.3f}s; np.interp clamps s0); "
                                             f"{r.src} geo {r.g:.4g} -> weak geo gm(geo {r.g:.4g}, CaW {r.cs:.4g})")
    json.dump(P, open(f"{R}/cache/tmp/ag_c44anchor_patch_{split}.json", "w"), indent=1)
    out[split] = (P, det, F)
    print(f"[{split}] flagged (mode 3d_calib*, 1 timed key, owner gap > {GAP}s): {len(F)} rows; patched {len(P)}")
    if len(F):
        print(F.drop(columns=["split"]).round(4).to_string(index=False))
        print("changed by cat:", F[F.new.notna()].cat.value_counts().to_dict())

from lab_rows import labelled
import analyze_caw as A
L = labelled(SUB)
for split in ("val", "test"):
    P = out[split][0]
    hit = [i for i in L[L.split == split].index if str(int(i)) in P]
    print(f"labelled {split} rows changed: {len(hit)}")
    for i in hit:
        y = L.y[i]; old = L.answer[i]; new = P[str(int(i))]["value"]
        print("  ", i, L.video_id[i], y, old, "->", new, A.row_mra(np.array([old]), np.array([y]))[0], "->", A.row_mra(np.array([new]), np.array([y]))[0])
F = out["flagged_flag"] = None
