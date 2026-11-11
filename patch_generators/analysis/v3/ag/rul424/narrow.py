#!/usr/bin/env python
"""Narrowed stale-track fix (ag rul424). The rj5sQ chains measure on v2sc / v2scu tracks but the cross-grounding flag
(xground.py: detector track never meets the VLM referring box -> weak VLM-track value, merge_src xg_v) was computed on the
original v2 tracks. analysis/v3/ag/stale/build_sQ3.sh recomputed it on the measured tracks (geo rj5sQ3). This script keeps
the rj5sQ2 geo and takes the rj5sQ3 row only where the recomputed flag changes the row's source AND none of:
  (1) ruler stage: Q2 or Q3 merge_src in {ruler, selfruler, sruler} (RL_OVER_VLM overwrites weak rows with ruler values);
  (2) the row is lab-pooled / depth_bound / already patched in sub_xt_Nzc, xt_Nzc or xt_Nzcu (answer_source);
  (3) a NEW flag (Q3 flagged, Q2 not) raised by a key whose measured track equals the mover / Molmo-regrounded track
      (v2sc == v3ntmv, v2scu == v3ntmvu): the regrounded track is the deliberate choice, the referring box is not.
writes preds/{split}_geo_v4ntrj5sQ3n{,u}srngsr_d_vw(_raw).csv and cache/tmp/ag_stale2_rows_{split}.csv
usage (rj5sQ chain env as in build_sQ3.sh): python narrow.py <split>"""
import json, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import measure_v3 as MV
import track_common as tc
from qp_common import OUT_COLS
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
split = sys.argv[1]
RULER = {"ruler", "selfruler", "sruler"}
BADSRC = ("lab_pool", "depth_bound", "patch")

def flags(trk, skip_sa2va):
    MV.FIX = set("abcdef"); MV.TRACK_VER = trk
    byv, vjs = MV.load_split_tracks(split, "e")
    SA = {(v, q["key"]) for v, vj in vjs.items() for q in (vj.get("queries") or []) if q.get("sa2va")} if skip_sa2va else set()
    VG = {}
    for ln in open(f"{R}/geo/out/vlmground_{split}_notext.jsonl", encoding="utf-8"):
        x = json.loads(ln); VG.setdefault((x["video_id"], x["key"]), []).append(x)
    agree = {}
    for (v, k), xs in VG.items():
        vj = vjs.get(v)
        if vj is None: continue
        V = MV.Video(vj, "e"); t = V.tr.get(k)
        if t is None or not t.n: continue
        hit = n = 0
        for x in xs:
            j = x["j"]; bl = [b for b in x["boxes_rel1000"] if b and len(b) == 4]
            if not bl or j >= len(t.st) or not t.st[j]: continue
            n += 1; hit += max(tc.iou(t.st[j]["box"], b) for b in bl) >= 0.5
        agree[(v, k)] = (hit, n)
    fl = {}
    for v, recs in byv.items():
        for r in recs:
            bad = []
            for role in ("target", "target2", "prior_obj"):
                k = MV.Video.key(r.get(role))
                if not k or (v, k) not in agree: continue
                h, n = agree[(v, k)]
                if n >= 2 and h == 0 and (v, k) not in SA: bad.append(k)
            if bad: fl[int(r["id"])] = (v, sorted(set(bad)))
    return fl

_J = {}
def track(ver, v, k):
    p = f"{R}/geo/out/tracks_{split}_{ver}/{v}.json"
    if p not in _J: _J[p] = json.load(open(p)) if os.path.exists(p) else None
    j = _J[p]
    if j is None: return None
    q = {q["key"]: q for q in (j.get("queries") or [])}.get(k)
    if q is None or q.get("obj") is None: return None
    t = j["objs"].get(str(q["obj"]))
    if not t: return None
    bx = tuple(tuple(np.round(e["st"]["box"], 3)) if isinstance(e, dict) and e.get("st") else None for e in t)
    return (tuple(j.get("fidx") or []), bx)

S = pd.read_csv(f"{R}/submissions/sub_xt_Nzc_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
src = [S.answer_source.astype(str).str.replace("mo_guard:", "", regex=False)]
for nm in ("xt_Nzc", "xt_Nzcu"):
    src.append(pd.read_csv(f"{R}/submissions/{nm}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0").answer_source.astype(str).reindex(S.index))
badsrc = pd.concat([s.str.startswith(BADSRC) for s in src], axis=1).any(axis=1)

fl2 = flags("v2", False)
allrows = []
for ch, trk, mv in (("", "v2sc", "v3ntmv"), ("u", "v2scu", "v3ntmvu")):
    flq = flags(trk, True)
    q2 = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2{ch}srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    q3 = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ3{ch}srngsr_d_vw_raw.csv").set_index("Unnamed: 0").reindex(q2.index)
    g2, g3 = pd.to_numeric(q2.geo_value, errors="coerce"), pd.to_numeric(q3.geo_value, errors="coerce")
    diff = ~np.isclose(g2.fillna(-1), g3.fillna(-1), rtol=1e-9) | (q2.meas_mode.astype(str) != q3.meas_mode.astype(str)) \
        | (q2.merge_src.astype(str) != q3.merge_src.astype(str)) | (q2.answer_source.astype(str) != q3.answer_source.astype(str))
    f2 = pd.Series(q2.index.isin(list(fl2)), index=q2.index); f3 = pd.Series(q2.index.isin(list(flq)), index=q2.index)
    x2, x3 = q2.merge_src.astype(str) == "xg_v", q3.merge_src.astype(str) == "xg_v"
    # sanity: xg_v rows are flagged rows
    print(f"[{split}{ch}] flags v2 {len(fl2)} {trk} {len(flq)}; xg_v Q2 {int(x2.sum())} (not flagged {int((x2 & ~f2).sum())}), Q3 {int(x3.sum())} (not flagged {int((x3 & ~f3).sum())})")
    print(f"[{split}{ch}] raw rows differing Q2 vs Q3: {int(diff.sum())}; with flag change {int((diff & (f2 != f3)).sum())}; without {int((diff & (f2 == f3)).sum())}")
    ruler = q2.merge_src.astype(str).isin(RULER) | q3.merge_src.astype(str).isin(RULER)
    regr, why3 = pd.Series(False, index=q2.index), pd.Series("", index=q2.index)
    for i in q2.index[diff & f3 & ~f2]:
        v, keys = flq[int(i)]
        for k in keys:
            a, b = track(trk, v, k), track(mv, v, k)
            if a is not None and a == b:
                regr[i] = True; why3[i] += f"{k}:{trk}=={mv} "
    take = diff & (f2 != f3) & ~ruler & ~regr & ~badsrc
    o = q2.copy()
    o.loc[take] = q3.loc[take]
    o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
    o.to_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ3n{ch}srngsr_d_vw_raw.csv")
    o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ3n{ch}srngsr_d_vw.csv", index=False)
    Z = pd.DataFrame(dict(ch=ch or "m", vid=q2.video_id, q=q2.question.str[:48], g2=g2, m2=q2.meas_mode, s2=q2.merge_src, g3=g3, m3=q3.meas_mode,
                          s3=q3.merge_src, f2=f2, f3=f3, ruler=ruler, regr=regr, badsrc=badsrc.reindex(q2.index), take=take, why3=why3,
                          nzc_src=src[0].reindex(q2.index)))[diff]
    print(f"[{split}{ch}] take {int(take.sum())} | excluded: ruler {int((diff & ruler).sum())}, regrounded-new-flag {int((diff & regr).sum())}, "
          f"lab/depth_bound/patch {int((diff & badsrc).sum())}, no flag change {int((diff & (f2 == f3)).sum())}")
    print(Z.round(4).to_string())
    allrows.append(Z)
pd.concat(allrows).to_csv(f"{R}/cache/tmp/ag_stale2_rows_{split}.csv")
