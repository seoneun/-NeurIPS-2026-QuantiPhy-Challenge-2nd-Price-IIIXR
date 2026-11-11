"""disc scan (audit7 rule 4): flat-disc diameter = the ellipse's long axis.

For every row whose PRIOR is a size stated as a diameter / radius, or whose TARGET is a diameter / radius, measure the
tracks measure_v3 used (v2sc, cfg e; v3ntmv when the raw row says meas_tracks=v1):
  old  = the extent measure_v3 used (ext_px -> rr[:,1], the min-area-rect SHORT side, 75th pct);
         for a same-class prior (raw meas_grav 'sameclass_N') the median over the other same-head-noun tracks, as in
         measure_v3 QP_SAMECLASS;
  long = the min-area-rect LONG side rr[:,0], median over the slowest third of the extent frames (centroid speed),
         so motion blur along the travel direction does not count;
  asp  = median rr0 / rr1 over the extent frames; blur = median rr0 of the fastest third / of the slowest third.
Writes cache/tmp/ag_disc_scan_{split}.csv."""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
import track_common as tc
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
SPLIT = sys.argv[1]
DIA = ("diameter", "radius")


def slow_long(tr):
    """long side of the min-area rect on the slowest third of the extent frames (>= 3 frames)"""
    js = np.asarray(tr.oke)
    if len(js) == 0:
        return dict(nfr=0)
    rr = np.array([tr.st[j]["rr"][:3] for j in js], float)
    box = np.array([tr.st[j]["box"] for j in js], float)
    tj = tr.t_all[js]
    c = np.array([tr.st[j]["c"] for j in js], float)
    sp = np.full(len(js), np.nan)
    for k in range(len(js)):
        a, b = max(k - 2, 0), min(k + 2, len(js) - 1)
        if tj[b] - tj[a] > 0:
            sp[k] = float(np.linalg.norm(c[b] - c[a]) / (tj[b] - tj[a]))
    o = np.argsort(np.where(np.isfinite(sp), sp, np.inf))
    n3 = max(3, len(js) // 3)
    slow, fast = o[:n3], o[-n3:]
    d = dict(nfr=len(js), long=float(np.median(rr[slow, 0])), long_all=float(np.median(rr[:, 0])),
             long_p75=float(np.percentile(rr[:, 0], 75)), short_p75=float(np.percentile(rr[:, 1], 75)),
             asp=float(np.median(rr[:, 0] / np.maximum(rr[:, 1], 1.0))),
             blur=float(np.median(rr[fast, 0]) / max(np.median(rr[slow, 0]), 1.0)),
             spd_slow=float(np.nanmedian(sp[slow])) if np.isfinite(sp[slow]).any() else np.nan,
             spd_fast=float(np.nanmedian(sp[fast])) if np.isfinite(sp[fast]).any() else np.nan,
             ang=float(np.median(rr[:, 2])), bw=float(np.median(box[:, 2] - box[:, 0])), bh=float(np.median(box[:, 3] - box[:, 1])),
             clip=bool(getattr(tr, "clipped_extent", False)), key=tr.q.get("key"), obj=tr.q.get("obj"))
    return d


def old_ext(tr, dim, phrase):
    e, js = tr.ext_px(dim, phrase)
    return MV.agg_ext(e, dim) if len(e) else np.nan


raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
MV.TRACK_VER = "v2sc"
byv, vjs = MV.load_split_tracks(SPLIT, "e")
MV.TRACK_VER = "v3ntmv"
_, vjs3 = MV.load_split_tracks(SPLIT, "e")
rows = []
for v, recs in byv.items():
    cand = []
    for r in recs:
        q = r.get("q") or {}
        pr = (r.get("priors") or [{}])[0]
        tdia = q.get("kind") == "size" and q.get("dim") in DIA
        pdia = pr.get("kind") == "size" and pr.get("dim") in DIA and not pr.get("gravity")
        if tdia or pdia:
            cand.append((r, tdia, pdia))
    if not cand:
        continue
    MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
    VV = {}
    for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
        if v in src_:
            VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
    for r, tdia, pdia in cand:
        q = r["q"]; pr = r["priors"][0]
        d = dict(id=r["id"], vid=v, cat=r["cat"], qk=q["kind"], qdim=q.get("dim"), tdia=tdia, pdia=pdia, praw=pr.get("raw"), pdim=pr.get("dim"),
                 vsi=pr.get("value_si"), tgt=(r.get("target") or {}).get("phrase"), pob=(r.get("prior_obj") or {}).get("phrase"), question=r["question"])
        if r["id"] not in raw.index:
            d["note"] = "no_raw"; rows.append(d); continue
        rr_ = raw.loc[r["id"]]
        d.update(mode=str(rr_.meas_mode), mtr=rr_.meas_tracks, mpp=rr_.meas_mpp, geo=rr_.geo_value, why=rr_.meas_why, grav=str(rr_.meas_grav), msrc=rr_.merge_src)
        mt_ = rr_.meas_tracks if rr_.meas_tracks in VV else "v2"
        if mt_ not in VV:
            d["note"] = "no_track_file"; rows.append(d); continue
        vj, V = VV[mt_]
        if tdia and (r.get("target") or {}).get("phrase"):
            tr = V.get(r["target"])
            if tr is not None and tr.n:
                d["t_old"] = old_ext(tr, q["dim"], r["target"]["phrase"])
                d.update({"t_" + k: x for k, x in slow_long(tr).items()})
        if pdia and (r.get("prior_obj") or {}).get("phrase"):
            ptr = V.get(r["prior_obj"])
            if ptr is not None and ptr.n:
                d["p_old"] = old_ext(ptr, pr["dim"], r["prior_obj"]["phrase"])
                d.update({"p_" + k: x for k, x in slow_long(ptr).items()})
            # same-class alternatives exactly as measure_v3 QP_SAMECLASS enumerates them
            hn_ = tc.head_noun(r["prior_obj"].get("phrase"))
            alts = []
            for trk_ in {id(x_): x_ for x_ in V.tr.values() if x_ is not None}.values():
                if trk_ is ptr or not trk_.n or tc.head_noun(trk_.q.get("phrase")) != hn_ or not MV.track_ok(trk_, False)[0]:
                    continue
                a_ = old_ext(trk_, pr["dim"], trk_.q.get("phrase"))
                if a_ is not None and np.isfinite(a_) and a_ > 0:
                    sl = slow_long(trk_)
                    alts.append((a_, sl, trk_.q.get("key")))
            d["sc_n"] = len(alts)
            if alts:
                d["sc_old"] = float(np.median([a[0] for a in alts]))
                d["sc_long"] = float(np.median([a[1]["long"] for a in alts]))
                d["sc_asp"] = float(np.median([a[1]["asp"] for a in alts]))
                d["sc_blur"] = float(np.median([a[1]["blur"] for a in alts]))
                d["sc_clip"] = bool(any(a[1]["clip"] for a in alts))
                d["sc_nfr"] = int(min(a[1]["nfr"] for a in alts))
                d["sc_keys"] = ";".join(str(a[2]) for a in alts)
        # the extent measure_v3 actually used for the prior (2D): value_si / mpp
        if pdia and np.isfinite(rr_.meas_mpp) and rr_.meas_mpp > 0:
            d["ppx_used"] = float(pr["value_si"] / rr_.meas_mpp)
        rows.append(d)
D = pd.DataFrame(rows).set_index("id")
D.to_csv(f"{R}/cache/tmp/ag_disc_scan_{SPLIT}.csv")
print(SPLIT, "rows", len(D), "tdia", int(D.tdia.sum()), "pdia", int(D.pdia.sum()), "videos", D.vid.nunique())
