"""rul110: coverage gap of rul435 for rows the det-track IoU veto removed before measurement, in a video whose
prior is a painted layout element measured by the scene ruler (QP_SCENE_RULER, merge_src 'sruler' on sibling rows).

measure_v3.track_ok() vetoes a target whose median IoU between the detector's per-frame top box and the SAM2 track box is
< 0.5. For a tiny object (< 25 px) that IoU measures the detector (it jumps to distractors, and a few px of jitter halve
the IoU of a 9 px box), not the track; the accepted rul435 rule lifts the veto when the track itself is long, covers the
clip and moves continuously - but only for rows that still got a weak geo value. In a scene-ruler video the veto fires
before the ruler branch, so the row has no geo at all and falls to CaW (caw_nogeo). The ruler (metres per pixel) is
a property of the video (painted court/lane element / frame width), already accepted on the sibling rows, so the row is
measured exactly as the ruler branch would have done (dist2: centroid distance at t; size: measure.agg_ext of the
extent) times the siblings' mpp, and routed as sruler_geo (geo alone, like every other scene-ruler row).

Gates: (a) raw meas_why 'target_det_track_iou(x)', geo NaN; (b) kind dist2 / size; (c) >= 1 sibling row of the video with
merge_src 'sruler' and finite meas_mpp, all within 2 %, median taken; (d) the vetoed target track passes rul435's
track_gates (< 25 px, n >= 10, coverage >= 0.5, step ratio <= 1); for dist2 target2's track is not vetoed itself
(det IoU >= 0.5) and passes the same n / coverage / step gates; (e) dist2: both tracks have a usable box within one
frame of round(t * fps) (or the start / end frame for when=start/end).
usage: python rule.py [sub=sub_xt_Nzz13] [geo=geo_v4ntrj5sQ2srngsr_d_vw]
"""
import sys, os, re, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/analysis/v3/ag/rul435")
import measure_v3 as MV
import measure as M1
from rul435 import track_gates
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz13"
GEO = sys.argv[2] if len(sys.argv) > 2 else "geo_v4ntrj5sQ2srngsr_d_vw"
MV.FIX = set("abcdef")
WHY = "rul110: IoU-vetoed tiny track measured with the video's scene ruler"


def load_parse(split):
    out = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        out[int(j["id"])] = j
    return out


def load_tracks(split, vid):
    for p in (f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json"):
        if os.path.exists(p):
            return json.load(open(p))
    return None


def qtrack(vj, cfg, ent):
    k = MV.Video.key(ent)
    for q in vj["queries" if cfg == "e" else "queries_d"]:
        if q["key"] == k:
            return q
    return None


def frame_ok(tr, vj, t_s):
    """a usable box within one frame of t_s"""
    f0 = round(t_s * vj["fps"])
    fidx = np.array(vj["fidx"])[tr.okc]
    return bool(len(fidx)) and np.min(np.abs(fidx - f0)) <= 1


def measure_row(r, P, vj, mpp):
    """-> (value_si, info) or (None, fail)"""
    q = P["q"]
    kind = q["kind"]
    iou_s = re.match(r"target_det_track_iou\(([0-9.]+)\)", r.meas_why).group(1)
    for cfg in ("e", "d"):
        qt = qtrack(vj, cfg, P["target"])
        if qt is None or qt.get("obj") is None:
            continue
        iou = float(np.median(qt["det_iou"])) if qt.get("det_iou") else 0.0
        if not (iou < 0.5 and f"{iou:.2f}" == iou_s):
            continue
        ok, gi = track_gates(vj, qt)
        info = dict(cfg=cfg, key=qt["key"], obj=qt["obj"], iou=round(iou, 3),
                    **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in gi.items()})
        if not ok:
            return None, f"target_track_gate {info}"
        V = MV.Video(vj, cfg)
        tgt = V.get(P["target"])
        if kind == "size":
            e, _ = tgt.ext_px(q["dim"], P["target"]["phrase"])
            v = MV.agg_ext(e, q["dim"])
            info["px"] = round(float(v), 2)
        else:
            q2 = qtrack(vj, cfg, P["target2"]) if P.get("target2") else None
            if q2 is None or q2.get("obj") is None:
                return None, f"no_target2 {info}"
            iou2 = float(np.median(q2["det_iou"])) if q2.get("det_iou") else 0.0
            ok2, g2 = track_gates(vj, q2)
            info.update(key2=q2["key"], obj2=q2["obj"], iou2=round(iou2, 3), n2=g2["n"], cov2=round(g2["cov"], 2),
                        sr2=round(g2["step_ratio"], 2), px2=round(g2["px"], 1))
            if q2["obj"] == qt["obj"]:
                return None, f"same_obj {info}"
            if iou2 < 0.5 or not (g2["n"] >= 10 and g2["cov"] >= 0.5 and g2["step_ratio"] <= 1.0):
                return None, f"target2_gate {info}"
            tgt2 = V.get(P["target2"])
            t_s, when = q["t"], q["when"]
            if t_s is None and when is None:
                return None, f"no_time {info}"
            if when == "start":
                t_s = max(tgt.t.min(), tgt2.t.min())
            elif when == "end":
                t_s = min(tgt.t.max(), tgt2.t.max())
            if not (frame_ok(tgt, vj, t_s) and frame_ok(tgt2, vj, t_s)):
                return None, f"no_box_at_t {info}"
            v = M1.dist2_px(tgt, tgt2, q["t"], q["when"])
            info["px"] = None if v is None else round(float(v), 2)
        if v is None or not np.isfinite(v) or v <= 0:
            return None, f"px_fail {info}"
        return float(v) * mpp, info
    return None, "no_vetoed_track_matching_why"


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    PP = load_parse(split)
    patch, rep = {}, []
    why = raw.meas_why.fillna("").astype(str)
    mppcol = pd.to_numeric(raw.meas_mpp, errors="coerce")
    cand = raw[why.str.startswith("target_det_track_iou") & pd.to_numeric(raw.geo_value, errors="coerce").isna()]
    for i, r in cand.iterrows():
        P = PP.get(int(i))
        kind = P["q"]["kind"] if P else None
        sib = raw[(raw.video_id == r.video_id) & (raw.merge_src == "sruler") & np.isfinite(mppcol)]
        if not len(sib):
            continue                                   # (c) no scene-ruler sibling: not this rule's population
        rec = dict(row=int(i), vid=r.video_id, kind=kind, q=str(r.question)[:70], src=det.answer_source.get(i),
                   ans=det.answer.get(i), cs=det.cs.get(i), nsib=len(sib))
        mp = mppcol[sib.index].astype(float)
        if kind not in ("dist2", "size"):
            rec["fail"] = f"kind={kind}"; rep.append(rec); continue
        if mp.max() / mp.min() > 1.02:
            rec["fail"] = "mpp_disagree"; rep.append(rec); continue
        mpp = float(mp.median())
        vj = load_tracks(split, r.video_id)
        if vj is None or vj.get("version") != "v2":
            rec["fail"] = "no_v2_tracks"; rep.append(rec); continue
        v, info = measure_row(r, P, vj, mpp)
        rec.update(mpp=mpp, info=info)
        if v is None:
            rec["fail"] = info; rep.append(rec); continue
        rec["new"] = v; rec["fail"] = None
        patch[str(int(i))] = dict(value=v, why=f"{WHY} ({kind}; {info.get('px')} px x mpp {mpp:.4g}; {r.meas_why}; sruler_geo)")
        rep.append(rec)
    return patch, rep, raw, PP


if __name__ == "__main__":
    for split in ("test", "val"):
        p, rep, raw, PP = run(split)
        json.dump(p, open(f"{R}/cache/tmp/ag_c63tinyiou_sr_patch_{split}.json", "w"), indent=1)
        print(f"== {split}: {len(rep)} candidate rows with a scene-ruler sibling, {len(p)} patched")
        for x in rep:
            print("  ", x)
        # sanity: re-measure the sibling sruler dist2 rows of candidate videos with the same code path
        for vid in sorted({x["vid"] for x in rep}):
            vj = load_tracks(split, vid)
            for i, s in raw[(raw.video_id == vid) & (raw.merge_src == "sruler")].iterrows():
                P = PP[int(i)]; q = P["q"]
                if q["kind"] != "dist2" or not P.get("target2"):
                    continue
                for cfg in ("e", "d"):
                    V = MV.Video(vj, cfg); a, b = V.get(P["target"]), V.get(P["target2"])
                    d = M1.dist2_px(a, b, q["t"], q["when"]) if a is not None and b is not None else None
                    print(f"   sibling {i} cfg {cfg}: raw geo {s.geo_value:.4f}  recomputed {None if d is None else round(d * s.meas_mpp, 4)}")
