"""Rule 3 (rul300): "horizontal distance" between two objects = their distance in the ground (XY) plane, i.e. in a top-down
view the plain image-plane Euclidean distance between the two object centres (sim ground truth is object positions).

The dataset's own wording fixes the convention: val 144 asks "the horizontal distance(XY-plane) between the big bird and the
small bird". When the same recording has the same question without "horizontal" in another variant (plain vs segmented copy),
"horizontal" does not select an image axis: both rows ask the planar centre distance. Those twin groups are the trigger.

What the pipeline does instead on these rows:
  - fix f replaces the centre distance by the polygon surface gap (gap < 0.5 x centre distance, or any "smallest" question),
    and for a question without a time it takes the median over frames, not the minimum;
  - a 2-decade sanity rule then rejects the surface gap and the row falls back to CaW;
  - in the plain copy "cars|" and "cars|#second" resolve to the same track (target_px_fail -> CaW).

Measurement (label-free, inputs only): centroids st.c of the two tracks, the video's prior-derived scale (median meas_mpp of the
video's 2d_ratio rows), the dataset fps column. At a time t: the centre distance at the nearest frame (3-frame median).
"smallest/minimum/closest/shortest": the minimum over frames of the 3-frame running median, over frames where neither box
touches the border. Collapsed identity: the other track, only when the video has exactly two tracked objects.
Fusion: the pipeline's recipe (agreement within 2x -> geo for dist2 (geo_kinds); disagreement in 2d_ratio -> geo).
Writes cache/tmp/ag_hdist_patch_{test,val}.json."""
import json, re, sys, math
import numpy as np, pandas as pd

import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3")
from variant_keys import split_variant

RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
DET = "sub_xt_Nzh"
TRK = "_v2sc"
MINRX = re.compile(r"\b(?:minimum|smallest|closest|shortest|least)\b")
HRX = re.compile(r"\bhorizontal\b")


def norm_q(q):
    q = q.lower().strip()
    q = HRX.sub(" ", q)
    q = re.sub(r"\(.*?\)", " ", q)
    q = re.sub(r"(\d+(?:\.\d+)?)\s*s\b", lambda m: f"{float(m.group(1)):g}s", q)
    q = re.sub(r"[^a-z0-9.]+", " ", q)
    return " ".join(q.split())


def centres(T, o):
    out = []
    for s in T["objs"][str(o)]:
        st = (s or {}).get("st")
        out.append(None if not st else (np.array(st["c"], float), int(st.get("border", 0) or 0)))
    return out


def run(split):
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{DET}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    rows = {p["id"]: p for p in P if p["q"].get("kind") == "dist2"}
    # twin groups: same recording (base video), same question once "horizontal" is dropped, at least one row says "horizontal"
    grp = {}
    for i, p in rows.items():
        grp.setdefault((split_variant(p["video_id"])[0], norm_q(p["question"])), []).append(i)
    trig = {}
    for k, ids in grp.items():
        hz = [i for i in ids if HRX.search(rows[i]["question"].lower())]
        pl = [i for i in ids if not HRX.search(rows[i]["question"].lower())]
        if hz and pl:
            for i in ids:
                trig[i] = k
    hz_all = [i for i, p in rows.items() if HRX.search(p["question"].lower())]
    print(f"[{split}] dist2 rows {len(rows)}, 'horizontal' dist2 rows {len(hz_all)} {sorted(hz_all)}, twin-group rows {sorted(trig)}")
    out, rep = {}, []
    for i in sorted(trig):
        p = rows[i]; v = p["video_id"]
        src = str(det.answer_source.get(i, ""))
        if src.startswith("patch"):
            print(f"  {i}: already patched ({src}), skip"); continue
        T = json.load(open(f"{R}/geo/out/tracks_{split}{TRK}/{v}.json"))
        qk = {q["key"]: q.get("obj") for q in T["queries"]}
        k1 = p["target"]["phrase"] + "|" + ",".join(p["target"].get("where") or [])
        k2 = p["target2"]["phrase"] + "|" + ",".join(p["target2"].get("where") or [])
        o1, o2 = qk.get(k1), qk.get(k2)
        note = ""
        if o1 is None or o2 is None:
            print(f"  {i}: no track for {k1}/{k2}, skip"); continue
        if o1 == o2:
            objs = list(T["objs"])
            if len(objs) != 2:
                print(f"  {i}: identity collapse with {len(objs)} objects, skip"); continue
            o2 = int([o for o in objs if int(o) != int(o1)][0]); note = " (collapsed #second -> the other of 2 tracks)"
        A, B = centres(T, o1), centres(T, o2)
        fps = float(raw.loc[i, "fps"])                                   # dataset fps column (never the container's)
        vr = raw[(raw.video_id == v) & (raw.meas_mode == "2d_ratio")]
        mpp = float(pd.to_numeric(vr.meas_mpp, errors="coerce").median())
        if not np.isfinite(mpp):
            print(f"  {i}: no 2d_ratio scale in {v}, skip"); continue
        fid = T["fidx"]
        d = np.array([np.linalg.norm(a[0] - b[0]) if (a is not None and b is not None) else np.nan for a, b in zip(A, B)])
        brd = np.array([(a is None or b is None or a[1] or b[1]) for a, b in zip(A, B)])
        dm = np.array([np.nanmedian(d[max(0, j - 1): j + 2]) if np.isfinite(d[max(0, j - 1): j + 2]).any() else np.nan for j in range(len(d))])
        t = p["q"].get("t")
        if MINRX.search(p["question"].lower()) and t is None:
            ok = np.isfinite(dm) & ~brd
            if not ok.any():
                ok = np.isfinite(dm)
            j = int(np.nanargmin(np.where(ok, dm, np.nan)))
            px = float(dm[j]); how = f"min over {int(ok.sum())} unclipped frames at t={fid[j] / fps:.2f}s"
        elif t is not None:
            tf = np.array(fid) / fps
            j = int(np.argmin(np.abs(tf - t)))
            if abs(tf[j] - t) > 2.0 / fps or not np.isfinite(dm[j]):
                print(f"  {i}: no frame at t={t}, skip"); continue
            px = float(dm[j]); how = f"at t={tf[j]:.2f}s"
        else:
            print(f"  {i}: no time and no minimum, skip"); continue
        g = px * mpp
        cs = pd.to_numeric(det.cs, errors="coerce").get(i, np.nan)
        # pipeline fusion: dist2 is a geo_kind -> agreement takes geo; disagreement in the strong 2d_ratio mode -> geo
        if np.isfinite(cs) and abs(math.log10(g / cs)) <= math.log10(2):
            fused, fs = g, "agree_geo_kind"
        else:
            fused, fs = g, "dis_geo"
        why = (f"rul300: planar centre distance obj {o1}/{o2}{note} {how}: {px:.1f}px x mpp {mpp:.5f} = {g:.3f} m "
               f"(was {det.answer.get(i):.4g} {src}; cs {cs:.3g} -> {fs})")
        out[str(i)] = {"value": float(fused), "why": why}
        rep.append((i, v, p["question"], det.answer.get(i), src, fused, why))
    json.dump(out, open(f"{R}/cache/tmp/ag_hdist_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] wrote {len(out)} rows -> cache/tmp/ag_hdist_patch_{split}.json")
    for r_ in rep:
        print("   ", r_[0], r_[1], "|", r_[2], "| old", round(float(r_[3]), 4), r_[4], "| new", round(r_[5], 4))
        print("       ", r_[6])
    return out


if __name__ == "__main__":
    for sp in ("test", "val"):
        run(sp)
