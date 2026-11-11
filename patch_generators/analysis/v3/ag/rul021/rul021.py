# rul021: dist2 row whose e-config target track collapsed onto target2 (e.g. "walking person" resolved to the rider on the
# "bike", so geo measures ~0 m), while the d-config track of the SAME target key is a distinct object (IoU < 0.05 with both)
# that is present in >= half the tracked frames.  The question names two different things (different head nouns), so a
# ~zero distance from two overlapping tracks is a grounding failure; the d-config object is taken as the target.
# value = |centre(d-target box) - centre(target2 box)| at the tracked frame nearest t (CSV fps) x the row's 2D meas_mpp
# (centre convention, no surface-gap exception).  d-target missing within +-2 frames of t -> CaW.
# Final answer follows the pipeline fusion: within 2x of CaW -> geometric mean (agree_gm), else the measurement (2d_ratio).
import json, sys, os, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
BUILD = "sub_xt_Nzo"
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
MODES_2D = ("2d_ratio", "sruler")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def boxes(T, o):
    """fidx -> box for obj o"""
    L = T["objs"].get(str(o)) or []
    return {T["fidx"][k]: np.array(s["st"]["box"], float) for k, s in enumerate(L) if s and s.get("st")}


def med_iou(A, B):
    c = sorted(set(A) & set(B))
    return (float(np.median([iou(A[k], B[k]) for k in c])) if c else 0.0), len(c)


def px_speed(A):
    f = np.array(sorted(A), float)
    if len(f) < 3:
        return np.nan
    c = np.array([(A[k][:2] + A[k][2:]) / 2 for k in sorted(A)])
    return float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(f)))


def head(ph):
    w = re.findall(r"[a-z]+", (ph or "").lower())
    if not w:
        return ""
    h = w[-1]
    return h[:-1] if len(h) > 3 and h.endswith("s") and not h.endswith("ss") else h


def obj_of(qs, key):
    for x in qs or []:
        if x["key"] == key:
            return x.get("obj")
    return None


def patched_rows(split):
    """rows already patched in the build (tags of the --patch_json list in sr74.sh = the sub_xt_Nzo build)"""
    s = open(f"{R}/cache/tmp/sr74.sh").read()
    m = re.search(r"--patch_json ([\w,]+)", s)
    tags = m.group(1).split(",") if m else []
    rows = {}
    for t in tags:
        f = f"{R}/cache/tmp/ag_{t}_patch_{split}.json"
        if os.path.exists(f):
            for k in json.load(open(f)):
                rows.setdefault(str(k), []).append(t)
    return rows, tags


def run(split, verbose=True):
    D = pd.read_csv(f"{R}/submissions/{BUILD}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        P[j["id"]] = j
    PR, tags = patched_rows(split)
    out, cnt, near = {}, dict(dist2=0, g1=0, g2=0, g3=0, g4=0), []
    modes = {}
    for i, d in D.iterrows():
        p = P.get(i) or {}
        q = p.get("q") or {}
        if q.get("kind") != "dist2":
            continue
        cnt["dist2"] += 1
        ph1 = ((p.get("target") or {}).get("phrase") or "").lower()
        ph2 = ((p.get("target2") or {}).get("phrase") or "").lower()
        mode = str(raw.meas_mode.get(i))
        modes[mode] = modes.get(mode, 0) + 1
        mpp = pd.to_numeric(raw.meas_mpp.get(i), errors="coerce")
        if not ph1 or not ph2 or head(ph1) == head(ph2) or q.get("t") is None:
            continue
        if mode not in MODES_2D or not (np.isfinite(mpp) and mpp > 0):
            continue
        cnt["g1"] += 1
        f = f"{R}/geo/out/tracks_{split}_v2sc/{d.video_id}.json"
        if not os.path.exists(f):
            continue
        T = json.load(open(f))

        def key_for(tp):                                   # parse phrase -> query key ('phrase|where')
            ph = ((p.get(tp) or {}).get("phrase") or "")
            wh = (p.get(tp) or {}).get("where") or []
            k = ph + "|" + ",".join(wh)
            ks = [x["key"] for x in T.get("queries") or []]
            if k in ks:
                return k
            c = [x for x in ks if x.split("|")[0] == ph]
            return c[0] if len(c) == 1 else None
        k1, k2 = key_for("target"), key_for("target2")
        if k1 is None or k2 is None:
            continue
        oe1, oe2 = obj_of(T.get("queries"), k1), obj_of(T.get("queries"), k2)
        od1, od2 = obj_of(T.get("queries_d"), k1), obj_of(T.get("queries_d"), k2)
        if oe1 is None or oe2 is None or od1 is None:
            continue
        E1, E2, D1 = boxes(T, oe1), boxes(T, oe2), boxes(T, od1)
        D2 = boxes(T, od2) if od2 is not None else {}
        ie12, nc = med_iou(E1, E2)
        rec = dict(id=i, vid=d.video_id, q=str(raw.question[i])[:70], mode=mode, oe1=oe1, oe2=oe2, od1=od1, od2=od2,
                   ie12=round(ie12, 3), ncommon=nc)
        if oe1 == oe2 or ie12 <= 0.3 or nc < 3:
            if ie12 > 0.2 or oe1 == oe2:
                near.append(rec)
            continue
        cnt["g2"] += 1
        sp1, sp2 = px_speed(E1), px_speed(E2)
        idE, _ = med_iou(D1, E1)
        idT, _ = med_iou(D1, E2)
        idT2 = med_iou(D1, D2)[0] if D2 else 0.0
        pres = len(D1) / max(len(T["fidx"]), 1)
        rec.update(sp_ratio=round(sp1 / sp2, 3) if sp2 > 0 else None, idE=round(idE, 3), idT=round(idT, 3),
                   idT2=round(idT2, 3), pres=round(pres, 2))
        if od1 in (oe1, oe2, od2) or idE >= 0.05 or idT >= 0.05 or idT2 >= 0.05 or pres < 0.5:
            near.append(rec)
            continue
        cnt["g3"] += 1
        if str(i) in PR:
            rec["patched_by"] = PR[str(i)]
            near.append(rec)
            continue
        cnt["g4"] += 1
        fps = float(raw.fps[i])                                  # CSV fps (never the container fps)
        ft = float(q["t"]) * fps
        B2 = D2 if D2 else E2
        both = [k for k in D1 if k in B2]
        unit = q.get("unit") or "m"
        cs = pd.to_numeric(d.cs, errors="coerce")
        meas, dpx = np.nan, np.nan
        if not both or min(abs(k - ft) for k in both) > 2:
            if pd.notna(cs) and cs > 0:
                val, how = float(cs), "caw (d-target missing near t)"
            else:
                near.append(rec)
                continue
        else:
            fk = min(both, key=lambda k: abs(k - ft))
            c1 = (D1[fk][:2] + D1[fk][2:]) / 2
            c2 = (B2[fk][:2] + B2[fk][2:]) / 2
            dpx = float(np.linalg.norm(c1 - c2))
            meas = dpx * mpp / SI.get(unit, 1.0)
            if pd.notna(cs) and cs > 0 and max(meas / cs, cs / meas) <= 2:
                val, how = float(np.sqrt(meas * cs)), "agree_gm"
            else:
                val, how = meas, "dis_geo"
            rec.update(frame=fk, ft=ft)
        rec.update(old=d.answer, old_src=d.answer_source, g=d.g, cs=cs, meas=round(meas, 4) if np.isfinite(meas) else None,
                   dpx=round(dpx, 1) if np.isfinite(dpx) else None, new=round(val, 4), how=how)
        out[str(i)] = dict(value=round(val, 4), why=(
            f"rul021: e-target collapsed onto target2; d-config target distinct ({how}; e obj {oe1}~{oe2} IoU {ie12:.2f}, "
            f"d obj {od1} IoU {max(idE, idT, idT2):.2f} pres {pres:.2f}; {rec.get('dpx')}px x {mpp:.5f}m/px = "
            f"{rec.get('meas')} {unit}, CaW {cs})"))
        if verbose:
            print("CHANGE", split, rec)
    if verbose:
        for r in near:
            print("near", split, r)
        print(split, "dist2 meas_mode counts", modes)
    json.dump(out, open(f"{R}/cache/tmp/ag_c22altd_patch_{split}.json", "w"), indent=1)
    print(split, "gate counts", cnt, "changed", len(out), "| patch tags checked", len(tags))
    return out


if __name__ == "__main__":
    for s in ("test", "val"):
        run(s)
