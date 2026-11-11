# rul091 - coverage extension of the accepted rul021 (dist2 target collapsed onto target2).
# rul021 looked only at strong 2D modes and only at the v2sc d-config.  This version also takes weak vlm_ 2D modes
# (VLM-track measurements) and searches more track dirs for a distinct target track:
#   v2sc queries_d -> v3nt queries -> v3nt queries_d -> v3 queries -> v3 queries_d.
# Gates: kind dist2 with a time t; meas_mode (vlm_ stripped) in {2d_ratio, sruler} with meas_mpp > 0; target / target2 have
# different head nouns; the v2sc e-track of target has median IoU > 0.3 with the target2 e-track over >= 3 common frames;
# the alternate track has median IoU < 0.05 with both v2sc tracks and with its own dir's target2 track, is a different obj,
# and is present in >= 50% of fidx.  Rows already patched, or whose new value is within 1.15x of the answer, are skipped.
# value = |c_alt(fk) - c_target2_v2sc(fk)| x meas_mpp / unit, with c the mask centroid (centre convention, as rul021 and
# rul264: simulation distances are centre-to-centre) and fk the tracked frame nearest t x CSV fps (within 2 frames).
# Fusion: dist2 agreement rows take geo alone (--geo_kinds dist2); a disagreement takes the measurement (the 2D-ratio
# scale comes from the strong prior track; the target track replaces a detector track demonstrably collapsed onto target2).
import json, sys, os, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
BUILD = os.environ.get("C52_SUB", "sub_xt_Nzz6")
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
ALT = [("tracks_%s_v2sc", "queries_d"), ("tracks_%s_v3nt", "queries"), ("tracks_%s_v3nt", "queries_d"),
       ("tracks_%s_v3", "queries"), ("tracks_%s_v3", "queries_d")]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def tr_of(T, o):
    L = T["objs"].get(str(o)) or []
    return {T["fidx"][k]: s for k, s in enumerate(L) if s and s.get("st")}


def bx(S):
    return {k: np.array(v["st"]["box"], float) for k, v in S.items()}


def med(A, B):
    c = sorted(set(A) & set(B))
    return (float(np.median([iou(A[k], B[k]) for k in c])) if c else 0.0), len(c)


def obj_of(qs, key):
    for x in qs or []:
        if x["key"] == key:
            return x.get("obj")
    return None


def head(ph):
    w = re.findall(r"[a-z]+", (ph or "").lower())
    if not w:
        return ""
    h = w[-1]
    return h[:-1] if len(h) > 3 and h.endswith("s") and not h.endswith("ss") else h


def load_T(split, d, vid):
    f = R + "/geo/out/" + (d % split) + "/" + vid + ".json"
    return json.load(open(f)) if os.path.exists(f) else None


def key_for(T, p, tp):
    ph = ((p.get(tp) or {}).get("phrase") or "")
    wh = (p.get(tp) or {}).get("where") or []
    k = ph + "|" + ",".join(wh)
    ks = [x["key"] for x in T.get("queries") or []]
    if k in ks:
        return k
    c = [x for x in ks if x.split("|")[0] == ph]
    return c[0] if len(c) == 1 else None


def fuse(meas, cs):
    """pipeline fusion for a dist2 row: agreement within 2x -> gm, then --geo_kinds dist2 makes agreement rows take geo
    alone (agree_geo_kind); disagreement -> the measurement (dis_geo)"""
    if pd.notna(cs) and cs > 0 and max(meas / cs, cs / meas) <= 2:
        return float(meas), "agree_geo_kind"
    return float(meas), "dis_geo"


def run(split):
    D = pd.read_csv(R + "/submissions/" + BUILD + "_" + ("detail" if split == "test" else "valdetail") + ".csv").set_index("Unnamed: 0")
    raw = pd.read_csv(R + "/preds/" + split + "_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(R + "/geo/out/parse_" + split + "_v2.jsonl"):
        j = json.loads(l)
        P[j["id"]] = j
    out, log, cnt = {}, [], dict(dist2_t=0, mode=0, heads=0, collapse=0, alt=0, written=0)
    for i, d in D.iterrows():
        p = P.get(i) or {}
        q = p.get("q") or {}
        if q.get("kind") != "dist2" or q.get("t") is None:
            continue
        cnt["dist2_t"] += 1
        mode = str(raw.meas_mode.get(i))
        mpp = pd.to_numeric(raw.meas_mpp.get(i), errors="coerce")
        base_mode = mode[4:] if mode.startswith("vlm_") else mode
        if base_mode not in ("2d_ratio", "sruler") or not (np.isfinite(mpp) and mpp > 0):
            continue
        cnt["mode"] += 1
        ph1 = ((p.get("target") or {}).get("phrase") or "").lower()
        ph2 = ((p.get("target2") or {}).get("phrase") or "").lower()
        if not ph1 or not ph2 or head(ph1) == head(ph2):
            continue
        cnt["heads"] += 1
        T = load_T(split, "tracks_%s_v2sc", d.video_id)
        if T is None:
            continue
        k1, k2 = key_for(T, p, "target"), key_for(T, p, "target2")
        if k1 is None or k2 is None:
            continue
        oe1, oe2 = obj_of(T.get("queries"), k1), obj_of(T.get("queries"), k2)
        if oe1 is None or oe2 is None or oe1 == oe2:
            continue
        S1, S2 = tr_of(T, oe1), tr_of(T, oe2)
        E1, E2 = bx(S1), bx(S2)
        ie12, nc = med(E1, E2)
        if ie12 <= 0.3 or nc < 3:
            continue
        cnt["collapse"] += 1
        fps = float(raw.fps[i])                       # CSV fps, never the container fps
        ft = float(q["t"]) * fps
        unit = q.get("unit") or "m"
        cs = pd.to_numeric(d.cs, errors="coerce")
        got, tried = None, []
        for dname, qk in ALT:
            Ta = load_T(split, dname, d.video_id)
            if Ta is None:
                tried.append((dname % split, qk, "no file"))
                continue
            oa = obj_of(Ta.get(qk), k1)
            if oa is None:
                tried.append((dname % split, qk, "no obj"))
                continue
            Sa = tr_of(Ta, oa)
            A = bx(Sa)
            ob = obj_of(Ta.get(qk), k2)
            Bb = bx(tr_of(Ta, ob)) if ob is not None else {}
            iA1, _ = med(A, E1)
            iA2, _ = med(A, E2)
            iAb = med(A, Bb)[0] if Bb else 0.0
            pres = len(A) / max(len(Ta["fidx"]), 1)
            if (ob is not None and ob == oa) or iA1 >= 0.05 or iA2 >= 0.05 or iAb >= 0.05 or pres < 0.5:
                tried.append((dname % split, qk, "obj %s iou %.2f/%.2f/%.2f pres %.2f" % (oa, iA1, iA2, iAb, pres)))
                continue
            both = [k for k in A if k in E2]
            if not both or min(abs(k - ft) for k in both) > 2:
                tried.append((dname % split, qk, "no frame near t"))
                continue
            fk = min(both, key=lambda k: abs(k - ft))
            c1 = np.array(Sa[fk]["st"]["c"], float)
            c2 = np.array(S2[fk]["st"]["c"], float)
            dpx = float(np.linalg.norm(c1 - c2))
            meas = dpx * mpp / SI.get(unit, 1.0)
            val, how = fuse(meas, cs)
            got = dict(dir=dname % split, qk=qk, obj=oa, iou=round(max(iA1, iA2, iAb), 3), pres=round(pres, 2), fk=fk, dpx=round(dpx, 1),
                       meas=round(meas, 4), val=round(val, 4), how=how)
            break
        rec = dict(id=i, vid=d.video_id, cat=d["cat"], q=str(raw.question[i])[:90], mode=mode, merge=raw.merge_src.get(i), ie12=round(ie12, 3),
                   old=round(float(d.answer), 4), src=d.answer_source, g=d.g, cs=cs, got=got, tried=tried)
        if got:
            cnt["alt"] += 1
            if str(d.answer_source).startswith("patch:"):
                rec["skip"] = "already patched"
            elif abs(np.log(got["val"] / d.answer)) < np.log(1.15):
                rec["skip"] = "within 1.15x"
            else:
                cnt["written"] += 1
                out[str(i)] = dict(value=got["val"], why="rul091: v2sc target track collapsed onto target2 (IoU %.2f); distinct target track %s/%s obj %s "
                                   "(IoU %.2f, pres %.2f); centre %.1fpx x %.5f = %.4f %s at frame %d, CaW %s -> %s" % (
                                       ie12, got["dir"], got["qk"], got["obj"], got["iou"], got["pres"], got["dpx"], mpp, got["meas"], unit, got["fk"], cs, got["how"]))
        log.append(rec)
    return out, log, cnt


if __name__ == "__main__":
    write = len(sys.argv) > 1 and sys.argv[1] == "write"
    for s in ("test", "val"):
        out, log, cnt = run(s)
        for r in log:
            print(s, r)
        print(s, "gate counts", cnt, "changed", len(out))
        if write:
            json.dump(out, open(R + "/cache/tmp/ag_c52altdv_patch_" + s + ".json", "w"), indent=1)
            print(s, "wrote", R + "/cache/tmp/ag_c52altdv_patch_" + s + ".json")
