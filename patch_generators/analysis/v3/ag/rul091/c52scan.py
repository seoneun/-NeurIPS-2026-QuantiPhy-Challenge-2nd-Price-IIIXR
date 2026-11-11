# c52 scan: (A) rul091 - rul021 coverage gap: dist2 target collapsed onto target2 in v2sc, distinct target track in
# another track dir (v2sc-d, v3nt, v3) and/or weak (vlm_) 2D modes; (B) rul092 - "<part> of the <parent>" target whose
# v2sc part track lies outside the parent's box while another dir's part track lies inside it.
import json, sys, os, re
import numpy as np, pandas as pd
R = "/data/iiixr/users/sehong/quantiphy"
BUILD = os.environ.get("C52_SUB", "sub_xt_Nzz6")
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
ALT = [("tracks_%s_v2sc", "queries_d"), ("tracks_%s_v3nt", "queries"), ("tracks_%s_v3nt", "queries_d"),
       ("tracks_%s_v3", "queries"), ("tracks_%s_v3", "queries_d")]
PARTW = (r"(?:outer |inner |far |near |front |back |rear |top |bottom |left |right |upper |lower )?"
         r"(?:end|tip|edge|corner|top|bottom|front|back|side|centre|center|middle|head|tail|nose)")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def contain(a, b, pad=3):
    """share of box a's area inside box b (b padded by pad px)"""
    b = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def tr_of(T, o):
    L = T["objs"].get(str(o)) or []
    return {T["fidx"][k]: s for k, s in enumerate(L) if s and s.get("st")}


def bx(S):
    return {k: np.array(v["st"]["box"], float) for k, v in S.items()}


def med(A, B, fn=iou):
    c = sorted(set(A) & set(B))
    return (float(np.median([fn(A[k], B[k]) for k in c])) if c else 0.0), len(c)


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


def poly_gap(pa, pb):
    import cv2
    best = np.inf
    for a in pa or []:
        a = np.asarray(a, float)
        for b in pb or []:
            b = np.asarray(b, float)
            if len(a) >= 3 and len(b) >= 3:
                if any(cv2.pointPolygonTest(b.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in a) or \
                   any(cv2.pointPolygonTest(a.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in b):
                    return 0.0
            for P_, S_ in ((a, b), (b, a)):
                s0, s1 = S_, np.roll(S_, -1, 0)
                d = s1 - s0
                L = np.maximum((d * d).sum(1), 1e-9)
                u = np.clip(((P_[:, None, :] - s0[None]) * d[None]).sum(-1) / L[None], 0, 1)
                proj = s0[None] + u[..., None] * d[None]
                best = min(best, float(np.linalg.norm(P_[:, None, :] - proj, axis=2).min()))
    return best


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
    if pd.notna(cs) and cs > 0 and max(meas / cs, cs / meas) <= 2:
        return float(np.sqrt(meas * cs)), "agree_gm"
    return float(meas), "dis_geo"


def run(split, verbose=True):
    D = pd.read_csv(R + "/submissions/" + BUILD + "_" + ("detail" if split == "test" else "valdetail") + ".csv").set_index("Unnamed: 0")
    raw = pd.read_csv(R + "/preds/" + split + "_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(R + "/geo/out/parse_" + split + "_v2.jsonl"):
        j = json.loads(l)
        P[j["id"]] = j
    outA, outB, log = {}, {}, []
    for i, d in D.iterrows():
        p = P.get(i) or {}
        q = p.get("q") or {}
        kind = q.get("kind")
        mode = str(raw.meas_mode.get(i))
        mpp = pd.to_numeric(raw.meas_mpp.get(i), errors="coerce")
        base_mode = mode[4:] if mode.startswith("vlm_") else mode
        if base_mode not in ("2d_ratio", "sruler") or not (np.isfinite(mpp) and mpp > 0):
            continue
        if kind != "dist2" or q.get("t") is None:
            continue
        T = load_T(split, "tracks_%s_v2sc", d.video_id)
        if T is None:
            continue
        k1, k2 = key_for(T, p, "target"), key_for(T, p, "target2")
        if k1 is None or k2 is None:
            continue
        oe1, oe2 = obj_of(T.get("queries"), k1), obj_of(T.get("queries"), k2)
        if oe1 is None or oe2 is None:
            continue
        S1, S2 = tr_of(T, oe1), tr_of(T, oe2)
        E1, E2 = bx(S1), bx(S2)
        fps = float(raw.fps[i])
        ft = float(q["t"]) * fps
        unit = q.get("unit") or "m"
        cs = pd.to_numeric(d.cs, errors="coerce")
        ph1 = ((p.get("target") or {}).get("phrase") or "").lower()
        ph2 = ((p.get("target2") or {}).get("phrase") or "").lower()
        raw1 = ((p.get("target") or {}).get("raw") or "").lower()
        minimum = bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", str(raw.question[i]).lower()))
        ie12, nc = med(E1, E2)
        # ---------------- (A) collapse
        if ph1 and ph2 and head(ph1) != head(ph2) and oe1 != oe2 and ie12 > 0.3 and nc >= 3:
            got = None
            for dname, qk in ALT:
                Ta = load_T(split, dname, d.video_id)
                if Ta is None:
                    continue
                oa = obj_of(Ta.get(qk), k1)
                if oa is None:
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
                    continue
                both = [k for k in A if k in E2]
                if not both or min(abs(k - ft) for k in both) > 2:
                    continue
                fk = min(both, key=lambda k: abs(k - ft))
                c1 = np.array(Sa[fk]["st"]["c"], float)          # mask centroids (the pipeline's dist2 convention)
                c2 = np.array(S2[fk]["st"]["c"], float)
                dpx = float(np.linalg.norm(c1 - c2))
                meas = dpx * mpp / SI.get(unit, 1.0)
                val, how = meas, "geo (strong 2D track; dist2 agree_geo_kind)"
                got = (dname % split, qk, oa, round(iA1, 3), round(iA2, 3), round(pres, 2), fk, round(dpx, 1), round(meas, 4), round(val, 4), how)
                break
            rec = dict(rule="A", id=i, vid=d.video_id, q=str(raw.question[i])[:80], mode=mode, ie12=round(ie12, 3), old=d.answer,
                       src=d.answer_source, g=d.g, cs=cs, got=got)
            log.append(rec)
            if got and (str(d.answer_source).startswith("patch:") or abs(np.log(got[9] / d.answer)) < np.log(1.15)):
                rec["skip"] = "already patched" if str(d.answer_source).startswith("patch:") else "answer already within 15%"
                got = None
            if got:
                outA[str(i)] = dict(value=got[9], why="rul091: v2sc target collapsed onto target2 (IoU %.2f); distinct target track in %s/%s obj %s "
                                    "(IoU %.2f, pres %s); %spx x %.5f = %s %s, CaW %s -> %s" % (ie12, got[0], got[1], got[2], max(got[3], got[4]), got[5],
                                                                                               got[7], mpp, got[8], unit, cs, got[10]))
        # ---------------- (B) part of parent
        mp = re.match(r"^(?:the\s+)?(" + PARTW + r")\s+of\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z \-]*?)\s*$", raw1.strip())
        if mp and ph1.strip() == mp.group(1).strip():
            parent = mp.group(2).strip()
            keys = [x["key"] for x in T.get("queries") or []]
            kp = parent + "|" if parent + "|" in keys else None
            if kp is None:
                c = [x for x in keys if x.split("|")[0] == parent]
                kp = c[0] if len(c) == 1 else None
            if kp is None:
                log.append(dict(rule="B-noparent", id=i, vid=d.video_id, q=str(raw.question[i])[:80], parent=parent))
                continue
            op = obj_of(T.get("queries"), kp)
            Pp = bx(tr_of(T, op)) if op is not None else {}
            cm, _ = med(E1, Pp, contain)
            got = None
            if cm < 0.2:
                for dname, qk in ALT[1:]:
                    Ta = load_T(split, dname, d.video_id)
                    if Ta is None:
                        continue
                    oa = obj_of(Ta.get(qk), k1)
                    opa = obj_of(Ta.get(qk), kp)
                    if oa is None or opa is None or oa == opa:
                        continue
                    Sa = tr_of(Ta, oa)
                    A = bx(Sa)
                    Pa = bx(tr_of(Ta, opa))
                    ca, _ = med(A, Pa, contain)
                    com = [k for k in A if k in Pa]
                    ar = float(np.median([(A[k][2] - A[k][0]) * (A[k][3] - A[k][1]) / max((Pa[k][2] - Pa[k][0]) * (Pa[k][3] - Pa[k][1]), 1) for k in com])) if com else 9
                    pres = len(A) / max(len(Ta["fidx"]), 1)
                    if ca < 0.8 or ar >= 0.5 or pres < 0.5:
                        continue
                    both = [k for k in Sa if k in S2]
                    if not both or min(abs(k - ft) for k in both) > 2:
                        continue
                    fk = min(both, key=lambda k: abs(k - ft))
                    c1 = np.array(Sa[fk]["st"]["c"], float)
                    c2 = np.array(S2[fk]["st"]["c"], float)
                    cen = float(np.linalg.norm(c1 - c2))
                    gap = poly_gap(Sa[fk].get("poly"), S2[fk].get("poly"))
                    use_gap = bool(np.isfinite(gap) and gap > 0 and (gap < 0.5 * cen or minimum))
                    dpx = gap if use_gap else cen
                    meas = dpx * mpp / SI.get(unit, 1.0)
                    val, how = meas, "geo (strong 2D track; dist2 agree_geo_kind)"
                    got = (dname % split, qk, oa, round(ca, 2), round(ar, 3), round(pres, 2), fk, round(cen, 1), round(gap, 1), use_gap, round(meas, 4), round(val, 4), how)
                    break
            rec = dict(rule="B", id=i, vid=d.video_id, q=str(raw.question[i])[:80], mode=mode, parent=kp, contain_main=round(cm, 3),
                       old=d.answer, src=d.answer_source, g=d.g, cs=cs, got=got)
            log.append(rec)
            if got and (str(d.answer_source).startswith("patch:") or abs(np.log(got[11] / d.answer)) < np.log(1.15)):
                rec["skip"] = "already patched" if str(d.answer_source).startswith("patch:") else "answer already within 15%"
                got = None
            if got:
                outB[str(i)] = dict(value=got[11], why="rul092: v2sc '%s' track lies outside its parent '%s' (contain %.2f); %s/%s part track inside it "
                                    "(contain %s, area ratio %s); %s %spx x %.5f = %s %s, CaW %s -> %s" % (
                                        k1, kp, cm, got[0], got[1], got[3], got[4], "gap" if got[9] else "centre", got[8] if got[9] else got[7],
                                        mpp, got[10], unit, cs, got[12]))
    return outA, outB, log


if __name__ == "__main__":
    write = len(sys.argv) > 1 and sys.argv[1] == "write"
    for s in ("test", "val"):
        A, B, log = run(s)
        for r in log:
            print(s, r)
        print(s, "A changed", len(A), "B changed", len(B))
        if write:
            json.dump(A, open(R + "/cache/tmp/ag_c52altdv_patch_" + s + ".json", "w"), indent=1)
            json.dump(B, open(R + "/cache/tmp/ag_c52partin_patch_" + s + ".json", "w"), indent=1)
