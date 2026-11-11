# rul094: the prior object P is asked as a MOVING target in the same video, the row for that motion question was
# flagged static (meas_why 'static'), i.e. the track geo used for P is a static look-alike.  If another track config
# (v3nt / v2sc, e- or d-config) holds a P track that is a different object (median IoU with the used P box < 0.1) and moves
# (excursion >= 1.5 own sizes), the prior ruler px is re-read from that moving track (median prior-dimension extent over
# frames whose box does not touch the frame border).  Every un-patched 2d_ratio SIZE row of the video that used the old
# ruler (meas_mpp) is rescaled by px_old/px_new; then re-fused like the pipeline (within 2x of CaW -> geometric mean, else geo).
import json, os, re, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C53_SUB", "sub_xt_Nzz7")
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRI = re.compile(r"\s*(?:the )?(?:total )?(diameter|length|width|height|size) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*(m|cm|mm)\s*$")
MOV = ("speed", "avgspeed", "accel")

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)

def used_boxes(split, v, ph, tracks):
    """boxes of phrase ph in the track set geo used (v1 = tracks_<split>, det boxes per det frame; v2 = v2/v2sc objs)."""
    if tracks == "v1":
        p = f"{R}/geo/out/tracks_{split}/{v}.json"
        if not os.path.exists(p): return []
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q["key"].split("|")[0] == ph:
                return [b for b in (q.get("det_boxes") or []) if b]
        return []
    for d in ("v2", "v2sc"):
        p = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
        if not os.path.exists(p): continue
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q["key"].split("|")[0] == ph and q.get("obj") is not None:
                return [s["st"]["box"] for s in T["objs"].get(str(q["obj"])) or [] if s and s.get("st")]
    return []

def alt_tracks(split, v, ph):
    out = []
    for d in ("v3nt", "v2sc"):
        p = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
        if not os.path.exists(p): continue
        T = json.load(open(p)); W, H = T.get("W") or 1e9, T.get("H") or 1e9
        for key in ("queries", "queries_d"):
            for q in T.get(key) or []:
                if not isinstance(q, dict) or q["key"].split("|")[0] != ph or q.get("obj") is None: continue
                b = np.array([s["st"]["box"] for s in T["objs"].get(str(q["obj"])) or [] if s and s.get("st")], float).reshape(-1, 4)
                if len(b) < 5: continue
                inb = (b[:, 0] > 2) & (b[:, 1] > 2) & (b[:, 2] < W - 2) & (b[:, 3] < H - 2)
                if inb.sum() < 3: continue
                c = (b[:, :2] + b[:, 2:]) / 2
                sz = float(np.median(np.maximum(b[inb, 2]-b[inb, 0], b[inb, 3]-b[inb, 1])))
                ex = float(np.max(np.linalg.norm(c - np.median(c, 0), axis=1)) / max(sz, 1e-9))
                out.append(dict(dir=d, cfg=key, obj=q["obj"], b=b, inb=inb, ex=ex))
    return out

def dim_px(b, dim):
    w, h = b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]
    if dim == "height": return float(np.median(h))
    if dim == "width": return float(np.median(w))
    return float(np.median(np.maximum(w, h)))

def run(split):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out = {}
    for v, g in raw.groupby("video_id"):
        statph = set()
        for i in g.index:
            q = (P.get(i) or {}).get("q") or {}
            ph = (((P.get(i) or {}).get("target") or {}).get("phrase") or "").lower()
            if q.get("kind") in MOV and ph and re.search(r"static", str(raw.meas_why[i])): statph.add(ph)
        if not statph: continue
        for i in g.index:
            p = P.get(i) or {}; q = p.get("q") or {}
            m = PRI.match(str(raw.ground_truth_prior[i]).lower())
            po = p.get("prior_obj") or {}
            pph = (po.get("phrase") or "").lower() if isinstance(po, dict) else ""
            if not m or pph not in statph or q.get("kind") != "size": continue
            if str(D.answer_source.get(i, "")).startswith("patch"): continue
            tph = ((p.get("target") or {}).get("phrase") or "").lower()
            if tph == pph: continue                                    # self rows: rul013 / own-ruler family
            mpp = pd.to_numeric(raw.meas_mpp[i], errors="coerce"); gv = pd.to_numeric(raw.geo_value[i], errors="coerce")
            if not (pd.notna(mpp) and mpp > 0 and pd.notna(gv) and gv > 0) or str(raw.meas_mode[i]) != "2d_ratio": continue
            prior_m = float(m.group(3)) * SI[m.group(4)]; px_old = prior_m / mpp
            ub = used_boxes(split, v, pph, str(raw.meas_tracks[i]))
            if not ub: continue
            alts = [a for a in alt_tracks(split, v, pph) if a["ex"] >= 1.5]
            best = None
            for a in alts:
                io = float(np.median([max(iou(u, bb) for bb in a["b"]) for u in ub]))
                if io < 0.1:
                    best = a; break
            if best is None: continue
            px_new = dim_px(best["b"][best["inb"]], m.group(1))
            k = px_old / px_new
            if not (1.2 <= max(k, 1 / k) <= 3): continue
            unit = q.get("unit") or "m"
            gnew = gv * k                                        # raw geo is already in the question unit
            cs = pd.to_numeric(D.cs.get(i), errors="coerce")
            if pd.notna(cs) and cs > 0 and max(gnew / cs, cs / gnew) <= 2: val, how = float(np.sqrt(gnew * cs)), "agree_gm"
            else: val, how = gnew, "dis_geo"
            out[str(i)] = dict(value=round(val, 4), why=f"rul094 {how}: prior '{pph}' used track static look-alike ({px_old:.0f}px); "
                               f"moving {best['dir']}/{best['cfg']} obj {best['obj']} ({px_new:.0f}px, exc {best['ex']:.1f}); geo {gv:.3g}->{gnew:.3g}, CaW {cs}")
            print(split, i, v, D.loc[i, "cat"], "|", raw.question[i][:60], "| old", round(D.answer[i], 3), D.answer_source[i], "->", out[str(i)])
    json.dump(out, open(f"{R}/cache/tmp/ag_c53pmove_patch_{split}.json", "w"), indent=1)
    print(split, "changed", len(out))

if __name__ == "__main__":
    for s in ("test", "val"): run(s)
