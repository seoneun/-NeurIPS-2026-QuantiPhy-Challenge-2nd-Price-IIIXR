"""rul078: build of the audit-22 runner-up c22fgub (foreground bottom-cut target measured at the prior's depth).

Mechanism: with tdepth=prior, geo gives a size row's target the prior object's depth. A target whose box touches the
bottom image border in most frames while the prior's box ends well above it (median bottom <= 0.7 H) stands nearer
to the camera than the prior on the ground plane, so the prior's depth over-states its size: geo is an UPPER bound.
Mirror of the accepted rul011 (target occluded by the prior -> geo is a lower bound).

Code conditions only (no ids), audit-22 spec:
  - S3/D3 row, parse q.kind == 'size'; geo meas_mode contains 'tdepth=prior' and not '_scene';
  - target box bottom at the border (y2 >= H-3) in >= 50% of its tracked frames; prior median bottom <= 0.7 H;
  - target phrase has no surface noun (lane, road, table, platform, floor, ground, street, track, field, court, pool, wall);
  - answer_source in {dis_size_gm, agree_gm, agree_wgm, dis_geo}, or var_ans_plain whose plain-video twin row (same
    question) has one of those sources; no 'patch:' source;
  - output: CaW if CaW < geo, else min(answer, geo); only decreases are written.
Tracks: measure_v3's chain (v2sc -> v2 -> v1).
Writes cache/tmp/ag_c47fgub_patch_{test,val}.json"""
import json, os, sys, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = "rul078"
SURF = re.compile(r"\b(lane|road|table|platform|floor|ground|street|track|field|court|pool|wall)s?\b", re.I)
OKSRC = {"dis_size_gm", "agree_gm", "agree_wgm", "dis_geo"}


def load_tracks(split, vid):
    for d in (f"tracks_{split}_v2sc", f"tracks_{split}_v2", f"tracks_{split}"):
        p = f"{R}/geo/out/{d}/{vid}.json"
        if os.path.exists(p):
            T = json.load(open(p)); out = {}
            for q in T["queries"]:
                if "objs" in T:
                    o = q.get("obj"); seq = T["objs"].get(str(o)) if o is not None else None
                else:
                    seq = q.get("frames")
                if seq is None:
                    continue
                out[q["key"]] = seq
            return T, out
    return None, {}


def boxes(seq):
    return np.array([s["st"]["box"] for s in seq if s and s.get("st")], float)


def run(split):
    D = pd.read_csv(f"{R}/submissions/sub_xt_Nzz3_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    W = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    key = {(str(j.get("video_id")), str(j.get("question"))): i for i, j in P.items()}
    out, log, cache = {}, [], {}
    for i, w in W.iterrows():
        if i not in D.index or i not in P:
            continue
        d = D.loc[i]; p = P[i]
        if d["cat"] not in ("S3", "D3") or (p.get("q") or {}).get("kind") != "size":
            continue
        mo = str(w.meas_mode)
        if "tdepth=prior" not in mo or "_scene" in mo or not np.isfinite(w.geo_value):
            continue
        src = str(d.answer_source)
        src_ok = not src.startswith("patch:")
        if src == "var_ans_plain":
            vp = str(p.get("video_id")).replace("_segmented", "")
            j0 = key.get((vp, str(p.get("question"))))
            src_ok = j0 is not None and j0 in D.index and str(D.loc[j0, "answer_source"]) in OKSRC
        elif src not in OKSRC:
            src_ok = False
        vid = p["video_id"]
        if vid not in cache:
            cache[vid] = load_tracks(split, vid)
        T, trs = cache[vid]
        if T is None:
            continue
        tph = (p.get("target") or {}).get("phrase") or ""; pph = (p.get("prior_obj") or {}).get("phrase") or ""
        if not tph or SURF.search(tph):
            continue
        tk = [k for k in trs if k.split("|")[0] == tph]; pk = [k for k in trs if k.split("|")[0] == pph]
        if not tk or not pk:
            continue
        bt = boxes(trs[tk[0]]); bp = boxes(trs[pk[0]])
        if len(bt) < 3 or len(bp) < 3:
            continue
        H = float(T["H"])
        cutb = float(np.mean(bt[:, 3] >= H - 3)); pbot = float(np.median(bp[:, 3]) / H)
        if cutb < 0.5 or pbot > 0.7:
            continue
        g = float(w.geo_value); cs = float(d.cs); ans = float(d.answer)
        if not src_ok:
            log.append((i, vid, "geometry gate passes, source excluded", src, round(ans, 3), round(g, 3), cs)); continue
        new = cs if (np.isfinite(cs) and cs < g) else min(ans, g)
        if not np.isfinite(new) or new >= ans * 0.98:
            log.append((i, vid, "no decrease", round(ans, 3), round(g, 3), cs)); continue
        out[str(int(i))] = dict(value=new, why=f"{TAG}: target bottom-cut in {cutb:.0%} of frames, prior bottom at {pbot:.2f}H -> nearer than "
                                f"the prior, tdepth=prior geo {g:.3g} is an upper bound; {'CaW' if new == cs else 'min(answer, geo)'}; was {ans:.4g} ({src})")
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    return out, log, D


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled("sub_xt_Nzz3")
    for sp in ("test", "val"):
        out, log, D = run(sp)
        print(f"== {sp}: {len(out)} changed; by cat", D["cat"].reindex([int(k) for k in out]).value_counts().to_dict())
        Ls = L[L.split == sp]
        for k, e in out.items():
            print("  CHANGE", k, D.loc[int(k), "video_id"], "->", round(e["value"], 3), "|", e["why"])
            if int(k) in Ls.index:
                y = Ls.y[int(k)]
                print("   LAB", y, A.row_mra(np.array([D.loc[int(k), "answer"]]), np.array([y])), "->", A.row_mra(np.array([e["value"]]), np.array([y])))
        for x in log:
            print("  LOG", x)
            if x[0] in Ls.index and len(x) >= 7:
                y = Ls.y[x[0]]; g, cs, a = x[5], x[6], x[4]
                newv = cs if (np.isfinite(cs) and cs < g) else min(a, g)
                print("   LAB(eval only) y", y, "m_ans", A.row_mra(np.array([a]), np.array([y]))[0], "m_geo", A.row_mra(np.array([g]), np.array([y]))[0],
                      "m_caw", A.row_mra(np.array([cs]), np.array([y]))[0], "m_rule", A.row_mra(np.array([newv]), np.array([y]))[0])
