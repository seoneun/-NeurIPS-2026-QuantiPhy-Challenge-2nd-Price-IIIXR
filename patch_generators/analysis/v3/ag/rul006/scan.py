"""B2 scan: rows with a size prior whose detector track is missing / weak / not round, in videos with a unique mover."""
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
MOTION = {"speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel", "orbitdiam"}


def load_tracks(split, tv, vid):
    p = f"{R}/geo/out/tracks_{split}{tv}/{vid}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def obj_stats(lst):
    fr = [(k, x["st"]) for k, x in enumerate(lst) if x and x.get("st")]
    if len(fr) < 3:
        return None
    c = np.array([s["c"] for _, s in fr], float)
    b = np.array([s["box"] for _, s in fr], float)
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    size = float(np.median(np.sqrt(w * h)))
    span = float(np.linalg.norm(c.max(0) - c.min(0)))
    return dict(ks=np.array([k for k, _ in fr]), c=c, w=float(np.median(w)), h=float(np.median(h)), size=size, span=span,
                cover=len(fr) / len(lst), asp=float(np.median(np.maximum(w, h) / np.maximum(1, np.minimum(w, h)))))


def movers(tj):
    out = []
    for o, lst in (tj.get("objs") or {}).items():
        st = obj_stats(lst)
        if st and st["cover"] >= 0.5 and st["span"] >= max(30.0, 1.5 * st["size"]):
            out.append((o, st))
    return out


def same_traj(a, b):
    ka = dict(zip(a["ks"].tolist(), a["c"])); d = [np.linalg.norm(ka[k] - cb) for k, cb in zip(b["ks"].tolist(), b["c"]) if k in ka]
    return len(d) >= 3 and np.median(d) < 0.5 * max(a["size"], b["size"])


def unique_movers(split, vid):
    allm = []
    for tv in ("_v2sc", "_v3nt"):
        tj = load_tracks(split, tv, vid)
        if tj is None:
            continue
        for o, st in movers(tj):
            allm.append((tv, o, st))
    groups = []
    for m in allm:
        for g in groups:
            if same_traj(g[0][2], m[2]):
                g.append(m); break
        else:
            groups.append([m])
    return groups


def qobj(tj, phrase, which="queries"):
    if tj is None:
        return None, None
    for q in tj.get(which) or []:
        if (q.get("phrase") or "").strip() == phrase:
            return q.get("obj"), q
    return None, None


if __name__ == "__main__":
    rows = []
    for split in ("test", "val"):
        P = {}
        for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
            x = json.loads(ln); P[int(x["id"])] = x
        raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
        m3 = pd.read_csv(f"{R}/geo/out/meas_{split}_v2_v3m_rrj5sQ.csv").set_index("id")
        det = pd.read_csv(f"{R}/submissions/sub_xt_Nza_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
        motion_t = {}
        for i, x in P.items():
            if (x.get("q") or {}).get("kind") in MOTION and (x.get("target") or {}).get("phrase"):
                motion_t.setdefault(x["video_id"], set()).add(x["target"]["phrase"].strip())
        cache = {}
        for i, x in P.items():
            pr = (x.get("priors") or [{}])[0]
            if pr.get("kind") != "size":
                continue
            vid = x["video_id"]; pph = ((x.get("prior_obj") or {}).get("phrase") or "").strip()
            why = str(m3.why.get(i, "")); wh = re.sub(r"\(.*", "", why)
            t2 = load_tracks(split, "_v2sc", vid); t3 = load_tracks(split, "_v3nt", vid)
            o2, q2 = qobj(t2, pph); o3, q3 = qobj(t3, pph)
            s2 = obj_stats(t2["objs"][str(o2)]) if (t2 and o2 is not None and str(o2) in t2["objs"]) else None
            s3 = obj_stats(t3["objs"][str(o3)]) if (t3 and o3 is not None and str(o3) in t3["objs"]) else None
            notround = pr.get("dim") == "diameter" and s2 is not None and s2["asp"] > 1.4
            trig = wh in ("prior_no_track", "prior_low_det") or notround
            if not trig:
                continue
            if vid not in cache:
                cache[vid] = unique_movers(split, vid)
            G = cache[vid]
            mov_ok = len(G) == 1
            mv = G[0] if mov_ok else None
            p3_is_mover = bool(mv and s3 is not None and any(tv == "_v3nt" and str(o) == str(o3) for tv, o, _ in mv))
            p2_is_mover = bool(mv and s2 is not None and any(tv == "_v2sc" and str(o) == str(o2) for tv, o, _ in mv))
            rows.append(dict(split=split, id=i, vid=vid, prior=x.get("prior_text", "")[:40], pdim=pr.get("dim"), q=x["question"][:55], why=why[:28], notround=notround,
                             nmov=len(G), pmot=pph in motion_t.get(vid, set()), p3mov=p3_is_mover, p2mov=p2_is_mover,
                             mode=raw.meas_mode.get(i), msrc=raw.merge_src.get(i), g=det.g.get(i), cs=det.cs.get(i), ans=det.answer.get(i), src=det.answer_source.get(i)))
    Z = pd.DataFrame(rows)
    Z.to_csv(f"{R}/cache/tmp/ag_bindmover_scan.csv", index=False)
    print(len(Z), "trigger rows;", Z.groupby(["split"]).vid.nunique().to_dict(), "videos")
    print(Z.groupby(["split", "nmov", "pmot"]).size().to_string())
    vs = Z.groupby(["split", "vid"]).agg(n=("id", "size"), prior=("prior", "first"), why=("why", "first"), nmov=("nmov", "first"), pmot=("pmot", "max"), p3mov=("p3mov", "max"), p2mov=("p2mov", "max"), mode=("mode", lambda s: ",".join(sorted(set(map(str, s))))))
    print(vs.to_string())
