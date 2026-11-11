"""B1 same-class ruler scan: simulation rows with a size prior whose object shares the target's head noun.
Per row: target-track features (coverage, box-area CV, quadratic R^2 of the centroid path) on the track source whose
target px quantity reproduces the row's raw geo (v2sc entity / v2sc dynamic / v3ntmv), else v2sc entity.
area_cv = box-area CV over all tracked frames; area_cv_in = the same over frames whose extent is not cut by the border.
writes cache/tmp/ag_samecls_rows_{split}.csv"""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_DRIFT", "1.5"); os.environ.setdefault("QP_PRIOR_ACCGUARD", "3")
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import numpy as np, pandas as pd
import measure_v3 as MV
import track_common as tc
MV.FIX = set("abcdef")
SUB = os.environ.get("SC_SUB", "sub_xt_Nza")
GEO = os.environ.get("SC_GEO", "geo_v4ntrj5sQ2srngsr_d_vw")
DYN_T = ("speed", "avgspeed", "accel", "disp", "path", "hvel", "maxheight")


def load_srcs(split, v):
    out = []
    for sub, cfg in (("v2sc", "e"), ("v2sc", "d"), ("v3ntmv", "e")):
        p = f"{R}/geo/out/tracks_{split}_{sub}/{v}.json"
        if os.path.exists(p):
            vj = json.load(open(p))
            try:
                out.append((f"{sub}:{cfg}", MV.Video(vj, cfg), vj))
            except Exception as e:
                pass
    return out


def feats(tr, vj):
    N = len(vj["fidx"])
    st = tr.st
    val = [s for s in st if s is not None]
    cov = len(val) / max(N, 1)
    area = np.array([s["area"] for s in val], float) if val else np.array([])
    cv = float(area.std() / area.mean()) if len(area) > 1 and area.mean() > 0 else np.nan
    r2 = np.nan; rng = np.nan; r2lin = np.nan
    if tr.n >= 5:
        t = tr.t; P = tr.uv
        ss_res = 0.0; ss_tot = 0.0; ss_res1 = 0.0
        for k in range(2):
            c2 = np.polyfit(t, P[:, k], 2); ss_res += float(((P[:, k] - np.polyval(c2, t)) ** 2).sum())
            c1 = np.polyfit(t, P[:, k], 1); ss_res1 += float(((P[:, k] - np.polyval(c1, t)) ** 2).sum())
            ss_tot += float(((P[:, k] - P[:, k].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2lin = 1 - ss_res1 / ss_tot if ss_tot > 0 else np.nan
        rng = float(np.linalg.norm(P.max(0) - P.min(0)))
    # area CV over the frames whose extent is not cut by the image border (tr.oke): an entering / leaving ball is
    # truncated by the border, which inflates the all-frame CV without any change of depth or pose
    ain = np.array([st[j]["area"] for j in tr.oke if st[j] is not None], float)
    cv_in = float(ain.std() / ain.mean()) if len(ain) >= 5 and ain.mean() > 0 else np.nan
    size = float(np.median([max(st[j]["rr"][0], 1.0) for j in tr.okc])) if tr.n else np.nan
    t_lo = float(tr.t.min()) if tr.n else np.nan; t_hi = float(tr.t.max()) if tr.n else np.nan
    return dict(t_lo=t_lo, t_hi=t_hi, N=N, n_ok=tr.n, n_oke=len(ain), cov=cov, area_cv=cv, area_cv_in=cv_in, r2=r2, r2lin=r2lin, rng_px=rng, size_px=size, rng_rel=rng / size if size and size > 0 else np.nan)


def tgt_px(tr, r):
    q = r["q"]
    try:
        if q["kind"] == "size":
            e, _ = tr.ext_px(q["dim"], r["target"]["phrase"])
            return MV.agg_ext(e, q["dim"]) if len(e) else None
        return MV.px_quantity(tr, q["kind"], q["t"], q["t1"], q["t2"], q["when"])
    except Exception:
        return None


LAB = set()
if os.environ.get("SC_ALL") == "1":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
    from lab_rows import labelled
    _L = labelled(SUB)
    LAB = set(int(i) for i, s_ in zip(_L.index, _L.split) if s_ == "test")


def run(split):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    byv = {}
    for i, r in P.items():
        if i not in det.index or not str(r["video_id"]).startswith("simulation"):
            continue
        pr = (r.get("priors") or [None])[0]
        if pr is None or pr.get("kind") != "size" or pr.get("gravity") or not pr.get("value_si"):
            continue
        tph = (r.get("target") or {}).get("phrase"); pph = (r.get("prior_obj") or {}).get("phrase")
        if not tph or not pph:
            continue
        ht, hp = tc.head_noun(tph), tc.head_noun(pph)
        if not (ht and hp and (ht == hp or tc.heads_compatible(tph, pph))) and os.environ.get("SC_ALL") != "1":
            continue
        r["_ht"], r["_hp"] = ht, hp
        if os.environ.get("SC_ALL") == "1" and split == "test" and i not in LAB:
            continue
        byv.setdefault(r["video_id"], []).append(r)
    rows = []
    for v, rs in sorted(byv.items()):
        S = load_srcs(split, v)
        for r in rs:
            i = r["id"]; x = raw.loc[i]; dd = det.loc[i]; pr = r["priors"][0]
            d = dict(id=i, split=split, video_id=v, cat=r["cat"], question=r["question"], prior=r["prior_text"], qkind=r["q"]["kind"],
                     dim=r["q"].get("dim"), tph=r["target"]["phrase"], pph=r["prior_obj"]["phrase"], ht=r["_ht"], hp=r["_hp"],
                     prior_raw=pr.get("raw"), rul413=bool(r["_ht"] and r["_hp"] and (r["_ht"] == r["_hp"] or tc.heads_compatible(tph, pph))),
                     qt=r["q"].get("t"), qt1=r["q"].get("t1"), qt2=r["q"].get("t2"), dim_prior=pr.get("dim"), any_inst=bool(re.search(r"\b(?:one|each|every|any)\b", str(pr.get("raw") or "").lower())),
                     mode=x.meas_mode, merge_src=x.merge_src, why=x.meas_why, grav=x.meas_grav, geo_raw=x.geo_value, mpp=x.meas_mpp,
                     answer=dd.answer, src=dd.answer_source, g=dd.g, cs=dd.cs, kind=dd.get("kind"))
            mpp = x.meas_mpp; geo = x.geo_value
            chosen = None
            for nm, V, vj in S:
                tr = V.get(r["target"])
                if tr is None or tr.n == 0:
                    continue
                ptr = V.get(r["prior_obj"])
                px = tgt_px(tr, r)
                match = (px is not None and pd.notna(mpp) and mpp > 0 and pd.notna(geo) and geo > 0 and abs(math.log(px * mpp / geo)) < 0.02)
                if chosen is None or (match and not chosen[3]):
                    chosen = (nm, tr, vj, match, ptr)
                if match:
                    break
            if chosen is None:
                d["status"] = "no_track"; rows.append(d); continue
            nm, tr, vj, match, ptr = chosen
            d.update(tsrc=nm, tmatch=match, same_obj=(ptr is tr), fps_trk=vj.get("fps"))
            d.update(feats(tr, vj))
            d["status"] = "ok"
            rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for split in ("val", "test"):
        df = run(split)
        df.to_csv(f"{R}/cache/tmp/ag_samecls_rows{'_all' if os.environ.get('SC_ALL') == '1' else ''}_{split}.csv", index=False)
        print(split, len(df), df.status.value_counts().to_dict(), df.cat.value_counts().to_dict())
