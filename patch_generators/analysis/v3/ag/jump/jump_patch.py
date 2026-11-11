"""jump rule: a prior-object (or target) track of a 2D dynamic-prior row whose box jumps (centroid step > 3x the median
step + 0.15 x object size, a rigid box changing size > 40 % in one step, or a partial overlap IoU > 0.3 with another
same-class query object) is re-measured without the jump frames (the frames on both sides of a flagged step, the overlap
frames) with measure_v3.prior_kin / px_quantity, unchanged otherwise. The 2D scale is prior_si / prior_px, so
new_geo = geo * old_px / new_px (prior) or geo * new_px / old_px (target).
The track source of each row (detector v2sc/v2 or VLM v3ntmv) is the one whose prior_kin reproduces the row's measured
metres-per-pixel (prior_si / meas_mpp) within 1 %.
Whole-clip overlaps (IoU on > 50 % of the frames) are duplicate SAM2 objects of one query, not jumps: not used.
writes cache/tmp/ag_jump_patch_{test,val}.json and cache/tmp/ag_jump_rows_{split}.csv
usage: python analysis/v3/ag/jump/jump_patch.py [sub_name]"""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_DRIFT", "1.5"); os.environ.setdefault("QP_PRIOR_ACCGUARD", "3")
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/analysis/v3/ag/jump")
import numpy as np, pandas as pd
import measure_v3 as MV
from scan import flags, sub_track, theil_sen_speed
MV.FIX = set("abcdef")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Ngd"
MIN_CHANGE = float(os.environ.get("JUMP_MIN", "0.02"))     # |log ratio| below this: no change
DYN_T = ("speed", "avgspeed", "accel", "disp", "path", "hvel", "maxheight")


def load_tracks(split, v):
    out = {}
    for nm, sub in (("d", "v2sc"), ("d", "v2"), ("v", "v3ntmv")):
        if nm in out:
            continue
        p = f"{R}/geo/out/tracks_{split}_{sub}/{v}.json"
        if os.path.exists(p):
            vj = json.load(open(p))
            out[nm] = (MV.Video(vj, "e"), float(vj["fps"]), sub)
    return out


def clean_idx(tr, fl):
    n = tr.n
    iou_ = fl["iou"] if len(fl["iou"]) <= 0.5 * n else []          # whole-clip overlap = duplicate object, not a jump
    drop = set(iou_)
    for k in sorted(set(fl["step"]) | set(fl["size"])):
        drop.add(k); drop.add(k + 1)
    return [i for i in range(n) if i not in drop], len(iou_)


def run(split):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    byv = {}
    for i, r in P.items():
        pr = (r.get("priors") or [None])[0]
        if r["cat"] != "D2" or pr is None or pr.get("kind") not in ("speed", "accel") or pr.get("gravity") or pr.get("value_si") is None:
            continue
        byv.setdefault(r["video_id"], []).append(r)
    rows = []
    for v, rs in sorted(byv.items()):
        T = load_tracks(split, v)
        if not T:
            continue
        cache = {}
        for r in rs:
            i = r["id"]; pr = r["priors"][0]
            x = raw.loc[i]
            mpp = x.meas_mpp
            d = dict(id=i, split=split, video_id=v, cat=r["cat"], question=r["question"], prior=r["prior_text"], qkind=r["q"]["kind"],
                     mode=x.meas_mode, merge_src=x.merge_src, geo_raw=x.geo_value)
            if not (pd.notna(mpp) and mpp > 0):
                d["status"] = "no_mpp"; rows.append(d); continue
            implied = pr["value_si"] / mpp
            src, ptr, fps = None, None, None
            for nm, (V, fps_, sub) in T.items():
                tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
                if tr is None or tr.n < 6:
                    continue
                k_ = ("old", nm, id(tr), pr["kind"], pr.get("t"), pr.get("t1"), pr.get("t2"))
                if k_ not in cache:
                    cache[k_] = MV.prior_kin(tr, pr)
                o_ = cache[k_]
                if o_ is not None and abs(math.log(o_ / implied)) < 0.01:
                    src, ptr, fps, Vs, old = nm, tr, fps_, V, o_
                    break
            if src is None:
                d["status"] = "src_unmatched"; rows.append(d); continue
            d.update(src=src, old_px=old)
            kf = ("flag", src, id(ptr), pr["kind"], pr.get("t"), pr.get("t1"), pr.get("t2"))
            if kf not in cache:
                fl = flags(ptr, Vs, fps)
                idx, niou = clean_idx(ptr, fl)
                new = None
                if (fl["step"] or fl["size"] or niou) and len(idx) >= 6:
                    new = MV.prior_kin(sub_track(ptr, idx), pr)
                cache[kf] = (fl, new, len(idx), niou)
            fl, new, nk, niou = cache[kf]
            d.update(p_n=ptr.n, p_step=len(fl["step"]), p_size=len(fl["size"]), p_iou=len(fl["iou"]), p_iou_used=niou, p_keep=nk,
                     p_new_px=new, p_steps=",".join(map(str, fl["step"][:5])), p_sizes=",".join(map(str, fl["size"][:5])))
            fac = 1.0
            if new is not None and new > 0 and abs(math.log(old / new)) >= MIN_CHANGE:
                fac *= old / new
                d["p_fac"] = old / new
            # ---- target track of a 2D motion question (same track source; skipped when the target is the prior track)
            if r["q"]["kind"] in DYN_T and str(x.meas_mode).startswith(("2d_ratio", "vlm_2d_ratio")):
                ttr = Vs.get(r["target"]) if (r.get("target") or {}).get("phrase") else None
                if ttr is not None and ttr.n >= 6 and ttr is not ptr:
                    q = r["q"]
                    t_old = MV.px_quantity(ttr, q["kind"], q["t"], q["t1"], q["t2"], q["when"])
                    if t_old is not None and pd.notna(x.geo_value) and abs(math.log(t_old * mpp / float(x.geo_value))) < 0.01:
                        tfl = flags(ttr, Vs, fps)
                        tidx, tniou = clean_idx(ttr, tfl)
                        d.update(t_step=len(tfl["step"]), t_size=len(tfl["size"]), t_iou=len(tfl["iou"]), t_old_px=t_old)
                        if (tfl["step"] or tfl["size"] or tniou) and len(tidx) >= 6:
                            t_new = MV.px_quantity(sub_track(ttr, tidx), q["kind"], q["t"], q["t1"], q["t2"], q["when"])
                            d["t_new_px"] = t_new
                            if t_new is not None and t_new > 0 and abs(math.log(t_new / t_old)) >= MIN_CHANGE:
                                fac *= t_new / t_old
                                d["t_fac"] = t_new / t_old
                    else:
                        d["t_status"] = "t_unmatched"
            d["fac"] = fac
            d["status"] = "ok"
            rows.append(d)
    df = pd.DataFrame(rows)
    df = df.join(det[["answer", "answer_source", "g", "cs"]], on="id")
    return df


def fuse(row, g_new):
    """the fusion of the row redone with the corrected geo"""
    a, src, g, cs = float(row.answer), str(row.answer_source), float(row.g), row.cs
    if not np.isfinite(g) or g <= 0:
        return None, "no_geo"
    if not (pd.notna(row.geo_raw) and abs(math.log(g / float(row.geo_raw))) < 0.01):
        return None, "g_changed_downstream"                  # a later rule rescaled geo: the px ratio no longer applies
    have_c = pd.notna(cs) and float(cs) > 0
    cs = float(cs) if have_c else np.nan
    if src.startswith("agree") or src.startswith("dis_geo") or src in ("meas", "geo"):
        if have_c and not (0.5 <= g_new / cs <= 2.0):
            return g_new, "dis_geo_after_jumpfix"            # 2D ratio is a strong mode: geo on disagreement
        if src == "agree_gm" and have_c:
            return math.sqrt(g_new * cs), "agree_gm"
        if src.startswith("agree") and have_c and abs(math.log(g / cs)) > 1e-6 and abs(math.log(a / g)) > 1e-4:
            w = math.log(a / cs) / math.log(g / cs)          # same weights as the original fusion
            w = min(max(w, 0.0), 1.0)
            return math.exp(w * math.log(g_new) + (1 - w) * math.log(cs)), src + f"(w={w:.2f})"
        return g_new, src
    if not (src.startswith("caw") or src.startswith("dis_caw")):
        return None, "other_source_kept"                     # var_ans_plain / phys_bound / kin ...: left to their own rule
    if have_c and 1 / 1.3 <= g_new / cs <= 1.3:
        return math.sqrt(g_new * cs), "caw_row_now_agree_gm"
    return None, "caw_kept"


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
    allp = {}
    for split in ("val", "test"):
        df = run(split)
        df.to_csv(f"{R}/cache/tmp/ag_jump_rows_{split}.csv", index=False)
        ok = df[df.status == "ok"]
        print(f"== {split}: D2 dynamic-prior rows {len(df)}, status {df.status.value_counts().to_dict()}, src {ok.src.value_counts().to_dict()}")
        print("   prior tracks flagged (rows):", int(((ok.p_step > 0) | (ok.p_size > 0) | (ok.p_iou_used > 0)).sum()),
              "| whole-clip duplicate overlaps (rows, unused):", int(((ok.p_iou > 0) & (ok.p_iou_used == 0)).sum()),
              "| target tracks checked:", int(ok.get("t_old_px", pd.Series(dtype=float)).notna().sum()),
              "flagged:", int(((ok.get("t_step", 0) > 0) | (ok.get("t_size", 0) > 0) | (ok.get("t_iou", 0) > 0)).sum()) if "t_step" in ok else 0)
        ch = ok[(np.abs(np.log(ok.fac)) >= MIN_CHANGE) & ok.g.notna()].copy()
        patch = {}
        out = []
        for _, rw in ch.iterrows():
            g_new = float(rw.g) * float(rw.fac)
            val, why = fuse(rw, g_new)
            if val is None:
                continue
            parts = []
            if pd.notna(rw.get("p_fac")):
                parts.append(f"prior track jump (steps {rw.p_steps or '-'} sizes {rw.p_sizes or '-'}): prior px {rw.old_px:.1f}->{rw.p_new_px:.1f}")
            if pd.notna(rw.get("t_fac")):
                parts.append(f"target track jump: target px {rw.t_old_px:.1f}->{rw.t_new_px:.1f}")
            patch[str(int(rw.id))] = dict(value=float(val), why="; ".join(parts) + f"; geo {rw.g:.4g}->{g_new:.4g}; {why}")
            out.append(dict(id=int(rw.id), video_id=rw.video_id, cat=rw["cat"], q=rw.question[:60], src_=rw.answer_source, old=rw.answer, new=val,
                            g=rw.g, g_new=g_new, cs=rw.cs, fac=rw.fac, how=why))
        json.dump(patch, open(f"{R}/cache/tmp/ag_jump_patch_{split}.json", "w"), indent=1)
        allp[split] = patch
        o = pd.DataFrame(out)
        print(f"   rows changed: {len(patch)}" + (f" by cat {o.cat.value_counts().to_dict()}" if len(o) else ""))
        if len(o):
            print(o.round(4).to_string())
    # ---- evaluation on labelled rows
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    L = L.reset_index().rename(columns={"index": "id", "Unnamed: 0": "id"})
    L["new"] = [allp[s].get(str(int(i)), {}).get("value", np.nan) for s, i in zip(L.split, L.id)]
    hit = L[L.new.notna()]
    print(f"== labelled rows (all-237 + val): {len(L)}, changed {len(hit)}")
    if len(hit):
        hit = hit.assign(m_new=A.row_mra(hit.new.values, hit.y.values))
        print(hit[["split", "id", "video_id", "cat", "answer", "new", "y", "m", "m_new"]].round(4).to_string())
        print("   MRA before", round(hit.m.mean(), 4), "after", round(hit.m_new.mean(), 4))
    # flagged-but-unchanged labelled rows (did the flag fire on labelled videos at all?)
    for split in ("val", "test"):
        df = pd.read_csv(f"{R}/cache/tmp/ag_jump_rows_{split}.csv")
        lab_ids = set(L[L.split == split].id.astype(int))
        f = df[(df.status == "ok") & df.id.isin(lab_ids)]
        fl = f[(f.p_step > 0) | (f.p_size > 0) | (f.p_iou > 0) | (f.get("t_step", 0) > 0) | (f.get("t_size", 0) > 0) | (f.get("t_iou", 0) > 0)]
        print(f"   {split}: labelled D2 dynamic rows measured {len(f)}, with any flag {len(fl)}:", fl[["id", "video_id", "p_step", "p_size", "p_iou", "fac"]].to_dict("records")[:10])
