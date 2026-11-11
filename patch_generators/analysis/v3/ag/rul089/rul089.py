"""rul089: coverage extension of the accepted `jump` rule (analysis/v3/ag/jump/jump_patch.py, rul063) to the VLM-chain rows.

Why rows were skipped: a row whose geo came from the VLM track chain (final raw merge_src 'v' / 'xg_v'; measure_v3 --tracks
v3ntmv, preds/<split>_geo_v3v_rntrj5sQ_raw.csv) keeps the DETECTOR measurement's meas_mpp / meas_why in the final raw
(fuse_v3 / xground copy the detector raw and replace only geo_value / meas_mode). jump reads that meas_mpp: NaN when the
detector measured nothing ('no_mpp'), or the detector's prior track. Here the same code runs with the measurement's own mpp
(v3v raw) and its own track file (tracks_<split>_v3ntmv, loaded as measure_v3 resolves it); prior_kin must reproduce
value_si / meas_mpp within 1 % (all 62 test VLM rows of jump's population do: 0.0 % error). Everything else is jump's:
flags / clean_idx / MIN_CHANGE / prior and target steps / fuse. Rows already set by any patch (answer_source patch*) are left.
Labels only evaluate.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul089/rul089.py [SUB=sub_xt_Nzz6]
writes cache/tmp/ag_c51jumpv_patch_{test,val}.json"""
import os, sys, json, math
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/rul089")
import vc                                        # env of the rj5sQ chain + measure_v3 (read-only import)
sys.path.insert(0, R + "/analysis/v3/ag/jump")
import numpy as np, pandas as pd
import jump_patch as JP
from scan import flags, sub_track
MV = vc.MV
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz6"
TAG = "rul089"
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)


def run(split):
    X = vc.vlm_rows(split)
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    rows = []
    for i in X.index:
        r = P[i]; pr = (r.get("priors") or [None])[0]
        if r["cat"] != "D2" or pr is None or pr.get("kind") not in ("speed", "accel") or pr.get("gravity") or pr.get("value_si") is None:
            continue
        d = dict(id=i, vid=r["video_id"], q=r["question"][:60], qkind=r["q"]["kind"], mode=X["mode"][i], ms=X.ms[i], geo_raw=X.g[i],
                 mpp_final_raw=X.meas_mpp_r[i], mpp=X.meas_mpp_v[i])
        mpp = X.meas_mpp_v[i]
        if not (pd.notna(mpp) and mpp > 0):
            d["status"] = "no_mpp"; rows.append(d); continue
        vj, td = vc.load_vj(split, r["video_id"], "v3ntmv")
        V = MV.Video(vj, "e"); fps = float(vj["fps"])
        tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
        if tr is None or tr.n < 6:
            d["status"] = "no_prior_track"; rows.append(d); continue
        o_ = MV.prior_kin(tr, pr)
        if o_ is None or abs(math.log(o_ / (pr["value_si"] / mpp))) >= 0.01:
            d["status"] = "src_unmatched"; rows.append(d); continue
        fl = flags(tr, V, fps)
        idx, niou = JP.clean_idx(tr, fl)
        new = MV.prior_kin(sub_track(tr, idx), pr) if ((fl["step"] or fl["size"] or niou) and len(idx) >= 6) else None
        d.update(td=td, old_px=o_, p_new_px=new, p_steps=",".join(map(str, fl["step"][:5])), p_sizes=",".join(map(str, fl["size"][:5])))
        fac = 1.0
        if new is not None and new > 0 and abs(math.log(o_ / new)) >= JP.MIN_CHANGE:
            fac *= o_ / new; d["p_fac"] = o_ / new
        q = r["q"]
        if q["kind"] in JP.DYN_T and str(X["mode"][i]).startswith(("2d_ratio", "vlm_2d_ratio")):
            ttr = V.get(r["target"]) if (r.get("target") or {}).get("phrase") else None
            if ttr is not None and ttr.n >= 6 and ttr is not tr:
                t_old = MV.px_quantity(ttr, q["kind"], q["t"], q["t1"], q["t2"], q["when"])
                if t_old is not None and abs(math.log(t_old * mpp / float(X.g[i]))) < 0.01:
                    tfl = flags(ttr, V, fps); tidx, tniou = JP.clean_idx(ttr, tfl)
                    d.update(t_old_px=t_old, t_steps=",".join(map(str, tfl["step"][:5])))
                    if (tfl["step"] or tfl["size"] or tniou) and len(tidx) >= 6:
                        t_new = MV.px_quantity(sub_track(ttr, tidx), q["kind"], q["t"], q["t1"], q["t2"], q["when"])
                        d["t_new_px"] = t_new
                        if t_new is not None and t_new > 0 and abs(math.log(t_new / t_old)) >= JP.MIN_CHANGE:
                            fac *= t_new / t_old; d["t_fac"] = t_new / t_old
                else:
                    d["t_status"] = "t_unmatched"
        d["fac"] = fac; d["status"] = "ok"
        rows.append(d)
    df = pd.DataFrame(rows).set_index("id").join(det[["answer", "answer_source", "g", "cs"]])
    patch = {}
    for i, rw in df[(df.status == "ok") & (np.abs(np.log(df.fac.astype(float))) >= JP.MIN_CHANGE)].iterrows():
        if str(rw.answer_source).startswith("patch"):
            continue
        g_new = float(rw.g) * float(rw.fac)
        val, why = JP.fuse(rw, g_new)
        if val is None:
            continue
        patch[str(int(i))] = dict(value=float(val), why=(f"{TAG}: jump on the VLM track the row's geo used (v3ntmv, own mpp {rw.mpp:.4g}; final raw mpp "
                                                         f"{rw.mpp_final_raw}): prior track jump (steps {rw.p_steps or '-'}): prior px {rw.old_px:.1f}->"
                                                         f"{rw.p_new_px:.1f}; geo {rw.g:.4g}->{g_new:.4g}; {why}"))
    return df, patch


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
    import analyze_caw as A
    from lab_rows import labelled
    out = {}
    for split in ("test", "val"):
        df, patch = run(split)
        out[split] = (df, patch)
        json.dump(patch, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        print(f"== {split}: VLM-chain rows in jump's population {len(df)}; status {df.status.value_counts().to_dict()}; "
              f"final-raw mpp NaN {int(df.mpp_final_raw.isna().sum())}; prior flagged {int(df.get('p_fac', pd.Series(dtype=float)).notna().sum())}; "
              f"target checked {int(df.get('t_old_px', pd.Series(dtype=float)).notna().sum())}; patched {len(patch)}")
        for k, v in patch.items():
            i = int(k)
            print(f"   {k} {df.vid[i]} | {df.q[i]} | {df.answer_source[i]} {df.answer[i]:.4g} -> {v['value']:.4g} | {v['why']}")
    L = labelled(SUB)
    hit = [(s, int(i)) for s, i in zip(L.split, L.index) if str(int(i)) in out[s][1]]
    print("labelled rows (val + all-237) touched:", hit)
    fz = pd.read_csv(f"{R}/cache/tmp/ag_a42s0_fuzzycopies.csv")
    fz = fz[fz.test != 494]
    print("fuzzy-copy test rows touched:", [int(t) for t in fz.test if str(int(t)) in out["test"][1]])
