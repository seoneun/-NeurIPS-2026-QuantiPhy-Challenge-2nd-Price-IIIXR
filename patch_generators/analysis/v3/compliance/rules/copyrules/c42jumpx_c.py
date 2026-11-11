"""rul063: coverage extension of the accepted `jump` rule (analysis/v3/ag/jump/jump_patch.py).
jump skips every row whose prior track source it cannot match ('src_unmatched': it only tries v2sc / v2 / v3ntmv and
matches against the OLD geo raw rj5sdGsrngsr). Here the same flags / clean / prior_kin / fuse are run against the CURRENT
raw (rj5sQ2srngsr meas_mpp) and a wider list of track dirs and Video configs. Nothing else changes: same thresholds,
same re-fusion, rows already in any applied patch are skipped.
usage: python rul063.py [sub_name]  -> cache/tmp/ag_c42jumpx_patch_{test,val}.json

copyrules fixed copy (2026-10-17, agent jumpffg) of analysis/v3/ag/rul063/rul063.py. One change:
  applied_ids() read the patch-tag list of the build from cache/tmp/sr84.sh, which no longer exists (the production
  re-run on any base failed with FileNotFoundError at rul063.py line 24). "Row already patched" is now the base row's
  own answer_source starting with "patch" (a property of that row in the base submission). Thresholds, track sources,
  the only_unmatched gate (rows whose jump status is src_unmatched) and the re-fusion are unchanged.
  When no row matches a track source (status ok), an empty patch is written instead of a KeyError on the empty table
  (happens since geo/out/tracks_test_v3nt32, the source of all 14 production rows, is absent from the local disk).
  The repo root is derived from QP_ROOT or five levels above this file (this file sits one level deeper)."""
import os, sys, json, math
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/analysis/v3/ag/jump")
import numpy as np, pandas as pd
import jump_patch as JP
import measure_v3 as MV
from scan import flags, sub_track
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzy"
RAWTAG = "geo_v4ntrj5sQ2srngsr_d_vw"
SRCS = [("v2sc", "d"), ("v2", "d"), ("v3ntmv", "v"), ("v3nt", "v"), ("v3ntmo", "v"), ("v3nt32", "v"), ("v3", "v"),
        ("v2full", "d"), ("v2repro", "d"), ("v2mv", "d"), ("v2sw", "d"), ("v2sa", "d")]
CFGS = ("e", "d")


def applied_ids(det):
    """rows already patched in the base: the row's own answer_source starts with 'patch' (replaces the sr84.sh tag list)"""
    return set(int(i) for i in det.index[det.answer_source.astype(str).str.startswith("patch")]), []


def run(split, only_unmatched=True):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWTAG}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    old = pd.read_csv(f"{R}/cache/tmp/ag_jump_rows_{split}.csv").set_index("id")
    done, _ = applied_ids(det)
    rows = []
    for i, r in sorted(P.items()):
        pr = (r.get("priors") or [None])[0]
        if r["cat"] != "D2" or pr is None or pr.get("kind") not in ("speed", "accel") or pr.get("gravity") or pr.get("value_si") is None:
            continue
        if only_unmatched and not (i in old.index and old.loc[i, "status"] == "src_unmatched"):
            continue
        x = raw.loc[i]; mpp = x.meas_mpp
        d = dict(id=i, split=split, video_id=r["video_id"], question=r["question"], qkind=r["q"]["kind"], mode=x.meas_mode,
                 geo_raw=x.geo_value)
        if not (pd.notna(mpp) and mpp > 0):
            d["status"] = "no_mpp"; rows.append(d); continue
        implied = pr["value_si"] / mpp
        hit = None
        for sub, nm in SRCS:
            p = f"{R}/geo/out/tracks_{split}_{sub}/{r['video_id']}.json"
            if not os.path.exists(p):
                continue
            vj = json.load(open(p))
            for cfg in CFGS:
                try:
                    V = MV.Video(vj, cfg)
                    tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
                except Exception:
                    continue
                if tr is None or tr.n < 6:
                    continue
                o_ = MV.prior_kin(tr, pr)
                if o_ is not None and o_ > 0 and abs(math.log(o_ / implied)) < 0.01:
                    hit = (sub, cfg, V, tr, float(vj["fps"]), o_); break
            if hit:
                break
        if hit is None:
            d["status"] = "src_unmatched2"; rows.append(d); continue
        sub, cfg, V, tr, fps, o_ = hit
        fl = flags(tr, V, fps)
        idx, niou = JP.clean_idx(tr, fl)
        new = MV.prior_kin(sub_track(tr, idx), pr) if ((fl["step"] or fl["size"] or niou) and len(idx) >= 6) else None
        d.update(src=sub + ":" + cfg, old_px=o_, p_new_px=new, p_step=len(fl["step"]), p_size=len(fl["size"]), p_iou_used=niou,
                 p_steps=",".join(map(str, fl["step"][:5])), p_sizes=",".join(map(str, fl["size"][:5])))
        fac = 1.0
        if new is not None and new > 0 and abs(math.log(o_ / new)) >= JP.MIN_CHANGE:
            fac = o_ / new
        d["fac"] = fac; d["status"] = "ok"; d["patched"] = i in done
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.join(det[["answer", "answer_source", "g", "cs"]], on="id")
    return df


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 50)
    allp = {}
    for split in ("val", "test"):
        df = run(split)
        df.to_csv(f"{R}/cache/tmp/ag_c42jumpx_rows_{split}.csv", index=False)
        print(f"== {split}: rows {len(df)}", df.status.value_counts().to_dict() if len(df) else {})
        if not len(df):
            json.dump({}, open(f"{R}/cache/tmp/ag_c42jumpx_patch_{split}.json", "w")); allp[split] = {}; continue
        ok = df[df.status == "ok"]
        if not len(ok):     # copyrules: no matched row (e.g. tracks_*_v3nt32 absent) -> empty patch instead of a KeyError
            json.dump({}, open(f"{R}/cache/tmp/ag_c42jumpx_patch_{split}.json", "w")); allp[split] = {}; continue
        print(ok[["id", "video_id", "src", "old_px", "p_new_px", "p_step", "p_size", "p_steps", "fac", "patched", "answer_source", "g", "answer"]].round(4).to_string())
        patch, out = {}, []
        for _, rw in ok.iterrows():
            if rw.patched or abs(math.log(rw.fac)) < JP.MIN_CHANGE or not pd.notna(rw.g):
                continue
            g_new = float(rw.g) * float(rw.fac)
            val, why = JP.fuse(rw, g_new)
            if val is None:
                out.append(dict(id=int(rw.id), video_id=rw.video_id, skip=why)); continue
            patch[str(int(rw.id))] = dict(value=float(val), why=f"rul063 prior track jump (steps {rw.p_steps or '-'} sizes {rw.p_sizes or '-'}, src {rw.src}): prior px {rw.old_px:.2f}->{rw.p_new_px:.2f}; geo {rw.g:.4g}->{g_new:.4g}; {why}")
            out.append(dict(id=int(rw.id), video_id=rw.video_id, q=rw.question[:60], src_=rw.answer_source, old=rw.answer, new=val, g=rw.g, g_new=g_new, cs=rw.cs, fac=rw.fac, how=why))
        json.dump(patch, open(f"{R}/cache/tmp/ag_c42jumpx_patch_{split}.json", "w"), indent=1)
        allp[split] = patch
        print(f"   rows changed: {len(patch)}")
        if out:
            print(pd.DataFrame(out).round(4).to_string())
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB).reset_index()
    L = L.rename(columns={L.columns[0]: "id"})
    L["new"] = [allp[s].get(str(int(i)), {}).get("value", np.nan) for s, i in zip(L.split, L.id)]
    hit = L[L.new.notna()]
    print(f"== labelled rows: {len(L)}, changed {len(hit)}")
    if len(hit):
        hit = hit.assign(m_new=A.row_mra(hit.new.values, hit.y.values))
        print(hit[["split", "id", "video_id", "cat", "answer", "new", "y", "m", "m_new"]].round(4).to_string())
