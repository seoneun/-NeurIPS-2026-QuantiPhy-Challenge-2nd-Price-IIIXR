"""c51jumpv_rep (check_repair grpD, 2026-10-17): analysis/v3/ag/rul089/rul089.py run() unchanged (imported read-only),
with one input replaced and the evaluation block dropped.
1. Input: rul089/vc.py vlm_rows() reads the VLM-chain measurement's own meas_mpp / meas_f / meas_why / meas_tracks /
   meas_mode from preds/<split>_geo_v3v_rntrj5sQ_raw.csv (production joint measurement; the file no longer exists, no git
   history). Here the same columns come from the per-question VLM-chain measurement that produced the clean geo table
   GEO_T=ffmv: cache/tmp/ag_ffmv_meas_<split>_v_ffmvall.csv (own-text track pool, analysis/v3/compliance/notes/ffmover.md
   section 3 and owntracks.md section 3; columns mpp, f, why, tracks, mode = meas_<c> as in copyrules/c51uncutv_c.py rawown).
   The final raw (preds/<split>_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv, the ffmv table in the clean shadow) gives the VLM-chain
   rows (merge_src v / xg_v), geo_value and meas_mode exactly as vc.vlm_rows does.
2. The original __main__ evaluation (labels, cache/tmp/ag_a42s0_fuzzycopies.csv at rul089.py:109, after the patches are
   written) is dropped: that file does not exist and the read only prints touched fuzzy copies.
Everything else (prior / target track jump on tracks_<split>_v3ntmv, 1 % src check, MIN_CHANGE, jump fuse on the base
row) is rul089.run().
usage: python analysis/v3/compliance/check_repair/c51jumpv_rep.py sub_xt_<base>
writes cache/tmp/ag_c51jumpv_patch_{test,val}.json"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/analysis/v3/ag/rul089")
import vc
import rul089 as CJ                        # module top level: SUB = argv[1]; writes nothing
MEAS = "cache/tmp/ag_ffmv_meas_{split}_v_ffmvall.csv"
COLS = {"meas_mpp": "mpp", "meas_f": "f", "meas_why": "why", "meas_tracks": "tracks", "meas_mode": "mode"}


def vlm_rows(split):
    """vc.vlm_rows with the VLM measurement columns from the per-question measurement table"""
    r = pd.read_csv(f"{R}/preds/{split}_{vc.RAW}_raw.csv").set_index("Unnamed: 0")
    v = pd.read_csv(f"{R}/{MEAS.format(split=split)}").set_index("id").reindex(r.index)
    vl = r.merge_src.astype(str).isin(["v", "xg_v"])
    X = pd.DataFrame(index=r.index[vl])
    X["vid"] = r.video_id[vl]; X["ms"] = r.merge_src[vl]; X["g"] = pd.to_numeric(r.geo_value[vl], errors="coerce")
    X["mode"] = r.meas_mode[vl]; X["fps"] = r.fps[vl]
    for c, cm in COLS.items():
        X[c + "_v"] = v[cm][vl]
        X[c + "_r"] = r[c][vl]
    return X


vc.vlm_rows = vlm_rows
for split in ("test", "val"):
    df, patch = CJ.run(split)
    json.dump(patch, open(f"{R}/cache/tmp/ag_{CJ.TAG}_patch_{split}.json", "w"), indent=1)
    print(f"== {split}: base {CJ.SUB}; VLM-chain rows in jump's population {len(df)}; status {df.status.value_counts().to_dict() if len(df) else {}}; patched {len(patch)}")
    for k, e in patch.items():
        i = int(k)
        print(f"   {k} {df.vid[i]} | {df.answer_source[i]} {df.answer[i]:.4g} g {df.g[i]:.4g} cs {df.cs[i]} -> {e['value']:.6g} | {e['why']}")
