"""uncut_geot (helper of uncut2r100_chain.py / uncut2r100_c.py / c51uncutv_c.py, 2026-10-17): the geo table of a base.

The rul439 family (analysis/v3/ag/rul439, rul441, rul090) rescales the base's own geo value g of a row and skips the
row when the base detail g differs from the raw geo table ('g_same' check: no other step changed g). The production rules
compare with preds/<split>_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv, the table of the production bases (sub_xt_Nz...). The strict
bases (build_candidate.sh, e.g. sub_xt_cand14f) are built on GEO_T=sil7: preds/<split>_geo_v4ntsil7srngsr_d_vw_raw.csv,
i.e. rj5sQ2 with geo_value of the `_segmented` copies (and lab s/x copies) multiplied by the exact-silhouette re-measurement
ratio (analysis/v3/compliance/inject_geo.py); every other column (meas_mpp, meas_tracks, meas_mode, merge_src) is identical
(checked 2026-10-17: 0 differing rows in test and val). With the rj5sQ2 table every copy row of a sil7 base fails g_same.

geo_t(R, SUB): env GEO_T when set; else the table among CANDS whose geo_value equals the base detail g (rel. 1e-4) on the
most test rows (ties -> rj5sQ2). The choice is a property of the base file and is the same for every row."""
import os
import numpy as np, pandas as pd

CANDS = ("rj5sQ2", "sil7")


def raw_path(R, split, gt):
    return f"{R}/preds/{split}_geo_v4nt{gt}srngsr_d_vw_raw.csv"


def geo_t(R, SUB):
    if os.environ.get("GEO_T"):
        return os.environ["GEO_T"], {"env": os.environ["GEO_T"]}
    det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv", index_col=0)
    g = pd.to_numeric(det.g, errors="coerce")
    n = {}
    for t in CANDS:
        raw = pd.read_csv(raw_path(R, "test", t), index_col=0)
        gr = pd.to_numeric(raw.geo_value.reindex(det.index), errors="coerce")
        n[t] = int(((g / gr - 1).abs() < 1e-4).sum())
    best = max(CANDS, key=lambda t: (n[t], t == "rj5sQ2"))
    return best, n
