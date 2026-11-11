#!/usr/bin/env python
import sys
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/tools")
from qp_common import OUT_COLS
split, T = sys.argv[1], sys.argv[2]
base = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{T}srng_d_vw_raw.csv").set_index("Unnamed: 0")
pre = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{T}g_d_vw_raw.csv").set_index("Unnamed: 0").reindex(base.index)
ids = set()
for tag in (f"_v3m_r{T}", f"_v3v_rnt{T}"):
    m = pd.read_csv(f"{R}/geo/out/meas_{split}_v2{tag}.csv")
    if "sruler" in m.columns:
        ids |= set(m.id[m.sruler.notna() & m.ok.astype(bool)].astype(int))
ok = base.index.isin(list(ids)) & (pre.answer_source == "meas") & (pd.to_numeric(pre.geo_value, errors="coerce") > 0)
o = base.copy()
o.loc[ok, "geo_value"] = pre.geo_value[ok]
o.loc[ok, "answer_source"] = "meas"; o.loc[ok, "meas_valid"] = True
o.loc[ok, "meas_mode"] = "2d_ratio"; o.loc[ok, "merge_src"] = "sruler"
HR = sys.argv[3] if len(sys.argv) > 3 else ""
nh = 0
if HR:
    mh = pd.read_csv(f"{R}/geo/out/meas_{split}_v2{HR}.csv").set_index("id")
    okh = mh.sruler.notna() & mh.ok.astype(bool) & (pd.to_numeric(mh.val_si, errors="coerce") > 0) if "sruler" in mh.columns else pd.Series(False, index=mh.index)
    for i in mh.index[okh]:
        if i in o.index and i not in ids:
            o.loc[i, "geo_value"] = float(mh.loc[i, "value"]); o.loc[i, "answer_source"] = "meas"; o.loc[i, "meas_valid"] = True
            o.loc[i, "meas_mode"] = "2d_ratio"; o.loc[i, "merge_src"] = "sruler_hr"; nh += 1
o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
sfx = "srngsrh" if HR else "srngsr"
o.to_csv(f"{R}/preds/{split}_geo_v4nt{T}{sfx}_d_vw_raw.csv")
o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_geo_v4nt{T}{sfx}_d_vw.csv", index=False)
if HR:
    print('')
ch = ok & ~np.isclose(pd.to_numeric(base.geo_value, errors="coerce").fillna(-1), pd.to_numeric(o.geo_value, errors="coerce").fillna(-1))
print('')
