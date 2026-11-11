"""rul044 step 3: geo raw with the accepted re-measured rows replaced -> cache/tmp/ag_c30win_raw_{split}.csv
A candidate row (winparse.py -> remeas.py) is accepted only when ALL hold (code conditions, no per-row choices):
  - its current answer is not from patch:twin or patch:rul024 (they already re-measured the same window);
  - the base measurement (current parse, same code) reproduces the value the current answer is built from:
    geo_value of the geo raw, or meas_value for patch:rul424 rows (rul424 uses the own-row measured track);
  - the new value is finite, positive and inside measure_v3.PLAUS;
  - the new value passes the pipeline's 2-decade rule against the prior copy h1 (geo/arbitrate_v2.py).
Accepted rows: geo_value = meas_value = new, meas_mode = the re-measure's mode (2d_ratio), merge_src d, answer_source meas."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
SKIP = ("patch:twin", "patch:rul024")
for sp in ("test", "val"):
    mp = f"{R}/cache/tmp/ag_c30win_meas_{sp}.csv"
    out = f"{R}/cache/tmp/ag_c30win_raw_{sp}.csv"
    if os.path.exists(out):
        os.remove(out)
    rec = {}
    M = pd.read_csv(mp) if os.path.exists(mp) and os.path.getsize(mp) > 5 else pd.DataFrame()
    if len(M) == 0:
        json.dump({}, open(f"{R}/cache/tmp/ag_c30win_rec_{sp}.json", "w"))
        print(sp, "no re-measured rows"); continue
    raw = pd.read_csv(f"{R}/preds/{sp}_{GEO}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/sub_xt_Nzr_{'detail' if sp == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    n = 0
    for _, m in M.iterrows():
        i = int(m["id"])
        src = str(det.answer_source.get(i))
        if src in SKIP:
            print(sp, i, "skip: answered by", src); rec[i] = dict(skip=src); continue
        mv = raw.meas_value.get(i) if src == "patch:rul424" else raw.geo_value.get(i)
        base, new = float(m["base"]), float(m["new"])
        rep = bool(np.isfinite(base) and pd.notna(mv) and abs(base / float(mv) - 1) < 1e-4)
        ok = bool(m["ok"]) and np.isfinite(new) and new > 0 and bool(m.get("plaus", False))
        h1 = abs(float(raw.h1_value.get(i)))
        valid = bool(ok and abs(np.log10(new / h1)) <= 2)
        acc = rep and valid
        rec[i] = dict(src=src, rep=rep, ok=ok, valid=valid, accept=acc, base=base, new=new, used=None if pd.isna(mv) else float(mv), h1=h1,
                      raw_geo=raw.geo_value.get(i), raw_mode=raw.meas_mode.get(i), mode=m["mode"], t1=m["t1"], t2=m["t2"])
        print(sp, i, f"src={src} reproduces={rep} plaus={ok} 2-decade(h1={h1:.4g})={valid} -> accept={acc} | used {mv} ({raw.meas_mode.get(i)}) "
                     f"base {base:.5g} -> new {new:.5g} ({m['mode']})")
        if not acc:
            continue
        raw.loc[i, "geo_value"] = new; raw.loc[i, "meas_value"] = new; raw.loc[i, "meas_valid"] = True
        raw.loc[i, "meas_mode"] = m["mode"]; raw.loc[i, "meas_why"] = "rul044"; raw.loc[i, "merge_src"] = "d"
        raw.loc[i, "meas_f"] = m["f"] if pd.notna(m["f"]) else np.nan
        raw.loc[i, "answer_source"] = "meas"; raw.loc[i, "parsed_value"] = new
        n += 1
    if n:
        raw.to_csv(out)
    json.dump({str(k): v for k, v in rec.items()}, open(f"{R}/cache/tmp/ag_c30win_rec_{sp}.json", "w"), indent=1, default=str)
    print(sp, "raw rows replaced", n)
