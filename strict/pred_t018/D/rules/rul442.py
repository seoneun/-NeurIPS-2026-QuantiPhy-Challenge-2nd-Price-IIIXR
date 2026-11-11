#!/usr/bin/env python
import sys, json, math, os, re
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refuse import fuse, AGREE
BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t060"; GT = sys.argv[2] if len(sys.argv) > 2 else "c17"
CLIP, NMIN, DMIN, MPPTOL, DISPF = 0.2, 6, 0.25, 0.01, 0.3
ph = lambda e: (e or {}).get("phrase") if isinstance(e, dict) else e


def r100_drop(p, kind, src, old, gn, how):
    q = p["q"]
    if q.get("kind") == "size" and ph(p.get("target")) == ph(p.get("prior_obj")) and src in AGREE and gn and abs(math.log(gn / old)) < math.log(1.05):
        return f"A self-size old {old:.4g} ~ g_new {gn:.4g}"
    if q.get("kind") == "speed" and re.search(r"agree_gm\b", str(how)):
        return f"B speed agree_gm refusion (old {old:.4g}, g_new {gn:.4g})"
    return None


for split in ("test",):
    D = pd.read_csv(f"{R}/cache/tmp/ag_uncut2S2_diag_{split}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    PZ = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")}
    for c_ in ("run_n", "run_dur", "cv_run", "chord_run", "run_disp", "sh_id", "sh_d", "tc1", "tc2", "clip", "mpp_err", "old_px", "size", "cv_full"):
        if c_ not in D: D[c_] = np.nan
    D["ok2d"] = D["mode"].astype(str).str.startswith("2d_ratio")
    D["f_clip"] = D.ok2d & (D["clip"] >= CLIP)
    D["f_run"] = D.f_clip & (D.run_n >= NMIN) & (D.run_dur >= DMIN - 1e-9) & (D.run_disp >= DISPF * D["size"])
    D["f_mpp"] = D.f_run & (D.mpp_err.abs() <= MPPTOL)
    D["f_clean"] = D.f_mpp & (D.cv_run <= D.cv_full)
    D["f_shared"] = D.sh_id.notna() | D.sh_d.notna()
    D["f_tcut"] = (D.qk == "dist2") & (D.tc1.notna() | D.tc2.notna()) & ((D.tc1.astype(str) != "ok") | (D.tc2.astype(str) != "ok"))
    D["ratio"] = D.old_px / D.chord_run
    out, log = {}, []
    for i, r in D[D.f_clean].iterrows():
        if i not in det.index: continue
        b = det.loc[i]; kind = (PZ[int(i)].get("q") or {}).get("kind")
        g = float(b.g) if pd.notna(b.g) else np.nan; graw = float(pd.to_numeric(raw.geo_value.get(i), errors="coerce"))
        if not (np.isfinite(g) and g > 0 and abs(g / graw - 1) < 1e-4):
            log.append((i, 't157')); continue
        if r.f_shared: log.append((i, 't158')); continue
        if r.f_tcut: log.append((i, "dist2 target cut")); continue
        gn = g * float(r.ratio)
        v, how = fuse(b, gn, kind)
        if v is None: log.append((i, f"no fusion route {how}")); continue
        if abs(v / b.answer - 1) < 1e-6: log.append((i, "unchanged")); continue
        wd = r100_drop(PZ[int(i)], kind, str(b.answer_source), float(b.answer), gn, how)
        if wd: log.append((i, wd[:12])); continue
        out[str(int(i))] = dict(value=float(v), why='t156')
    json.dump(out, open(f"{R}/cache/tmp/ag_rul442_patch_{split}.json", "w"), indent=0)
    print(split, "2d speed-prior rows", int(D.ok2d.sum()), "clip", int(D.f_clip.sum()), "clean", int(D.f_clean.sum()), "patch", len(out),
          "skips", pd.Series([w for _, w in log]).value_counts().to_dict())
