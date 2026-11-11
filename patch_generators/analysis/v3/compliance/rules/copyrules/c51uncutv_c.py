"""c51uncutv_c (copyrules, 2026-10-17): analysis/v3/ag/rul090/rul090.py with three changes, no gate value changed.
 1. load_rule(): the exec namespace gets __file__ (= analysis/v3/ag/rul439/rule.py). The strict re-run without QP_ROOT failed
    with NameError '__file__' inside the exec of rul439/rule.py (its line 27 evaluates __file__ when QP_ROOT is unset).
 2. make_diag(): the generated copy of rul439/diag.py is written to cache/tmp/ag_c51uncutv_diag_vlm_gen.py instead of
    analysis/v3/ag/rul090/diag_vlm.py, so a run does not rewrite a production file (in rerun_shadow.sh, analysis/ is a
    symlink to the real repo; the shadow cache/tmp is reset after the run).
 3. g_same check: the base detail g is compared with geo_value of preds/<split>_geo_v4nt<GEO_T>srngsr_d_vw_raw.csv, GEO_T =
    the base's geo table (env GEO_T, else uncut_geot.geo_t: sil7 for the strict base sub_xt_cand14f, rj5sQ2 for production
    bases). With the production rj5sQ2 table every `_segmented` copy of a sil7 base fails g_same although its g is the
    base's own geo. The rawown table used by the diag (meas_* columns) is unchanged.
 4. rawown(): the VLM-chain measurement columns are read from geo/out/meas_<split>_v2_v3v_rntrj5sQ.csv (the input of
    geo/arbitrate_v2.py) because preds/<split>_geo_v3v_rntrj5sQ_raw.csv no longer exists; same values (see rawown).
 The final call of c51uncutv_eval.py (evaluation that emulates submit_v3's patch_copy, i.e. copies following their twin)
 and the fuzzy-copy report are removed: both evaluate cross-row copying, which the strict build does not use.

rul090: coverage extension of the accepted `rul439` rule (analysis/v3/ag/rul439/{diag,rule}.py) to the VLM-chain rows.

rul439 rescales every 2D-ratio row of a video whose time-less speed prior track is border-cut in >= 20 % of its boxes
(prior px speed re-read on the longest uncut run). It skipped the VLM-chain rows (final raw merge_src 'v' / 'xg_v') only
because of their track source: their mode carries the chain's 'vlm_' prefix (vlm_2d_ratio = 2d_ratio measured on the
VLM track) and the final raw holds the DETECTOR measurement's meas_mpp / meas_tracks (fuse_v3 / xground copy the detector
raw and replace only geo_value / meas_mode). Here:
  1. a raw copy whose VLM-chain rows carry their own measurement's meas_* columns (preds/<split>_geo_v3v_rntrj5sQ_raw.csv)
     and meas_tracks 'v1' -> rul439/diag.py's own v1 -> tracks_<split>_v3ntmv mapping reads the VLM track file;
  2. rul439/diag.py run unchanged on it (generated copy diag_vlm.py: raw path, VLM-video filter, output name only);
  3. rul439/rule.py's gates and fuse (exec'd from the accepted file) on the VLM-chain rows, 'vlm_2d_ratio' counted as the
     2d_ratio it is: clip >= 20 %, uncut run >= 6 frames / 0.25 s / 0.3 sizes, own mpp reproduced within 1 %, run cleaner
     than the whole track, no shared prior object, dist2 target not cut; rows already patched (answer_source patch*) left.
Segmented copies (answer_source var_ans_plain) follow through submit_v3's patch_copy, as rul439's own rows did.
Labels only evaluate.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul090/rul090.py [SUB=sub_xt_Nzz6]
(evaluation incl. the patch_copy follow of var_ans_plain copies: c51uncutv_eval.py, called at the end)
writes cache/tmp/ag_c51uncutv_patch_{test,val}.json"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))  # repo root (QP_ROOT overrides)
HERE = f"{R}/analysis/v3/ag/rul090"
GEN = f"{R}/cache/tmp/ag_c51uncutv_diag_vlm_gen.py"     # copyrules change 2 (was HERE/diag_vlm.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncut_geot import geo_t, raw_path  # noqa: E402
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz6"
TAG = "rul090"
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)


def rawown(split):
    r = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    # copyrules change 4: preds/<split>_geo_v3v_rntrj5sQ_raw.csv does not exist any more (checked 2026-10-17, find over /).
    # It is geo/arbitrate_v2.py's output for --meas_tag _v3v_rntrj5sQ; the columns used here are its meas_* = the measurement
    # csv geo/out/meas_<split>_v2_v3v_rntrj5sQ.csv (meas_value = value, meas_why = why, meas_kind = kind, meas_<c> = c) and
    # geo_value = value where ok. The 2-decade sanity rejection of arbitrate_v2 is not rebuilt: the assert below checks that
    # geo_value equals the final raw on every VLM-chain row (312 test / 15 val rows, all equal, checked 2026-10-17).
    m_ = pd.read_csv(f"{R}/geo/out/meas_{split}_v2_v3v_rntrj5sQ.csv").set_index("id").reindex(r.index)
    v = pd.DataFrame(index=r.index)
    v["geo_value"] = np.where(m_.ok.fillna(False).astype(bool), m_.value, np.nan)
    v["meas_value"] = m_.value; v["meas_why"] = m_.why; v["meas_kind"] = m_.kind
    for c in ("grav", "gap", "f", "mpp", "owlvote", "nseg", "colswap"):
        v["meas_" + c] = m_[c] if c in m_.columns else np.nan
    vl = r.merge_src.astype(str).isin(["v", "xg_v"])
    assert np.allclose(pd.to_numeric(r.geo_value[vl]), pd.to_numeric(v.geo_value[vl]), rtol=1e-9)
    for c in ("meas_value", "meas_mpp", "meas_f", "meas_why", "meas_grav", "meas_gap", "meas_kind", "meas_owlvote", "meas_nseg", "meas_colswap"):
        r.loc[vl, c] = v.loc[vl, c]
    r.loc[vl, "meas_tracks"] = "v1"
    r.to_csv(f"{R}/cache/tmp/ag_{TAG}_rawown_{split}.csv")
    return set(int(x) for x in r.index[vl]), r


def make_diag():
    s = open(f"{R}/analysis/v3/ag/rul439/diag.py").read()
    reps = [('f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv"', f'f"{{R}}/cache/tmp/ag_{TAG}_rawown_{{SPLIT}}.csv"'),
            ('f"{R}/cache/tmp/ag_uncut2_diag_{SPLIT}.csv"', f'f"{{R}}/cache/tmp/ag_{TAG}_diag_{{SPLIT}}.csv"'),
            ("for v, recs in byv.items():\n",
             'VLMV_ = set(raw.video_id[raw.merge_src.astype(str).isin(["v", "xg_v"])])\nfor v, recs in byv.items():\n    if v not in VLMV_:\n        continue\n')]
    for a, b in reps:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    open(GEN, "w").write(s)


def load_rule():
    s = open(f"{R}/analysis/v3/ag/rul439/rule.py").read()
    ns = {"__name__": "c51uncutv_rule", "__file__": f"{R}/analysis/v3/ag/rul439/rule.py"}   # copyrules change 1
    old = sys.argv; sys.argv = ["x", SUB]
    try:
        exec(compile(s[:s.index("\nout = {}")], "rul439/rule.py", "exec"), ns)
    finally:
        sys.argv = old
    return ns


def run(split, ns):
    VL, raw = rawown(split)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    p = subprocess.run([sys.executable, GEN, split], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    if os.environ.get("COPYRULES_KEEP"):             # analysis only: keep a copy of the diag csv outside the shadow
        import shutil
        os.makedirs(os.environ["COPYRULES_KEEP"], exist_ok=True)
        shutil.copy(f"{R}/cache/tmp/ag_{TAG}_diag_{split}.csv", os.environ["COPYRULES_KEEP"])
    D = pd.read_csv(f"{R}/cache/tmp/ag_{TAG}_diag_{split}.csv", index_col=0)
    D = D[D.index.isin(VL)].copy()
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    for c_ in ("run_n", "run_dur", "cv_run", "chord_run", "run_t0", "run_t1", "run_disp", "sh_id", "sh_d", "sh_dup", "tc1", "tc2", "tm1", "tm2",
               "tk1", "tk2", "clip", "mpp_err", "cv_full", "size", "old_px", "pkey", "mode", "qk"):
        if c_ not in D:
            D[c_] = np.nan
    D["ok2d"] = D["mode"].astype(str).str.replace("vlm_", "", regex=False).str.startswith("2d_ratio")
    D["f_clip"] = D.ok2d & (D["clip"] >= ns["CLIP"])
    D["f_run"] = D.f_clip & (D.run_n >= ns["NMIN"]) & (D.run_dur >= ns["DMIN"] - 1e-9) & (D.run_disp >= ns["DISPF"] * D["size"])
    D["f_mpp"] = D.f_run & (D.mpp_err.abs() <= ns["MPPTOL"])
    D["f_clean"] = D.f_mpp & (D.cv_run <= D.cv_full)
    D["f_shared"] = D.sh_id.notna() | D.sh_d.notna()
    D["f_tcut"] = (D.qk == "dist2") & (D.tc1.notna() | D.tc2.notna()) & ((D.tc1.astype(str) != "ok") | (D.tc2.astype(str) != "ok"))
    D["ratio"] = D.old_px / D.chord_run
    D = D.join(det[[c_ for c_ in ("answer", "answer_source", "g", "cs", "cat", "kind") if c_ in det]].rename(columns={"cat": "cat_d", "kind": "kind_d"}))
    graw = pd.read_csv(raw_path(R, split, GEO_T), index_col=0).geo_value      # copyrules change 3 (was the rj5sQ2 raw)
    D["g_same"] = (D.g / pd.to_numeric(graw.reindex(D.index)) - 1).abs() < 1e-4
    P, log = {}, []
    for i, r in D[D.f_clean].iterrows():
        kind = r.kind_d if isinstance(r.get("kind_d"), str) else r.qk     # as rul439/rule.py: the detail kind, else the parse kind
        if not r.g_same or not (np.isfinite(r.g) and r.g > 0):
            log.append((i, "detail g != raw geo")); continue
        if r.f_shared:
            log.append((i, "SKIP shared prior track")); continue
        if r.f_tcut:
            log.append((i, "SKIP dist2 target cut at query time")); continue
        if str(r.answer_source).startswith("patch"):
            log.append((i, f"already {r.answer_source}")); continue
        gn = float(r.g * r.ratio)
        r2 = r.copy(); r2["cat"] = r.cat_d
        v, how = ns["fuse"](r2, gn, kind)
        log.append((i, f"{r.answer_source}: {how} -> {v}"))
        if v is None or abs(v / r.answer - 1) < 1e-6:
            continue
        P[str(int(i))] = dict(value=float(v), why=(f"{TAG}: rul439 on the VLM track the row's geo used (v3ntmv, own mpp, mpp_err {r.mpp_err:+.4f}); "
                                                   f"prior speed px {r.old_px:.1f}->{r.chord_run:.1f} on uncut run {r.run_t0:.2f}-{r.run_t1:.2f}s "
                                                   f"(border-cut {r['clip']:.0%}); g {r.g:.4g}->{gn:.4g}; {how}"))
    return D, P, log


if __name__ == "__main__":
    sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
    from lab_rows import labelled
    GEO_T, n_ = geo_t(R, SUB)
    print(f"base {SUB}: geo table {GEO_T} (rows with detail g == table geo: {n_})")
    make_diag()
    ns = load_rule()
    out = {}
    for split in ("test", "val"):
        D, P, log = run(split, ns)
        out[split] = P
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        print(f"== {split}: VLM-chain rows with a time-less speed prior in 2D ratio {int(D.ok2d.sum())}; own mpp reproduced (1%) "
              f"{int((D.ok2d & (D.mpp_err.abs() <= 0.01)).sum())}; cut>=20% {int(D.f_clip.sum())}; uncut run {int(D.f_run.sum())}; "
              f"clean {int(D.f_clean.sum())}; patched {len(P)}")
        for i, s in log:
            print(f"   {i} {D.vid[i]}: {s}")
        for k, v in P.items():
            print(f"   PATCH {k}: {D.answer[int(k)]:.4g} -> {v['value']:.4g} | {v['why']}")
    L = labelled(SUB)
    print("labelled rows (val + all-237) touched:", [(s, int(i)) for s, i in zip(L.split, L.index) if str(int(i)) in out[s]])
    # copyrules: fuzzy-copy report and c51uncutv_eval.py (patch_copy emulation) removed
