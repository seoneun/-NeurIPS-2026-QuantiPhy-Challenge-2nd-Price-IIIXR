"""rul433: tedge (analysis/v3/ag/tedge) with the reproduction gate relaxed from 1% to 3%; keeps rul432's sruler_geo exclusion.

Background (see analysis/v3/ag/tedge/rule.py): while the target's box touches the image border at the query time, the SAM2
mask centroid follows only the visible part and moves at about half the true speed; tedge re-reads the cut axis from the
FREE edge (the edge opposite the border) over the same fit frames and rescales the row's own geo: g_new = g x new_px / old_px.

The reproduction gate checks that the re-measured track/fit is the one measure_v3 used: old_px x meas_mpp must reproduce the
raw meas_value. tedge used 1%. Why rows 326/329 missed it: tedge/diag.py re-measures rows whose raw says meas_tracks=v1 on the
v3ntmv track files, but the raw run (TRACK_VER=v2sc) read those videos from geo/out/tracks_{split}/ (the loader's fallback when
a video has no v2sc / v2 file). rul433/diag_own.py repeats the diag on the raw's own track file: rep_err is then 0.000 on both
rows (and on 51 of the 58 test v1 rows that failed the 1% gate).

MODE rep3 (default; the rule as specified): a row changes only if rul432 skipped it because of the gate, i.e.
  - with tedge's diag, it passes every rul432 gate at 3% but not at 1% (1% < |rep_err| <= REP_TOL), and
  - the own-track re-measurement (diag_own) passes every rul432 gate at 1% (|rep_err| <= 1%, free edge matches the uncut-run
    centroid within x1.25 and closer than the centroid did, change >= 3%); the value uses the own-track ratio.
MODE own (variant, reported only): every row rul432 rejected at the gate whose own-track re-measurement passes all gates at 1%.
Unchanged from rul432: 2D ratio modes only (2d_ratio, vlm_2d_ratio); g == raw geo; dynamic priors read from cut boxes skipped;
sruler_geo rows skipped (also under mo_guard); rows already patched (answer_source patch*, incl. patch:rul432) skipped.
Fusion: tedge's (the base build's), see fuse(). Labels only evaluate.
usage: python analysis/v3/ag/rul433/rule.py [SUB=sub_xt_Nzh] [REP_TOL=0.03] [OLD_TOL=0.01] [TAG=rul433] [MODE=rep3|own]
reads cache/tmp/ag_tedge_diag_{split}.csv (tedge/diag.py) and cache/tmp/ag_tedge3_diagown_{split}.csv (rul433/diag_own.py)
writes cache/tmp/ag_<TAG>_patch_{test,val}.json"""
import sys, json, math, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzh"
REP_TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
OLD_TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
TAG = sys.argv[4] if len(sys.argv) > 4 else "rul433"
MODE = sys.argv[5] if len(sys.argv) > 5 else "rep3"
MATCH, MINCH = 1.25, 0.03
RUN_GAP = 0.5
PERP = 0.3
MODES = {"2d_ratio", "vlm_2d_ratio"}
STRONG = {"2d_ratio"}
BAND = math.log(2.0)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid"}      # sruler_geo removed (rul432)
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}


def fuse(r, gn, kind, strong):
    """new answer for a re-measured row (None = leave it) and how (tedge's fusion, sruler_geo excluded)"""
    s, a, g = str(r.answer_source), float(r.answer), float(r.g)
    if "sruler" in s:
        return None, f"{s} (sruler_geo excluded, rul432)"
    if s.startswith("mo_guard:"):
        if s[9:] in GEO and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    c = float(r.cs) if np.isfinite(r.cs) else np.nan
    in_old = np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if s == "geo_cawinvalid":
        return gn, s
    if kind in GEOK:
        if s in AGREE or s in GEO:
            if in_new:
                return gn, f"{s} -> in 2x band, geo kind -> geo (agree_geo_kind)"
            return (gn, f"{s} -> outside 2x band, strong -> geo") if strong else (None, f"{s} -> outside band, weak -> CaW (skip)")
        if s in CAWD and in_new and not in_old:
            return gn, f"{s} entered 2x band, geo kind -> geo (agree_geo_kind)"
        return None, s
    # speed
    if s in AGREE:
        if not in_new:
            return (gn, "left 2x band, strong -> geo") if strong else (None, "left 2x band, weak -> CaW (skip)")
        lg = math.log(g / c)
        w = math.log(a / c) / lg if abs(lg) > 1e-6 else 0.5
        if not (0.3 <= w <= 0.9):
            w = 0.5
        return a * (gn / g) ** w, f"{s} w{w:.2f}"
    if s in GEO:
        if in_new and s == "dis_geo":
            return math.sqrt(gn * c), "entered 2x band -> agree_gm"
        return gn, s + (" -> outside 2x band, strong -> geo" if strong and not in_new else "")
    if s in CAWD:
        if in_new and not in_old:
            return math.sqrt(gn * c), f"{s} entered 2x band -> agree_gm"
        return None, s
    return None, s


def gates(D, det, raw, tol):
    D = D[D.ncut.fillna(0) > 0].copy()
    D = D.join(det[[c for c in ["question", "answer", "answer_source", "g", "cs"] if c in det]])
    D["g_raw"] = raw.geo_value.reindex(D.index)
    D["r"] = D.new_px / D.old_px
    for c_ in ("pers", "run2_px", "run2_gap", "run2_side", "run2_vx", "run2_vy"):
        if c_ not in D:
            D[c_] = np.nan
    D["pers"] = D.pers.fillna("").astype(str)
    hasP = D.pers != ""
    px_ = D.pers.str.contains("L|R"); py_ = D.pers.str.contains("T|B")
    perp = ((~px_ | (D.run2_vx.abs() <= PERP * D.run2_px)) & (~py_ | (D.run2_vy.abs() <= PERP * D.run2_px)) & ~(px_ & py_))
    D["run_src"] = np.where(hasP, np.where(perp, "run2_perp", "run2_notperp"), "run")
    for a_, b_ in (("run_px", "run2_px"), ("run_gap", "run2_gap"), ("run_side", "run2_side")):
        D[a_] = D[a_].where(~hasP, D[b_].where(perp, np.nan))
    D["e_old"] = np.log(D.old_px / D.run_px); D["e_new"] = np.log(D.new_px / D.run_px)
    D["f_mode"] = D["mode"].isin(MODES)
    D["f_edge"] = D.f_mode & D.new_px.notna() & (D.new_px > 0) & D.old_px.notna() & (D.old_px > 0)
    if "rep_err" not in D:
        D["rep_err"] = np.nan
    D["f_graw"] = D.f_edge & ((D.g / D.g_raw - 1).abs() < 1e-4)
    D["f_g"] = D.f_graw & (D.rep_err.abs() <= tol)
    D["f_run"] = D.f_g & D.run_px.notna() & (D.run_gap <= RUN_GAP + 1e-9)
    D["f_match"] = D.f_run & (D.e_new.abs() <= math.log(MATCH)) & (D.e_new.abs() < D.e_old.abs())
    D["f_ch"] = D.f_match & (np.log(D.r).abs() >= MINCH)
    D["f_patch"] = D.answer_source.astype(str).str.startswith("patch")
    for c_ in ("p_kind", "p_t", "p_t1", "p_ncut", "p_nw", "p_same"):
        if c_ not in D:
            D[c_] = np.nan
    timed = D.p_t.notna() | D.p_t1.notna()
    pfrac = D.p_ncut / D.p_nw.where(D.p_nw > 0)
    D["f_pcut"] = D.p_kind.notna() & (D.p_ncut >= 0) & ((timed & (D.p_ncut > 0)) | (~timed & (pfrac >= 0.1)))
    return D


L = labelled(SUB)
out = {}
for split in ("test", "val"):
    D0 = pd.read_csv(f"{R}/cache/tmp/ag_tedge_diag_{split}.csv", index_col=0)
    O0 = pd.read_csv(f"{R}/cache/tmp/ag_tedge3_diagown_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    D1 = gates(D0, det, raw, OLD_TOL)       # rul432's gates (tedge diag, 1%)
    D3 = gates(D0, det, raw, REP_TOL)       # relaxed gate (tedge diag, 3%)
    Do = gates(O0, det, raw, OLD_TOL)       # own-track re-measurement, 1%
    # rows rul432 rejected AT THE GATE: free edge, 2D ratio mode, g == raw geo, but |rep_err| > 1% (or not reproducible)
    gate_rej = D1.index[D1.f_graw & ~(D1.rep_err.abs() <= OLD_TOL)]
    if MODE == "rep3":
        cand = D3.index[D3.f_ch & (D3.rep_err.abs() > OLD_TOL)]
    else:
        cand = Do.index[Do.f_ch].intersection(gate_rej)
    P, rows = {}, []
    for i in cand:
        if i not in gate_rej:
            rows.append((i, D1.vid[i], D1.answer_source[i], np.nan, np.nan, "SKIP not a gate rejection in rul432")); continue
        rep0 = float(D1.rep_err[i]) if np.isfinite(D1.rep_err[i]) else np.nan
        if i not in Do.index or not bool(Do.f_ch[i]):
            why_ = "missing" if i not in Do.index else ("rep" if not Do.f_g[i] else ("run" if not Do.f_run[i] else ("match" if not Do.f_match[i] else "minch")))
            rows.append((i, D1.vid[i], D1.answer_source[i], rep0, Do.rep_err.get(i, np.nan), f"SKIP own-track re-measurement fails the 1% gates ({why_})")); continue
        r = Do.loc[i]
        if r.f_patch:
            rows.append((i, r.vid, r.answer_source, rep0, r.rep_err, "SKIP already patched")); continue
        if r.f_pcut:
            rows.append((i, r.vid, r.answer_source, rep0, r.rep_err, f"SKIP prior {r.p_kind} read from cut boxes ({int(r.p_ncut)}/{int(r.p_nw)})")); continue
        gn = float(r.g * r.r)
        v, how = fuse(r, gn, r.qk, r["mode"] in STRONG)
        rows.append((i, r.vid, r.answer_source, rep0, r.rep_err, how))
        if v is None or abs(v / r.answer - 1) < 1e-6:
            continue
        P[str(int(i))] = dict(value=float(v), why=f"{TAG}: rul432 skipped at the 1% repro gate (tedge diag rep_err {rep0:+.3f}, v3ntmv tracks); on the raw's own "
                                                   f"track rep_err {r.rep_err:+.3f}; target box border-cut ({r.sidesw}) in {int(r.ncut)}/{int(r.nw)} fit frames; "
                                                   f"free-edge px speed {r.old_px:.1f}->{r.new_px:.1f} (uncut-run centroid {r.run_px:.1f}, {r.run_side} {r.run_gap:.2f}s"
                                                   f"{'; side ' + r.pers + ' touched all along, motion perpendicular' if r.pers else ''}); g {r.g:.4g}->{gn:.4g}; {how}")
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    out[split] = (Do, P, det, D1)
    print(f"[{split}] MODE {MODE}: cut-at-query rows {len(D1)}; rul432 gate rejections {len(gate_rej)}; candidates {len(cand)}; patched {len(P)} | "
          f"all-gates pass: tedge diag 1% {int(D1.f_ch.sum())}, 3% {int(D3.f_ch.sum())}; own-track 1% {int(Do.f_ch.sum())}")
    if rows:
        print(pd.DataFrame(rows, columns=["id", "vid", "src", "rep_tedge", "rep_own", "how"]).round(4).to_string(index=False))

# ---------------------------------------------------------------- evaluation
print("\n=== labelled rows (val + all-237 test copies) whose target is border-cut at the query time (own-track diag) ===")
for split in ("val", "test"):
    D, P, det, D1 = out[split]
    Ls = L[L.split == split]
    J = D.join(Ls[["y", "m", "mg", "mc"]], how="inner")
    if len(J):
        J["rep_tedge"] = D1.rep_err.reindex(J.index)
        J["new"] = [P[str(i)]["value"] if str(i) in P else np.nan for i in J.index]
        J["g_new"] = J.g * J.r
        J["mg_new"] = A.row_mra(J.g_new.fillna(1).values, J.y.values)
        J["m_new"] = np.where(J.new.notna(), A.row_mra(J.new.fillna(1).values, J.y.values), J.m)
        print(f"[{split}] {len(J)} labelled cut rows; patched {int(J.new.notna().sum())}")
        print(J[["vid", "cat", "qk", "mode", "sidesw", "rep_tedge", "rep_err", "r", "e_old", "e_new", "run_gap", "f_ch", "answer_source", "y", "answer",
                 "new", "m", "m_new", "g", "g_new", "mg", "mg_new"]].round(3).to_string())
for nm, LL, P in (("val", L[L.split == "val"], out["val"][1]), ("all-237 test copies", L[L.split == "test"], out["test"][1])):
    new = LL.answer.copy()
    for k, v in P.items():
        if int(k) in new.index:
            new[int(k)] = v["value"]
    m0 = A.row_mra(LL.answer.values, LL.y.values); m1 = A.row_mra(new.values, LL.y.values)
    print(f"{nm}: rows changed {int((new != LL.answer).sum())}; macro {A.macro(pd.Series(m0, index=LL.index), LL.cat)[0]:.4f} -> "
          f"{A.macro(pd.Series(m1, index=LL.index), LL.cat)[0]:.4f}")

D, P, det, D1 = out["test"]
ch = det.loc[[int(k) for k in P]].copy()
ch["new"] = [P[str(i)]["value"] for i in ch.index]
print("\ntest rows changed by category:", ch.cat.value_counts().to_dict(), " by source:", ch.answer_source.value_counts().to_dict())
if len(ch):
    X = D.loc[ch.index]
    print(pd.concat([X[["vid", "qk", "qt", "mode", "sidesw", "rep_err", "ncut", "nw", "old_px", "new_px", "run_px", "run_side", "run_gap"]],
                     ch[["answer_source", "answer", "new", "g", "cs"]]], axis=1).round(3).to_string())
for i in ch.index:
    print(i, ch.loc[i, "video_id"], "|", ch.loc[i, "question"], "|", round(ch.loc[i, "answer"], 4), "->", round(ch.loc[i, "new"], 4), "|", P[str(i)]["why"])
# overlap with the 46 patch files of the Nzh build
lst = open(f"{R}/cache/tmp/sr67.sh").read().split("--patch_json ")[1].split('"')[0].split(",")
hit = {}
for t in lst:
    f = f"{R}/cache/tmp/ag_{t.strip()}_patch_test.json"
    if os.path.exists(f):
        for k in json.load(open(f)):
            if k in P:
                hit.setdefault(k, []).append(t)
print(f"overlap of changed test rows with the {len(lst)} Nzh patch files:", hit or "none")
