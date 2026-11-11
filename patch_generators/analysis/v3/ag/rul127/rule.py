"""rul127: pipeline-bug fix inside the accepted tedge mechanism (rul432 / rul433 / rul066).

tedge re-reads a border-cut target's px speed from the FREE box edge and accepts it only if it matches the same object's
speed on the nearest uncut run (|log(new/run)| <= log 1.25 AND closer to it than the old centroid speed). That run speed is
a linear fit of the SAM2 mask CENTROID tr.uv. On an uncut frame the box is the whole object, so the box centre moves exactly
with the object; the mask centroid drifts when the mask's shape changes along the run (e.g. a white hull whose mask loses /
gains pixels against water foam), so it can understate the run speed and make tedge refuse a correct free-edge value.
Change: over the same uncut frames and fit window (rul127/diag.py) the run slope along the motion axis is computed from the
centroid and from the box centre (x1+x2)/2; if they differ by more than SLOPE_TOL (3%), the box-centre slope is used as the
run speed. The 4% / >= 20-frame _segmented condition of the spec is implied by the 3% one, so no video-id test is needed.
tedge's gates and output are unchanged (2D ratio modes, g == raw geo, own-track repro <= 1%, run within 0.5 s, match x1.25 and
closer than the centroid, change >= 3%, cut dynamic prior skipped, sruler skipped, patched rows skipped); value = tedge's
g x new_px / old_px with tedge's fusion. Only rows tedge currently REFUSES at the run-match gate can change.
usage: python analysis/v3/ag/rul127/rule.py [SUB=sub_xt_Nzz18] [SLOPE_TOL=0.03] [TAG=rul127]
writes cache/tmp/ag_<TAG>_patch_{test,val}.json; labels only evaluate."""
import sys, json, math, os
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz18"
SLOPE_TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
TAG = sys.argv[3] if len(sys.argv) > 3 else "rul127"
OLD_TOL = 0.01
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
    """rul433's fusion, unchanged"""
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
    """rul433's gates, unchanged"""
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
    O0 = pd.read_csv(f"{R}/cache/tmp/ag_tedge3_diagown_{split}.csv", index_col=0)
    X = pd.read_csv(f"{R}/cache/tmp/ag_c69segrun_diag_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    # sanity: the centroid fit here reproduces the diag's run / run2 speed
    for c_ in ("pers", "run2_px"):   # 10-17 clean6: rul433 diag without person rows has no pers / run2 columns -> ref = run_px
        if c_ not in O0: O0[c_] = np.nan
    ref = np.where(O0.reindex(X.index).pers.fillna("").astype(str) != "", O0.reindex(X.index).run2_px, O0.reindex(X.index).run_px)
    X["ref_px"] = ref
    ok = X.c_px.notna() & np.isfinite(X.ref_px)
    print(f"[{split}] centroid run fit reproduces tedge's run speed within 0.1%: {int(((X.c_px / X.ref_px - 1).abs()[ok] < 1e-3).sum())}/{int(ok.sum())}")
    G0 = gates(O0, det, raw, OLD_TOL)
    O1 = O0.copy()
    use = X.index[X.c_px.notna() & (X.slope_diff.abs() > SLOPE_TOL)]
    for i in use:
        col = "run2_px" if X.run_used[i] == "run2" else "run_px"
        O1.loc[i, col] = X.b_px[i]
        vc = "run2_vx" if col == "run2_px" else "run_vx"
        if X.ax[i] == "x":
            O1.loc[i, vc] = X.b_vx[i]
    G1 = gates(O1, det, raw, OLD_TOL)
    refused = G0.index[G0.f_run & ~G0.f_match]
    print(f"[{split}] cut rows {len(G0)}; accepted by tedge gates now (f_ch) {int(G0.f_ch.sum())}; refused at the run-match gate {len(refused)}; "
          f"box/centroid run slopes differ > {SLOPE_TOL:.0%} on {len(use)} cut rows ({len(set(use) & set(refused))} of them refused)")
    J = G0.loc[refused, ["vid", "cat", "qk", "mode", "sidesw", "old_px", "new_px", "run_px", "answer_source"]].join(
        X[["run_used", "ax", "c_ax", "b_ax", "slope_diff", "run_nfit", "c_px", "b_px"]])
    J["e_old0"] = G0.e_old[refused]; J["e_new0"] = G0.e_new[refused]
    J["e_old1"] = G1.e_old.reindex(refused); J["e_new1"] = G1.e_new.reindex(refused); J["f_ch1"] = G1.f_ch.reindex(refused)
    print(J.round(3).to_string())
    P, rows = {}, []
    for i in refused:
        if i not in use or not bool(G1.f_ch.get(i, False)):
            continue
        r = G1.loc[i]
        if r.f_patch:
            rows.append((i, r.vid, r.answer_source, "SKIP already patched")); continue
        if r.f_pcut:
            rows.append((i, r.vid, r.answer_source, f"SKIP prior {r.p_kind} read from cut boxes")); continue
        gn = float(r.g * r.r)
        v, how = fuse(r, gn, r.qk, r["mode"] in STRONG)
        rows.append((i, r.vid, r.answer_source, how))
        if v is None or abs(v / r.answer - 1) < 1e-6:
            continue
        x = X.loc[i]
        P[str(int(i))] = dict(value=float(v), why=f"{TAG}: tedge refused at the run-match gate because the uncut-run mask-centroid slope "
                                                    f"({x.c_ax:.1f} px/s on {x.ax}) differs {x.slope_diff:+.1%} from the box-centre slope ({x.b_ax:.1f}) over the same "
                                                    f"{int(x.run_nfit)} uncut frames; with the box-centre run speed {x.b_px:.1f} the free-edge px speed "
                                                    f"{r.old_px:.1f}->{r.new_px:.1f} passes tedge's gates; g {r.g:.4g}->{gn:.4g}; {how}")
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    out[split] = (G1, P, det)
    print(f"[{split}] flips refused -> accepted: {len(rows)}; patched {len(P)}")
    if rows:
        print(pd.DataFrame(rows, columns=["id", "vid", "src", "how"]).to_string(index=False))

print("\n=== labelled rows ===")
for nm, sp in (("val", "val"), ("all-237 test copies", "test")):
    LL = L[L.split == sp]; P = out[sp][1]
    new = LL.answer.copy()
    for k, v in P.items():
        if int(k) in new.index:
            new[int(k)] = v["value"]
    ch = new.index[new != LL.answer]
    m0 = A.row_mra(LL.answer.values, LL.y.values); m1 = A.row_mra(new.values, LL.y.values)
    print(f"{nm}: rows changed {len(ch)}; macro {A.macro(pd.Series(m0, index=LL.index), LL.cat)[0]:.4f} -> {A.macro(pd.Series(m1, index=LL.index), LL.cat)[0]:.4f}")
    for i in ch:
        print("  ", i, LL.video_id[i], LL.cat[i], "y", LL.y[i], "old", round(LL.answer[i], 4), "new", round(new[i], 4),
              "m", round(float(A.row_mra(np.array([LL.answer[i]]), np.array([LL.y[i]]))[0]), 3), "->", round(float(A.row_mra(np.array([new[i]]), np.array([LL.y[i]]))[0]), 3))
G1, P, det = out["test"]
ch = det.loc[[int(k) for k in P]].copy()
ch["new"] = [P[str(i)]["value"] for i in ch.index]
print("\ntest rows changed by category:", ch.cat.value_counts().to_dict())
for i in ch.index:
    print(i, ch.loc[i, "video_id"], "|", ch.loc[i, "question"], "|", round(ch.loc[i, "answer"], 4), "->", round(ch.loc[i, "new"], 4), "|", P[str(i)]["why"])
lst = open(f"{R}/cache/tmp/ag_c69segrun_sr102.sh").read().split("--patch_json ")[1].split('"')[0].split(",") if os.path.exists(f"{R}/cache/tmp/ag_c69segrun_sr102.sh") else []
hit = {}
for t in lst:
    f = f"{R}/cache/tmp/ag_{t.strip()}_patch_test.json"
    if os.path.exists(f):
        for k in json.load(open(f)):
            if k in P:
                hit.setdefault(k, []).append(t)
print(f"overlap of changed test rows with the {len(lst)} applied patch files:", hit or "none")
