"""rul117: a free-edge prior ruler for time-less non-gravity ACCEL priors and for SPEED priors that rul439 does not patch,
when the prior object's box is pinned at / entering the image border.

measure_v3.prior_kin reads 'acceleration of X = a' / 'speed of X = v' (no time) from the mask centroid of X's track. While X is
cut by the border (box within 2 px of it) the centroid follows only the visible part, so the centroid's px rate is biased
(a lifting post whose base never leaves the bottom border moves its centroid at half speed; a car entering the frame gets a
spurious centroid acceleration when it becomes fully visible). The box edge OPPOSITE the cut side (the free edge) is a rigid
point of the object for as long as it does not touch the border itself, so it moves exactly with the object.
Gates (code only): simulation 2D rows; prior kind accel / speed, t None, not gravity; raw meas_mode 2d_ratio / vlm_2d_ratio with a
finite meas_mpp that value_si / prior_kin reproduces within 1 %; prior box touching the border on >= 30 % of tracked frames;
motion axis = larger centroid travel; free edge uncut on >= 80 % of frames; fit on those frames: quadratic (a_px = 2 c2) for
accel, linear for speed after dropping rest frames (|v| < 20 % of median); >= 20 frames, residual RMS <= 1 px (<= 2 px with
>= 25 frames); k = old_px / edge_px with |ln k| >= 0.1 and k in [0.4, 1.0] for speed (a cut centroid can only be slower) and
[0.4, 2.5] for accel (a spurious entry / exit acceleration has either sign) [C66_ACCHI=1.0 gives the strict brief band].
Groups already patched by rul439 / rul001 / rul034 (/ rul090) are skipped.
Output: every row of the group whose meas_mpp equals the group's mpp within 1 %: g' = g x k, then
 - plain fusion sources: re-fused (rul439.fuse: geo kinds -> geo, sizes agree_wgm w0.7 in D2, ...);
 - c2a_mix rows: the c2_anchor trigger re-checked with g' (a class-bound violation in the video and |median log10(cs/g')| > 0.3
   over the video's D2 size rows, >= 2 rows); if it no longer fires, the standard fusion of g' and cs;
 - phys_bound rows: the standard fusion of g' and cs, then the class-bound check again (outside -> the one candidate inside);
 - kin / xrow_prior / scene_prior / other patch:* rows are left alone;
 - then the rul034 / rul033 witness gates are re-run with g' (receivers listed in cache/tmp/ag_c24{tat,sz}_X.pkl);
 - _segmented twins whose answer is a copy of a changed plain row get the new value.
writes cache/tmp/ag_c66freeedge_patch_{test,val}.json (FINAL answers)."""
import sys, json, math, os, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
import kinematic as K_
from lab_rows import labelled
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
SUB = os.environ.get("C66_SUB", "sub_xt_Nzz15")
TAG = os.environ.get("C66_TAG", "rul117")
ACCHI = float(os.environ.get("C66_ACCHI", "2.5"))
WRITE = os.environ.get("C66_WRITE", "1") == "1"
CUT, FREE, NMIN, RMS1, RMS2, N2, LNK, KLO, MPPTOL = 0.3, 0.8, 20, 1.0, 2.0, 25, 0.1, 0.4, 0.01
BAND = math.log(2.0)
SKIPTAGS = ("patch:rul439", "patch:rul001", "patch:rul034", "patch:rul090")
K_.EXTRA_BOUNDS = True; K_.EXTRA2 = True; K_.EXTRA3 = True
K_.SIZE3 = [x_ for x_ in K_.SIZE3 if not any(w_ in x_[0] for w_ in ("pepperoni", "rabbit"))]   # extra3clean
GEOK = {"dist2", "avgspeed"}
PLAIN = {"agree_gm", "agree_wgm", "dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid", "dis_size_gm",
         "dis_size_low_gm", "dis_caw_routed", "caw_nogeo", "caw_c1"}
sys.path.insert(0, f"{R}/analysis/v3/ag/rul439")
import importlib.util
_sp = importlib.util.spec_from_file_location("u2", f"{R}/analysis/v3/ag/rul439/rule.py")
# rul439.fuse is copied (importing rule.py would run it)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid"}
GM = {"dis_size_gm", "dis_size_low_gm"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}


def agree_w(kind, cat):
    if kind == "size":
        return 0.7 if cat == "D2" else 0.65
    return 0.5


def fuse(s, a, g, c, gn, kind, cat):
    """rul439.fuse (verbatim logic)"""
    if s.startswith("mo_guard:"):
        if s[9:] in GEO and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    in_old = np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if kind in GEOK:
        if s in AGREE or s in GEO or s in GM:
            return gn, (f"{s} -> in 2x band, geo kind -> geo" if in_new else f"{s} -> outside 2x band -> geo")
        if s in CAWD and in_new and not in_old:
            return gn, f"{s} entered 2x band, geo kind -> geo"
        return None, s
    if s in AGREE:
        if not in_new:
            return gn, "left 2x band -> geo"
        lg = math.log(g / c)
        w = math.log(a / c) / lg if abs(lg) > 1e-6 else agree_w(kind, cat)
        if not (0.3 <= w <= 0.9):
            w = agree_w(kind, cat)
        return a * (gn / g) ** w, f"{s} w{w:.2f}"
    if s in GEO:
        if s == "dis_geo" and in_new:
            w = agree_w(kind, cat)
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return gn, s
    if s in GM:
        if in_new:
            w = agree_w(kind, cat)
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return math.sqrt(gn * c), s
    if s in CAWD:
        if in_new and not in_old:
            w = agree_w(kind, cat)
            return gn ** w * c ** (1 - w), f"{s} entered 2x band -> gm w{w:.2f}"
        return None, s
    return None, s


def std_fuse(gn, c, kind, cat, mode):
    """the base fusion of a fresh geo value: in the 2x band geo kinds -> geo, sizes agree_wgm, else agree_gm; outside it the strong
    2d_ratio mode keeps geo; vlm_2d_ratio (not strong) keeps CaW"""
    if not (np.isfinite(c) and c > 0):
        return gn, "geo (no CaW)"
    if abs(math.log(gn / c)) <= BAND:
        if kind in GEOK:
            return gn, "agree_geo_kind"
        w = agree_w(kind, cat)
        return gn ** w * c ** (1 - w), f"agree_{'wgm' if w != 0.5 else 'gm'} w{w:.2f}"
    if mode == "2d_ratio":
        return gn, "dis_geo"
    return c, "dis_caw (weak mode)"


def bound_fix(v, gn, c, q, prior, kind):
    b_ = K_.class_bound(q, prior, kind)
    if b_ is None:
        return v, ""
    f_ = K_.q_si(q)
    inside = lambda x: np.isfinite(x) and x > 0 and b_[0] <= x * f_ <= b_[1]
    if inside(v):
        return v, ""
    cand = [x for x in (gn, c) if inside(x)]
    return (cand[0], " -> phys_bound") if len(cand) == 1 else (v, " (outside bound, no single candidate)")


def c2a_fires(vrows, gnew):
    """consistency.c2_apply trigger with thr 0.3, nmin 2 (mix) + submit_v3's class-bound violation gate, for one video"""
    viol, lr = False, []
    for i, r in vrows.iterrows():
        gi = gnew.get(i, r.g)
        b_ = K_.class_bound(str(r.question), str(r.prior), r["kind"])
        if b_ is not None and np.isfinite(gi) and gi > 0 and not (b_[0] <= gi * K_.q_si(str(r.question)) <= b_[1]):
            viol = True
        if r["kind"] == "size" and r["cat"] == "D2" and np.isfinite(gi) and gi > 0 and np.isfinite(r.cs) and r.cs > 0:
            lr.append(math.log10(r.cs / gi))
    med = float(np.median(lr)) if lr else 0.0
    return bool(viol and len(lr) >= 2 and abs(med) > 0.3), med, viol


out = {}
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_c66freeedge_diag_{split}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    det["question"] = raw.question.reindex(det.index).astype(str)
    det["prior"] = raw.ground_truth_prior.reindex(det.index).astype(str)
    det["rmode"] = raw.meas_mode.reindex(det.index).astype(str)
    det["mpp"] = raw.meas_mpp.reindex(det.index)
    det["g_raw"] = raw.geo_value.reindex(det.index)
    if "mode" not in D or "edge_rate" not in D:
        out[split] = ({}, det, pd.DataFrame()); print(f"[{split}] no candidates"); continue
    G = D[D.note.isna()].drop_duplicates(["vid", "praw"]).copy()
    G["f_cut"] = G["cut"] >= CUT
    G["f_free"] = G.f_cut & (G.free_ok >= FREE)
    G["f_fit"] = G.f_free & (G.nfit >= NMIN) & ((G.edge_rms <= RMS1) | ((G.nfit >= N2) & (G.edge_rms <= RMS2)))
    G["f_mpp"] = G.f_fit & (G.mpp_err.abs() <= MPPTOL)
    hi = np.where(G.pkind == "accel", ACCHI, 1.0)
    G["f_k"] = G.f_mpp & (np.log(G.k).abs() >= LNK) & (G.k >= KLO) & (G.k <= hi)
    # skip groups patched by rul439 / rul001 / rul034: any row of the same video and prior carries such a source
    det_src = det.answer_source.astype(str).str.replace("mo_guard:", "", regex=False)
    sk = []
    for i, r in G.iterrows():
        rr = det[(det.video_id == r.vid) & (det.prior == raw.ground_truth_prior[i])]
        sk.append(",".join(sorted(set(s for s in det_src[rr.index] if s in SKIPTAGS))))
    G["skip"] = sk
    for c_ in ("sh_id", "sh_d"):
        G[c_] = G[c_].fillna("").astype(str) if c_ in G else ""
    G["f_shared"] = (G.sh_id != "") | (G.sh_d != "")          # rul439 fix 2: the prior's SAM2 object is shared with a differently named query
    G["fire"] = G.f_k & (G.skip == "") & ~G.f_shared
    print(f"[{split}] groups {len(G)}: cut>=30% {int(G.f_cut.sum())}, free edge ok {int(G.f_free.sum())}, fit ok {int(G.f_fit.sum())}, "
          f"mpp ok {int(G.f_mpp.sum())}, k ok {int(G.f_k.sum())}, not skipped {int(G.fire.sum())}")
    print(G[G.f_cut][["vid", "pkind", "praw", "mode", "cut", "ax", "cutside", "free_ok", "nfit", "edge_rms", "old_px", "edge_rate", "cen_uncut_rate",
                      "k", "mpp_err", "f_free", "f_fit", "f_mpp", "f_k", "skip", "sh_id", "sh_d", "fire"]].round(3).to_string())
    P, rows = {}, []
    gnew = {}
    for gi_, gr in G[G.fire].iterrows():
        prior = raw.ground_truth_prior[gi_]
        V = det[(det.video_id == gr.vid)].copy()
        mem = V[(V.prior == prior) & V.rmode.isin(["2d_ratio", "vlm_2d_ratio"]) & ((V.mpp / gr.mpp - 1).abs() <= MPPTOL)]
        for i, r in mem.iterrows():
            if np.isfinite(r.g) and r.g > 0 and np.isfinite(r.g_raw) and abs(r.g / r.g_raw - 1) < 1e-4:
                gnew[i] = float(r.g * gr.k)
        fired, med, viol = c2a_fires(V, gnew)
        for i, r in mem.iterrows():
            if i not in gnew:
                rows.append((split, i, gr.vid, r.answer_source, "g != raw geo / no geo, skipped")); continue
            s = str(r.answer_source); s0 = s.replace("mo_guard:", "")
            gn, c, a = gnew[i], float(r.cs) if np.isfinite(r.cs) else np.nan, float(r.answer)
            if s0 in PLAIN:
                v, how = fuse(s, a, float(r.g), c, gn, r["kind"], r["cat"])
                if v is not None:
                    v, hb = bound_fix(v, gn, c, str(r.question), r.prior, r["kind"]); how += hb
            elif s0 == "phys_bound":
                v, how = std_fuse(gn, c, r["kind"], r["cat"], r.rmode)
                v, hb = bound_fix(v, gn, c, str(r.question), r.prior, r["kind"]); how = "phys_bound -> " + how + hb
            elif s0.startswith("c2a_"):
                if fired:
                    v, how = None, f"c2a still fires (med {med:.2f}) -> unchanged"
                else:
                    v, how = std_fuse(gn, c, r["kind"], r["cat"], r.rmode)
                    v, hb = bound_fix(v, gn, c, str(r.question), r.prior, r["kind"])
                    how = f"c2a no longer fires (viol {viol}, med {med:.2f}) -> " + how + hb
            else:
                v, how = None, f"{s} left alone"
            rows.append((split, i, gr.vid, s, how))
            if v is None or abs(v / a - 1) < 1e-6:
                continue
            P[str(int(i))] = dict(value=float(v), why=f"{TAG}: prior {gr.pkind} px {gr.old_px:.1f} -> free {gr.cutside}-opposite edge "
                                  f"{abs(gr.edge_rate):.1f} (border {gr['cut']:.0%}, rms {gr.edge_rms:.2f}, n {int(gr.nfit)}); k {gr.k:.3f}; "
                                  f"g {r.g:.4g}->{gn:.4g}; {how}; was {a:.4g} ({s})")
    # ---- rul034 / rul033 witness gates with the new geo
    try:
        XT = pd.read_pickle(f"{R}/cache/tmp/ag_c24tat_X.pkl")
        XT = XT[XT.sp == split]
        for (vid, obj), Z in XT.groupby(["vid", "obj"]):
            if not any(int(i) in gnew for i in Z.rid):
                continue
            gw = {int(i): (lambda g_: np.isfinite(g_) and g_ > 0 and v_ / 1.25 <= g_ <= 1.25 * v_)(gnew.get(int(i), g0))
                  for i, g0, v_ in zip(Z.rid, Z.g, Z.v)}
            sibbad = any((np.isfinite(gnew.get(int(i), g0)) and gnew.get(int(i), g0) > 0) and not gw[int(i)] for i, g0 in zip(Z.rid, Z.g))
            for r in Z.itertuples():
                i = int(r.rid)
                cur = P[str(i)]["value"] if str(i) in P else float(det.answer[i])
                fire = r.agree and not r.tacc and gw[i] and not sibbad and not abs(cur - r.v) <= 0.02 * r.v and not r.covd and \
                    not str(det.answer_source[i]).startswith("patch:")
                rows.append((split, i, vid, "rul034-rul407", f"gw {gw[i]} sibbad {sibbad} -> {'FIRE %.4g' % r.v if fire else 'no'}"))
                if fire:
                    P[str(i)] = dict(value=float(r.v), why=f"{TAG} + rul034 gate with new geo {gnew.get(i, r.g):.4g}: template ({r.srcb}) states "
                                     f"{r.raw0!r}; was {float(det.answer[i]):.4g} ({det.answer_source[i]})")
        XS = pd.read_pickle(f"{R}/cache/tmp/ag_c24sz_X.pkl")
        XS = XS[(XS.sp == split) & XS.rid.isin(list(gnew))]
        for r in XS.itertuples():
            i = int(r.rid); g_ = gnew[i]
            wit = (np.isfinite(g_) and r.v / 1.25 <= g_ <= 1.25 * r.v) or bool(r.cw)
            fire = r.agree and wit and not r.have2 and not r.covd and not r.patched and r.scene
            rows.append((split, i, r.vid, "rul033-rul407", f"wit {wit} -> {'FIRE' if fire else 'no'}"))
            if fire:
                P[str(i)] = dict(value=float(r.v), why=f"{TAG} + rul033 gate with new geo {g_:.4g}: template states {r.raw0!r}")
    except FileNotFoundError as e:
        print("c24 pickles missing", e)
    # ---- _segmented twins that copy a changed plain row
    det["question"] = raw.question.reindex(det.index).astype(str)
    det["ql"] = det.question.str.strip().str.lower()
    for k in list(P):
        i = int(k)
        vid = str(det.video_id[i])
        if vid.endswith("_segmented"):
            continue
        tw = det[(det.video_id == vid + "_segmented") & (det.ql == det.ql[i]) & (det.prior == det.prior[i])]
        for j, rj in tw.iterrows():
            if abs(float(rj.answer) / float(det.answer[i]) - 1) < 1e-6 and str(j) not in P:
                P[str(int(j))] = dict(value=P[k]["value"], why=f"{TAG}: copy of plain row {i} ({rj.answer_source})")
                rows.append((split, j, vid + "_segmented", rj.answer_source, f"twin copy of {i}"))
    if WRITE:
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    RR = pd.DataFrame(rows, columns=["split", "id", "vid", "src", "how"])
    print(RR.to_string())
    out[split] = (P, det, G)

# ---------------------------------------------------------------- evaluation
L = labelled(SUB)
for split in ("val", "test"):
    P, det, G = out[split]
    Ls = L[(L.split == split) & L.index.isin([int(k) for k in P])]
    print(f"labelled {split} rows changed: {len(Ls)}")
    if len(Ls):
        new = np.array([P[str(i)]["value"] for i in Ls.index])
        print(pd.DataFrame({"vid": Ls.video_id, "y": Ls.y, "old": Ls.answer, "new": new, "m_old": Ls.m, "m_new": A.row_mra(new, Ls.y.values)}).to_string())
P, det, G = out["test"]
if P:
    ch = det.loc[[int(k) for k in P]].copy()
    ch["new"] = [P[str(i)]["value"] for i in ch.index]
    print("\ntest rows changed by category:", ch["cat"].value_counts().to_dict())
    print(ch[["video_id", "question", "answer_source", "kind", "answer", "new", "g", "cs"]].round(4).to_string())
    for k, v in P.items():
        print(k, v["why"])
