"""rul174: tedge free-edge speed for rows tedge rejected only at its uncut-run match because the target ACCELERATES.

tedge (analysis/v3/ag/tedge) re-reads a border-cut target's speed from the free edge and accepts it only when it matches the
centroid speed on the nearest uncut run (constant-speed assumption). An object that speeds up while leaving the frame fails
that check although the free edge is right. Here the uncut run that PRECEDES the cut frames is fitted with a quadratic
x(t), y(t); when the run clearly accelerates (|v(end)| >= ACC x |v(start)|, pooled R^2 >= R2MIN) the quadratic's mean speed
over the cut window [w_t0, w_t1] (v_ext) replaces the run speed in the match: accept when |log(new/v_ext)| <= log MATCH and
new is closer to v_ext than the centroid speed was. Output g_new = g x new_px / old_px, fused with tedge's fuse().
Reads cache/tmp/ag_tedge_diag_{split}.csv read-only; writes only cache/tmp/ag_c84tedgeacc_patch_{split}.json.
f_g: as tedge (g == raw geo, rep_err <= 1%); rows whose raw has no meas_mpp (rep_err NaN) are admitted when the implied
scale g/old_px equals prior_size / (target own uncut-box height or width) within 1% (self-scaled from uncut frames, so the
scale carries no border-cut bias and g is exactly proportional to the px speed)."""
import os, sys, json, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, R + "/geo")
import analyze_caw as A
from lab_rows import labelled
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 60)
TAG = "rul174"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz27"
MATCH, ACC, R2MIN = 1.25, 1.15, 0.9
RUN_GAP, PERP, TOL = 0.5, 0.3, 2.0
MODES = {"2d_ratio", "vlm_2d_ratio"}
STRONG = {"2d_ratio"}
BAND = math.log(2.0)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid", "sruler_geo"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}


def fuse(r, gn, kind, strong):   # copied verbatim from analysis/v3/ag/tedge/rule.py
    s, a, g = str(r.answer_source), float(r.answer), float(r.g)
    if s.startswith("mo_guard:"):
        if s[9:] in GEO and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    c = float(r.cs) if np.isfinite(r.cs) else np.nan
    in_old = np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if s == "geo_cawinvalid" or s == "sruler_geo":
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
        return gn, s
    if s in CAWD:
        if in_new and not in_old:
            return math.sqrt(gn * c), f"{s} entered 2x band -> agree_gm"
        return None, s
    return None, s


def tedge_flags(D, det, raw, parse):
    """tedge gates, recomputed exactly (plus the self-scale f_g extension)"""
    D = D[D.ncut.fillna(0) > 0].copy()
    D = D.join(det[[c for c in ["question", "answer", "answer_source", "g", "cs"] if c in det]])
    D["g_raw"] = raw.geo_value.reindex(D.index)
    D["r"] = D.new_px / D.old_px
    for c_ in ("pers", "run2_px", "run2_gap", "run2_side", "run2_vx", "run2_vy", "rep_err", "hfull", "wfull"):
        if c_ not in D:
            D[c_] = np.nan
    D["pers"] = D.pers.fillna("").astype(str)
    hasP = D.pers != ""
    px_ = D.pers.str.contains("L|R"); py_ = D.pers.str.contains("T|B")
    perp = ((~px_ | (D.run2_vx.abs() <= PERP * D.run2_px)) & (~py_ | (D.run2_vy.abs() <= PERP * D.run2_px)) & ~(px_ & py_))
    for a_, b_ in (("run_px", "run2_px"), ("run_gap", "run2_gap"), ("run_side", "run2_side")):
        D[a_] = D[a_].where(~hasP, D[b_].where(perp, np.nan))
    D["e_old"] = np.log(D.old_px / D.run_px); D["e_new"] = np.log(D.new_px / D.run_px)
    D["f_mode"] = D["mode"].isin(MODES)
    D["f_edge"] = D.f_mode & D.new_px.notna() & (D.new_px > 0) & D.old_px.notna() & (D.old_px > 0)
    gr = (D.g / D.g_raw - 1).abs() < 1e-4
    D["f_g_strict"] = D.f_edge & gr & (D.rep_err.abs() <= 0.01)
    selfok = []
    for i, r in D.iterrows():
        p = parse.get(int(i)); ok = False
        if p and p.get("priors") and not np.isfinite(r.rep_err) and np.isfinite(r.old_px) and r.old_px > 0 and np.isfinite(r.g):
            pr = p["priors"][0]
            if pr.get("kind") == "size" and pr.get("value_si") and str((p.get("prior_obj") or {}).get("phrase")) == str((p.get("target") or {}).get("phrase")):
                u = str(p["q"].get("unit", "")).split("/")[0].strip()
                f = {"m": 1, "cm": 100, "mm": 1000, "km": 1e-3}.get(u)
                if f:
                    k = r.g / r.old_px
                    for dim in (r.hfull, r.wfull):
                        if np.isfinite(dim) and dim > 0 and abs(k * dim / (pr["value_si"] * f) - 1) <= 0.01:
                            ok = True
        selfok.append(ok)
    D["f_selfscale"] = selfok
    D["f_g"] = D.f_g_strict | (D.f_edge & gr & D.f_selfscale)
    D["f_run"] = D.f_g & D.run_px.notna() & (D.run_gap <= RUN_GAP + 1e-9)
    D["f_match"] = D.f_run & (D.e_new.abs() <= math.log(MATCH)) & (D.e_new.abs() < D.e_old.abs())
    D["f_patch"] = D.answer_source.astype(str).str.startswith("patch")
    for c_ in ("p_kind", "p_t", "p_t1", "p_ncut", "p_nw", "p_same"):
        if c_ not in D:
            D[c_] = np.nan
    timed = D.p_t.notna() | D.p_t1.notna()
    pfrac = D.p_ncut / D.p_nw.where(D.p_nw > 0)
    D["f_pcut"] = D.p_kind.notna() & (D.p_ncut >= 0) & ((timed & (D.p_ncut > 0)) | (~timed & (pfrac >= 0.1)))
    return D


def sides(tr, W, H):
    out = []
    for j in tr.okc:
        b = tr.st[j]["box"]
        out.append((b[0] <= TOL, b[2] >= W - 1 - TOL, b[1] <= TOL, b[3] >= H - 1 - TOL))
    return np.array(out, bool).reshape(-1, 4)


def acc_fit(tr, W, H, r):
    """quadratic fit of the centroid on the uncut run; returns the gate quantities"""
    t, P = np.asarray(tr.t, float), np.asarray(tr.uv, float)
    S = sides(tr, W, H); cut = S.any(1)
    m = (t >= r.run_t0 - 1e-6) & (t <= r.run_t1 + 1e-6) & ~cut
    out = dict(nrun=int(m.sum()))
    if m.sum() < 5:
        out["fail"] = "run<5"; return out
    wcut = (t >= r.w_t0 - 1e-6) & (t <= r.w_t1 + 1e-6) & cut
    out["cut_t0"] = float(t[wcut].min()) if wcut.any() else np.nan
    if not wcut.any() or t[wcut].min() <= r.run_t1 + 1e-9:
        out["fail"] = "cut frames not after run"; return out
    tt = t[m] - r.run_t0
    co = [np.polyfit(tt, P[m, k], 2) for k in range(2)]
    res = sum(((np.polyval(co[k], tt) - P[m, k]) ** 2).sum() for k in range(2))
    tot = sum(((P[m, k] - P[m, k].mean()) ** 2).sum() for k in range(2))
    r2 = 1 - res / tot if tot > 0 else 0.0
    sp = lambda x: float(np.hypot(np.polyval(np.polyder(co[0]), x), np.polyval(np.polyder(co[1]), x)))
    v0, v1 = sp(tt[0]), sp(tt[-1])
    g = np.linspace(r.w_t0, r.w_t1, 201) - r.run_t0
    vext = float(np.mean([sp(x) for x in g]))
    out.update(r2=r2, v0=v0, v1=v1, accr=v1 / v0 if v0 > 0 else np.inf, v_ext=vext)
    return out


L = labelled(SUB)
out = {}
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_tedge_diag_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    parse = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        d = json.loads(l); parse[int(d["id"])] = d
    D = tedge_flags(D, det, raw, parse)
    print(f"[{split}] cut rows {len(D)}; f_g strict {int(D.f_g_strict.sum())}; +selfscale {int(D.f_g.sum())}; f_run {int(D.f_run.sum())}; f_match {int(D.f_match.sum())}")
    C = D[D.f_run & ~D.f_match & D.qk.isin(["avgspeed", "speed"]) & ~D.f_pcut & ~D.f_patch
          & D.run_side.isin(["inside", "before"]) & (D.new_px > D.old_px)].copy()
    print(f"[{split}] candidates (rejected only at f_match, gates ok): {len(C)}; of them strict f_g: {int(C.f_g_strict.sum())}")
    P = {}
    if len(C):
        MV.TRACK_VER = "v2sc"; byv, vjs = MV.load_split_tracks(split, "e")
        MV.TRACK_VER = "v3ntmv"; _, vjs3 = MV.load_split_tracks(split, "e")
        recs = {r_["id"]: r_ for v_ in byv.values() for r_ in v_}
        rows = []
        for i, r in C.iterrows():
            rec = recs[int(i)]
            MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(rec.get("video_source", "")), is2d=str(rec["video_type"])[1] == "2")
            src_ = vjs3 if (r.mtr == "v1" and r.vid in vjs3) else vjs
            vj = src_[r.vid]; V = MV.Video(vj, "e")
            tr = V.get(rec["target"])
            o = acc_fit(tr, float(vj["W"]), float(vj["H"]), r)
            ok = "fail" not in o and o["accr"] >= ACC and o["r2"] >= R2MIN
            if ok:
                en, eo = math.log(r.new_px / o["v_ext"]), math.log(r.old_px / o["v_ext"])
                ok = abs(en) <= math.log(MATCH) and abs(en) < abs(eo)
            if ok:
                gn = float(r.g * r.r)
                v, how = fuse(r, gn, r.qk, r["mode"] in STRONG)
                if v is not None and abs(v / r.answer - 1) > 1e-6:
                    P[str(int(i))] = dict(value=float(v), why=f"{TAG}: target border-cut ({r.sidesw}) in {int(r.ncut)}/{int(r.nw)} fit frames; free-edge px {r.old_px:.1f}->{r.new_px:.1f}; "
                                                               f"uncut run {r.run_t0:.2f}-{r.run_t1:.2f}s accelerates (|v| {o['v0']:.0f}->{o['v1']:.0f}, quad R2 {o['r2']:.3f}), "
                                                               f"extrapolated mean over {r.w_t0:.2f}-{r.w_t1:.2f}s {o['v_ext']:.1f}; g {r.g:.4g}->{gn:.4g}; {how}")
            rows.append(dict(id=i, vid=r.vid, cat=r["cat"], qk=r.qk, mode=r["mode"], strict=r.f_g_strict, old=r.old_px, new=r.new_px, run=r.run_px,
                             side=r.run_side, rt=f"{r.run_t0}-{r.run_t1}", wt=f"{r.w_t0}-{r.w_t1}", **{k: o.get(k) for k in ("nrun", "cut_t0", "r2", "v0", "v1", "accr", "v_ext", "fail")},
                             ok=ok, src=r.answer_source, ans=r.answer, newans=P.get(str(int(i)), {}).get("value")))
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] patched {len(P)}")
    for k, v in P.items():
        print(" ", k, det.loc[int(k), "video_id"], "|", det.loc[int(k), "question"], "|", round(det.loc[int(k), "answer"], 4), "->", round(v["value"], 4), "|", v["why"])
    out[split] = P

print("\n=== labelled ===")
for split in ("val", "test"):
    Ls = L[L.split == split]
    new = Ls.answer.copy()
    for k, v in out[split].items():
        if int(k) in new.index:
            y = float(Ls.loc[int(k), "y"])
            print(f"  labelled {split} {k}: y {y:.4g} old {Ls.loc[int(k), 'answer']:.4g} (m {Ls.loc[int(k), 'm']:.3f}) -> {v['value']:.4g} (m {A.row_mra(np.array([v['value']]), np.array([y]))[0]:.3f})")
            new[int(k)] = v["value"]
    m0 = A.row_mra(Ls.answer.values, Ls.y.values); m1 = A.row_mra(new.values, Ls.y.values)
    print(f"{split}: changed {int((new != Ls.answer).sum())}; macro {A.macro(pd.Series(m0, index=Ls.index), Ls.cat)[0]:.4f} -> {A.macro(pd.Series(m1, index=Ls.index), Ls.cat)[0]:.4f}")
