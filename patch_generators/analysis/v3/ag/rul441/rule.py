"""rul441 = analysis/v3/ag/rul439/rule.py (copied, TAG rul441) + round-100 filters (label-free; parse + base detail):
  (A) self-scaled size rows (q.kind size, parse target phrase == prior_obj phrase) whose base answer is an agree_gm /
      agree_wgm fusion already within 5% of g_new: re-fusing with CaW only pulls it away from the corrected geo -> keep base.
  (B) speed rows whose re-fusion is agree_gm (a CaW blend) -> keep the base answer.
Same inputs as rul439 (its diag csv, raw geo, base sub_xt_Nz). writes cache/tmp/ag_uncut2r100_patch_{test,val}.json
(OUTSUFFIX env var for a staging copy).

rul439 = analysis/v3/ag/uncut/rule.py + three fixes from the adversarial review.

uncut: a 2D scale from a time-less SPEED prior ('speed of X = v') whose prior track is cut by the image border.
measure_v3.prior_kin reads such a prior as the mean |v| of local fits over the WHOLE track. While the prior object enters /
leaves the frame (box within 2 px of the border), the mask centroid follows only the visible part and moves at about half
the object's speed, so the px speed is too low, metres-per-pixel too high, and every 2d_ratio row of the video inherits the
bias. Gate: >= 20% of the prior track's boxes cut. Then the px speed is re-measured on the longest uncut run (>= 6 frames,
>= 0.25 s, >= 0.3 object sizes of travel; median chord speed over 0.25 s) and every 2d_ratio row of the video whose meas_mpp
reproduces value_si / prior_kin within 1% is rescaled: g_new = g x old_px / new_px (accepted only when the uncut run is at
least as clean as the whole track).

Fixes (rul439):
 1. fusion of geo kinds: the base build runs --geo_kinds dist2,avgspeed (an in-band agreement of those kinds takes geo
    alone -> agree_geo_kind), and 2d_ratio is a strong mode (disagreement -> geo). A rescaled dist2 / avgspeed row therefore
    takes g_new whether it stays in, enters or leaves the 2x band (CaW-routed rows only when g_new enters the band).
 2. shared prior track: a video group is skipped when the prior's SAM2 object is also the object of a differently named
    query (phrase words not subset-related, or both carry a qualifier and they differ: 'virus|right' vs 'virus|left' /
    'viruses|#second') in the measuring track file's 'queries' (the cfg 'e' list measure_v3 uses) or in its 'queries_d'
    (same obj id). One of the two groundings is wrong, so the prior px speed may be measured on the wrong object.
 3. dist2 rows whose target or target2 box (the tracks measure_v3 uses, incl. the second_fix instance) is cut by the image
    border (within 2 px) or missing at the query time are skipped: the centroid distance there is itself biased.

inputs: cache/tmp/ag_uncut2_diag_{split}.csv (analysis/v3/ag/rul439/diag.py), preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv,
submissions/{SUB}_detail / _valdetail. writes cache/tmp/ag_uncut2_patch_{test,val}.json"""
import sys, json, math, os, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 62)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nz"
TAG = "rul441"
OUTSUFFIX = os.environ.get("OUTSUFFIX", "")


def load_parse(split):
    Pz = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); Pz[j["id"]] = j
    return Pz


def ph(e):
    return (e or {}).get("phrase") if isinstance(e, dict) else e


def r100_drop(p, kind, src, old, gn, how):
    """round-100 filter: reason string when the rescaled row should keep its base answer, else None"""
    q = p["q"]
    if q.get("kind") == "size" and ph(p.get("target")) == ph(p.get("prior_obj")) and src in AGREE             and gn and abs(math.log(gn / old)) < math.log(1.05):
        return f"A self-size old {old:.4g} ~ g_new {gn:.4g}"
    if q.get("kind") == "speed" and re.search(r"agree_gm\b", str(how)):
        return f"B speed agree_gm refusion (old {old:.4g}, g_new {gn:.4g})"
    return None

CLIP, NMIN, DMIN, MPPTOL, DISPF = 0.2, 6, 0.25, 0.01, 0.3
BAND = math.log(2.0)
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid"}
GM = {"dis_size_gm", "dis_size_low_gm"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}                       # base: --geo_kinds dist2,avgspeed


def agree_w(kind, cat):
    if kind == "size":
        return 0.7 if cat == "D2" else 0.65        # --agree_w size:0.65:strong, probe D2:agree_wgm:size:w0.7
    return 0.5


def fuse(r, gn, kind):
    """new answer for a rescaled row, or None to leave it"""
    s, a, g = str(r.answer_source), float(r.answer), float(r.g)
    if s.startswith("mo_guard:"):          # guarded row: follows only when its answer is still the plain geo
        if s[9:] in GEO and abs(a / g - 1) < 1e-6:
            return gn, s
        return None, s
    c = float(r.cs) if np.isfinite(r.cs) else np.nan
    in_old = np.isfinite(c) and c > 0 and abs(math.log(g / c)) <= BAND
    in_new = np.isfinite(c) and c > 0 and abs(math.log(gn / c)) <= BAND
    if kind in GEOK:                       # fix 1: geo kinds take geo alone inside the band, and geo outside it (strong mode)
        if s in AGREE or s in GEO or s in GM:
            return gn, (f"{s} -> in 2x band, geo kind -> geo (agree_geo_kind)" if in_new else f"{s} -> outside 2x band -> geo")
        if s in CAWD and in_new and not in_old:
            return gn, f"{s} entered 2x band, geo kind -> geo (agree_geo_kind)"
        return None, s
    if s in AGREE:
        if not in_new:
            return gn, "left 2x band -> geo"
        lg = math.log(g / c)
        w = math.log(a / c) / lg if abs(lg) > 1e-6 else agree_w(kind, r["cat"])
        if not (0.3 <= w <= 0.9):
            w = agree_w(kind, r["cat"])
        return a * (gn / g) ** w, f"{s} w{w:.2f}"
    if s in GEO:
        if s == "dis_geo" and in_new:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return gn, s
    if s in GM:
        if in_new:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"entered 2x band -> gm w{w:.2f}"
        return math.sqrt(gn * c), s
    if s in CAWD:
        if in_new and not in_old:
            w = agree_w(kind, r["cat"])
            return gn ** w * c ** (1 - w), f"{s} entered 2x band -> gm w{w:.2f}"
        return None, s
    return None, s


out = {}
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_uncut2_diag_{split}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    D["ok2d"] = D["mode"].astype(str).str.startswith("2d_ratio")
    for c_ in ("run_n", "run_dur", "cv_run", "chord_run", "run_t0", "run_t1", "run_disp", "sh_id", "sh_d", "sh_dup", "tc1", "tc2", "tm1", "tm2", "tk1", "tk2"):
        if c_ not in D:
            D[c_] = np.nan
    D["f_clip"] = D.ok2d & (D["clip"] >= CLIP)
    # the run must show real translation: >= 0.3 object sizes, so a centroid shift from shape change (a swimming whale's tail
    # kick, a few % of the size) cannot dominate the chord speed (simulation_0279: 71 px of travel for a 764 px whale = 0.09)
    D["f_run"] = D.f_clip & (D.run_n >= NMIN) & (D.run_dur >= DMIN - 1e-9) & (D.run_disp >= DISPF * D["size"])
    D["f_mpp"] = D.f_run & (D.mpp_err.abs() <= MPPTOL)
    D["f_clean"] = D.f_mpp & (D.cv_run <= D.cv_full)
    # fix 2: the prior's SAM2 object is shared with a differently named query (queries or queries_d of the measuring file)
    D["f_shared"] = D.sh_id.notna() | D.sh_d.notna()
    # fix 3: a dist2 row whose target / target2 box is border-cut (or missing) at the query time
    D["f_tcut"] = (D.qk == "dist2") & (D.tc1.notna() | D.tc2.notna()) & ((D.tc1.astype(str) != "ok") | (D.tc2.astype(str) != "ok"))
    D["ratio"] = D.old_px / D.chord_run
    D = D.join(det[[c_ for c_ in ["question", "answer", "answer_source", "g", "cs", "cat", "kind"] if c_ in det]].rename(columns={"cat": "cat_d", "kind": "kind_d"}))
    if "question" not in D:
        D["question"] = raw.question.reindex(D.index)
    D["g_raw"] = raw.geo_value.reindex(D.index)
    D["g_same"] = (D.g / D.g_raw - 1).abs() < 1e-4
    P, rows, dry = {}, [], {}
    PZ = load_parse(split); DROP = {}
    for i, r in D[D.f_clean].iterrows():
        kind = r.kind_d if isinstance(r.get("kind_d"), str) else r.qk
        if not r.g_same or not (np.isfinite(r.g) and r.g > 0):
            rows.append((i, r.vid, r.answer_source, "detail g != raw geo, skipped")); continue
        gn = float(r.g * r.ratio)
        r2 = r.copy(); r2["cat"] = r.cat_d
        v, how = fuse(r2, gn, kind)
        dry[i] = (gn, v, how)                       # what the fusion gives before the fix-2 / fix-3 skips
        if r.f_shared:
            rows.append((i, r.vid, r.answer_source, "SKIP shared prior track")); continue
        if r.f_tcut:
            rows.append((i, r.vid, r.answer_source, "SKIP dist2 target cut at query time")); continue
        rows.append((i, r.vid, r.answer_source, how))
        if v is None or abs(v / r.answer - 1) < 1e-6:
            continue
        why_d = r100_drop(PZ[int(i)], kind, str(r.answer_source), float(r.answer), gn, how)
        if why_d:
            DROP[int(i)] = why_d; continue
        P[str(int(i))] = dict(value=float(v), why=f"prior speed px {r.old_px:.1f}->{r.chord_run:.1f} on uncut run {r.run_t0:.2f}-{r.run_t1:.2f}s "
                                                   f"(border-cut {r['clip']:.0%}); g {r.g:.3g}->{gn:.3g}; {how}")
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}{OUTSUFFIX}.json", "w"), indent=1)
    print(f"[{split}] r100 dropped {len(DROP)}:", DROP)
    out[split] = (D, P, det, dry)
    G = D[D.ok2d].drop_duplicates(["vid", "pkey", "praw"])
    print(f"[{split}] speed-prior 2d_ratio rows {int(D.ok2d.sum())} (videos {G.vid.nunique()}); cut>=20% rows {int(D.f_clip.sum())} "
          f"videos {D[D.f_clip].vid.nunique()}; with uncut run {int(D.f_run.sum())}; mpp reproduced {int(D.f_mpp.sum())}; clean {int(D.f_clean.sum())} "
          f"videos {D[D.f_clean].vid.nunique()}; shared-prior skip {int((D.f_clean & D.f_shared).sum())}; dist2 target-cut skip "
          f"{int((D.f_clean & ~D.f_shared & D.f_tcut).sum())}; patched {len(P)}")
    if rows:
        RR = pd.DataFrame(rows, columns=["id", "vid", "src", "how"])
        print(RR.groupby(["src", "how"]).size().to_string())

# ---------------------------------------------------------------- evaluation
L = labelled(SUB)
for split in ("val", "test"):
    D, P, det, dry = out[split]
    Ls = L[(L.split == split) & L.index.isin([int(k) for k in P])]
    print(f"labelled {split} rows changed: {len(Ls)}")
    if len(Ls):
        new = np.array([P[str(i)]["value"] for i in Ls.index])
        print(pd.DataFrame({"vid": Ls.video_id, "y": Ls.y, "old": Ls.answer, "new": new, "m_old": Ls.m, "m_new": A.row_mra(new, Ls.y.values)}).to_string())
    J = D[D.ok2d].join(L[L.split == split][["y", "m", "mg", "mc"]], how="inner")
    print(f"labelled {split} rows with a time-less speed prior in 2d_ratio: {len(J)}; cut>=20% {int(J.f_clip.sum())}; videos {sorted(J.vid.unique())}")

D, P, det, dry = out["test"]
print("\nflagged groups (test):")
print(D[D.f_clip].drop_duplicates(["vid", "pkey", "praw"])[["vid", "praw", "pkey", "mtr", "clip", "run_t0", "run_t1", "old_px", "chord_run", "ratio",
                                                         "f_clean", "sh_id", "sh_d", "sh_dup"]].round(3).to_string())
print("\ndist2 rows of the flagged groups (target boxes at the query time; tm = border margin px):")
print(D[D.f_clip & (D.qk == "dist2")][["vid", "qt", "tk1", "tc1", "tm1", "tk2", "tc2", "tm2", "f_tcut", "f_shared"]].to_string())
print("\nrows 214-219 (test):")
for i in (214, 215, 216, 217, 218, 219):
    if i in D.index:
        r = D.loc[i]
        gn, v, how = dry.get(i, (np.nan, None, "not f_clean"))
        st = "PATCH %.4g" % P[str(i)]["value"] if str(i) in P else ("skip: shared prior" if r.f_clean and r.f_shared else
                                                                     "skip: dist2 target cut" if r.f_clean and r.f_tcut else "unchanged")
        print(f"  {i} {r.vid} {r.qk} t={r.qt} src={r.answer_source} ans={r.answer:.4g} g={r.g:.4g} cs={r.cs:.4g} g_new={gn:.4g} "
              f"fusion-if-kept={v if v is None else round(v, 4)} ({how}) -> {st}")
    else:
        print(f"  {i} not a candidate row (mode / prior)")
# rows changed vs the old uncut patch
try:
    P0 = json.load(open(f"{R}/cache/tmp/ag_uncut_patch_test.json"))
except Exception:
    P0 = {}
k0, k1 = set(P0), set(P)
print(f"\nvs old uncut patch: old {len(k0)} rows, new {len(k1)}; dropped {sorted(k0 - k1, key=int)}; added {sorted(k1 - k0, key=int)}")
chv = [(k, P0[k]["value"], P[k]["value"]) for k in sorted(k0 & k1, key=int) if abs(P[k]["value"] / P0[k]["value"] - 1) > 1e-6]
print("same rows with a different value:", [(k, round(a, 4), round(b, 4)) for k, a, b in chv])
ch = det.loc[[int(k) for k in P]].copy()
ch["new"] = [P[str(i)]["value"] for i in ch.index]; ch["why"] = [P[str(i)]["why"] for i in ch.index]
print("\ntest rows changed by category:", ch.cat.value_counts().to_dict(), " by source:", ch.answer_source.value_counts().to_dict(),
      " by kind:", ch.kind.value_counts().to_dict() if "kind" in ch else None)
print(ch[["video_id", "question", "answer_source", "kind", "answer", "new", "g", "cs"]].round(3).to_string())
