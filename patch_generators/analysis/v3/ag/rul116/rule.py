"""rul116: (A) rigid-length centre rebuild for dist2 rows whose target box is border-cut at the query time,
(B) the positional-qualifier exception to rul439's fix-2 (shared prior track) skip.

A. Scope: dist2 rows with a query time t in geo mode 2d_ratio (not 'minimum/closest' questions). At t, the target or target2
   box (the tracks measure_v3 uses, incl. the second_fix instance) touches the image border (2 px) on exactly ONE side of that
   track's motion axis (larger centroid travel). The mask centroid / box centre of a cut box follows only the visible part.
   L = median box extent on that axis over the track's fully uncut frames (>= 8 frames, CV <= 3 %: a rigid object). The
   free (uncut) edge is a rigid point of the object, so the centre on the axis = free edge -/+ L/2; perpendicular = box centre.
   px' = distance of the two box centres (rebuilt where cut); g' = px' x mpp x k, where k is the prior-ruler fix of the
   group: rul117's k when it patched the group, else rul439's old_px / new_px when rul439 (or part B) rescaled it.
   Guard: when the group's prior is a time-less speed / accel prior whose own track is border-cut (rul439 diag clip >= 0.2 or
   rul117 diag cut >= 0.3) and NO ruler fix fired, part A is skipped (the px fix alone would multiply a biased ruler).
   Answer = g' (dist2 is a geo kind in the strong 2d_ratio mode: in band -> agree_geo_kind, outside -> geo).
B. rul439 fix 2 skipped a clean group whose prior SAM2 object is shared with a differently named query. Keep it when the prior
   carries a positional qualifier (left/right/top/bottom...) and the prior track lies on that side of EVERY other distinct
   same-noun track of the measuring file (different obj, not an IoU duplicate) in >= 90 % of the common frames with median
   offset > 0.25 x its box extent (the grounding is then self-consistent). Then rul439's rescale and fusion apply to the
   group's 2d_ratio rows (dist2 rows cut at t go through part A with that k; other cut dist2 rows stay skipped).
   The rescale k = old_px / new_px takes new_px from rul117's free-edge line fit when that fit passes rul117's own
   gates (only its shared-prior skip is lifted), else rul439's uncut-run chord speed (C66_BKFE=0: always rul439's).
inputs: cache/tmp/ag_c66d2edge_diagA_{split}.csv (diagA.py), cache/tmp/ag_uncut2_diag_{split}.csv,
cache/tmp/ag_c66freeedge_{diag,patch}_{split}.*, preds raw, submissions/{SUB}_detail.
writes cache/tmp/ag_c66d2edge_patch_{test,val}.json (FINAL answers). Place after rul117, before c63plainkey."""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, R + "/geo")
import analyze_caw as A
from lab_rows import labelled
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
SUB = os.environ.get("C66_SUB", "sub_xt_Nzz15")
TAG = "rul116"
FE = os.environ.get("C66_FE", "rul117")
WRITE = os.environ.get("C66_WRITE", "1") == "1"
NOGUARD = os.environ.get("C66_NOGUARD", "0") == "1"
BKFE = os.environ.get("C66_BKFE", "1") == "1"          # part B px speed from rul117's edge fit when its gates pass
POSFRAC = 0.9
BAND = math.log(2.0)
POS = {"left": (0, -1), "leftmost": (0, -1), "right": (0, 1), "rightmost": (0, 1),
       "top": (1, -1), "upper": (1, -1), "bottom": (1, 1), "lower": (1, 1)}
STOP = {"the", "a", "an", "of", "in", "on"}
AGREE = {"agree_gm", "agree_wgm"}
GEO = {"dis_geo", "dis_motion_geo", "agree_geo_kind", "cqc_geo", "geo_cawinvalid"}
GM = {"dis_size_gm", "dis_size_low_gm"}
CAWD = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOK = {"dist2", "avgspeed"}


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


def sing(w):
    if w.endswith("es") and len(w) > 4 and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def head(p):
    ws = [w for w in re.findall(r"[a-z]+", (p or "").lower()) if w not in STOP]
    return sing(ws[-1]) if ws else ""


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def common(ta, tb):
    n = min(len(ta.st), len(tb.st))
    return [j for j in range(n) if ta.st[j] and tb.st[j] and ta.st[j].get("box") is not None and tb.st[j].get("box") is not None]


def dup(ta, tb):
    com = common(ta, tb)
    if len(com) < 5:
        return False
    io = np.array([iou(ta.st[j]["box"], tb.st[j]["box"]) for j in com])
    return bool(np.median(io) >= 0.8 and (io >= 0.8).mean() >= 0.8)


def pos_check(V, tr):
    """part B: the prior track lies on its qualifier's side of every other distinct same-noun track (>= 90 % of common frames,
    median offset > 0.25 x its box extent)"""
    pw = [w for w in (tr.q.get("where") or []) if w in POS]
    if not pw:
        return False, "no_pos_qualifier"
    ax, sg = POS[pw[0]]
    pk, pobj, h0 = tr.q.get("key"), tr.q.get("obj"), head(tr.q.get("phrase"))
    oth, seen = [], set()
    for k, t2 in V.tr.items():
        o2 = t2.q.get("obj")
        if k == pk or t2 is tr or head(t2.q.get("phrase")) != h0 or (pobj is not None and o2 == pobj) or o2 in seen:
            continue
        if dup(tr, t2):
            continue
        seen.add(o2); oth.append((k, t2))
    if not oth:
        return False, "no_other_distinct_instance"
    ok, msgs = True, []
    for k, t2 in oth:
        com = common(tr, t2)
        if len(com) < 5:
            ok = False; msgs.append(f"{k}:few_common"); continue
        d = np.array([sg * (tr.st[j]["c"][ax] - t2.st[j]["c"][ax]) for j in com], float)
        ext = float(np.median([tr.st[j]["box"][ax + 2] - tr.st[j]["box"][ax] for j in com]))
        o_ = bool(np.median(d) > 0.25 * ext and (d > 0).mean() >= POSFRAC)
        ok &= o_
        msgs.append(f"{k}(obj{t2.q.get('obj')}):{np.median(d):.0f}px/{(d > 0).mean():.2f}")
    return ok, ";".join(msgs)


def load_fe(split):
    try:
        return json.load(open(f"{R}/cache/tmp/ag_{FE}_patch_{split}.json")), True
    except FileNotFoundError:
        return {}, False


out = {}
MV.TRACK_VER = "v2sc"
TRK = {}
for split in ("test", "val"):
    _, vjs = MV.load_split_tracks(split, "e")
    TRK[split] = vjs
for split in ("test", "val"):
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    det["prior"] = raw.ground_truth_prior.reindex(det.index).astype(str)
    det["g_raw"] = raw.geo_value.reindex(det.index)
    det["rmode"] = raw.meas_mode.reindex(det.index).astype(str)
    DA = pd.read_csv(f"{R}/cache/tmp/ag_{TAG}_diagA_{split}.csv", index_col=0)
    DU = pd.read_csv(f"{R}/cache/tmp/ag_uncut2_diag_{split}.csv", index_col=0)
    try:
        DF = pd.read_csv(f"{R}/cache/tmp/ag_{FE}_diag_{split}.csv", index_col=0)
    except FileNotFoundError:
        DF = pd.DataFrame(columns=["vid", "praw", "cut"])
    FP, have_fe = load_fe(split)
    # ---------------- part B: groups rul439 cleaned but skipped as shared, re-checked with the positional rule
    DU["ok2d"] = DU["mode"].astype(str).str.startswith("2d_ratio")
    for c_ in ("run_n", "run_dur", "cv_run", "chord_run", "run_disp", "sh_id", "sh_d", "tc1", "tc2"):
        if c_ not in DU:
            DU[c_] = np.nan
    f_clean = DU.ok2d & (DU["clip"] >= 0.2) & (DU.run_n >= 6) & (DU.run_dur >= 0.25 - 1e-9) & (DU.run_disp >= 0.3 * DU["size"]) & \
        (DU.mpp_err.abs() <= 0.01) & (DU.cv_run <= DU.cv_full)
    f_shared = DU.sh_id.notna() | DU.sh_d.notna()
    DU["f_clean"] = f_clean
    DU["ratio"] = DU.old_px / DU.chord_run
    GB = DU[f_clean & f_shared].drop_duplicates(["vid", "praw"])
    keepB = {}
    for i, r in GB.iterrows():
        vj = TRK[split].get(r.vid)
        if vj is None:
            continue
        V = MV.Video(vj, "e")
        tr = V.tr.get(r.pkey)
        if tr is None:
            continue
        ok, why = pos_check(V, tr)
        # the group's px speed: rul117's free-edge line fit when its own fit gates pass (an unbiased fit of a rigid point;
        # rul439's median chord of the mask centroid is biased up by centroid jitter), else rul439's uncut-run chord speed
        kb, kbsrc = float(r.ratio), "rul439"
        if BKFE:
            ff = DF[(DF.vid == r.vid) & (DF.praw == r.praw) & DF["k"].notna()] if len(DF) and "k" in DF else DF.iloc[0:0]
            if len(ff):
                f0 = ff.iloc[0]
                try:
                    gate = (f0["cut"] >= 0.3 and f0.free_ok >= 0.8 and f0.nfit >= 20 and (f0.edge_rms <= 1.0 or (f0.nfit >= 25 and f0.edge_rms <= 2.0))
                            and abs(f0.mpp_err) <= 0.01 and 0.4 <= f0.k <= 1.0 and str(f0.pkind) == "speed")
                except Exception:
                    gate = False
                if gate:
                    kb, kbsrc = float(f0.k), f"{FE} edge fit (rul439 {r.ratio:.3f})"
        print(f"[{split}] part B group {r.vid} prior {r.praw!r} key {r.pkey} obj {tr.q.get('obj')}: pos {ok} ({why}); k {kb:.3f} [{kbsrc}]")
        if ok:
            keepB[(r.vid, r.praw)] = (kb, why, kbsrc)
    P, rows = {}, []
    # part B rows other than dist2-cut ones: rul439 fusion with g x k
    for i, r in DU[f_clean & f_shared].iterrows():
        key = (r.vid, r.praw)
        if key not in keepB or i not in det.index:
            continue
        d = det.loc[i]
        k = keepB[key][0]
        tcut = r.qk == "dist2" and (pd.notna(r.tc1) or pd.notna(r.tc2)) and ((str(r.tc1) != "ok") or (str(r.tc2) != "ok"))
        if tcut:
            rows.append((i, r.vid, d.answer_source, "B: dist2 target cut at t -> part A")); continue
        if not (np.isfinite(d.g) and d.g > 0 and abs(d.g / d.g_raw - 1) < 1e-4):
            rows.append((i, r.vid, d.answer_source, "B: g != raw geo, skipped")); continue
        gn = float(d.g * k)
        v, how = fuse(str(d.answer_source), float(d.answer), float(d.g), float(d.cs) if np.isfinite(d.cs) else np.nan, gn, d["kind"], d["cat"])
        rows.append((i, r.vid, d.answer_source, "B: " + how))
        if v is None or abs(v / d.answer - 1) < 1e-6:
            continue
        P[str(int(i))] = dict(value=float(v), why=f"{TAG}(B): rul439 group kept (prior qualifier side-consistent: {keepB[key][1]}); "
                                                   f"k {k:.3f} [{keepB[key][2]}]; g {d.g:.4g}->{gn:.4g}; {how}; was {d.answer:.4g} ({d.answer_source})")
    # ---------------- part A
    X = DA[DA["mode"].astype(str).str.startswith("2d_ratio") & DA.fire.fillna(False).astype(bool) & ~DA.minq.fillna(False).astype(bool)]
    for i, r in X.iterrows():
        if i not in det.index:
            continue
        d = det.loc[i]
        praw = d.prior
        k, ksrc = 1.0, "none"
        grp = det[(det.video_id == r.vid) & (det.prior == praw)].index
        fe_k = None
        for j in grp:
            w = FP.get(str(int(j)), {}).get("why", "")
            m = re.search(r"^" + re.escape(FE) + r": .*?; k ([0-9.]+);", w)
            if m:
                fe_k = float(m.group(1)); break
        u2 = [j for j in grp if str(det.answer_source[j]).replace("mo_guard:", "") == "patch:rul439"]
        if fe_k is not None:
            k, ksrc = fe_k, FE
        elif (r.vid, praw) in keepB:
            k, ksrc = keepB[(r.vid, praw)][0], "B: " + keepB[(r.vid, praw)][2]
        elif u2:
            uu = DU[(DU.vid == r.vid) & (DU.praw == praw) & DU.f_clean]
            if len(uu):
                k, ksrc = float(uu.ratio.iloc[0]), "rul439"
        if ksrc == "none" and not NOGUARD:
            cu = DU[(DU.vid == r.vid) & (DU.praw == praw) & DU.ok2d]
            cf = DF[(DF.vid == r.vid) & (DF.praw == praw) & DF["cut"].notna()] if len(DF) and "cut" in DF else DF.iloc[0:0]
            clip_u = float(cu["clip"].iloc[0]) if len(cu) else np.nan
            cut_f = float(cf["cut"].iloc[0]) if len(cf) and pd.notna(cf["cut"].iloc[0]) else np.nan
            if (np.isfinite(clip_u) and clip_u >= 0.2) or (np.isfinite(cut_f) and cut_f >= 0.3):
                rows.append((i, r.vid, d.answer_source, f"A: SKIP prior track border-cut (clip {clip_u}, cut {cut_f}) and no ruler fix")); continue
        mpp = float(r.mpp)
        gn = float(r.px_new) * mpp * k
        v = gn
        which = "+".join(n_ for n_, f_ in (("target", r.fire1), ("target2", r.fire2)) if bool(f_))
        how = f"px {r.px_pipe:.1f}->{r.px_new:.1f} (rebuilt {which}), mpp {mpp:.5f} x k {k:.3f} ({ksrc})"
        rows.append((i, r.vid, d.answer_source, "A: " + how))
        if abs(v / d.answer - 1) < 1e-6:
            continue
        Lr = r.L1 if bool(r.fire1) else r.L2
        P[str(int(i))] = dict(value=float(v), why=f"{TAG}(A): box cut at t={r.qt} -> centre = free edge -/+ rigid L/2 (L {Lr:.0f}px); {how}; "
                                                   f"g' {gn:.4g} (geo kind, strong mode); was {d.answer:.4g} ({d.answer_source})")
    # _segmented twins that copy a changed plain row
    det["ql"] = raw.question.reindex(det.index).astype(str).str.strip().str.lower()
    for kk in list(P):
        i = int(kk); vid = str(det.video_id[i])
        if vid.endswith("_segmented"):
            continue
        tw = det[(det.video_id == vid + "_segmented") & (det.ql == det.ql[i]) & (det.prior == det.prior[i])]
        for j, rj in tw.iterrows():
            if abs(float(rj.answer) / float(det.answer[i]) - 1) < 1e-6 and str(j) not in P:
                P[str(int(j))] = dict(value=P[kk]["value"], why=f"{TAG}: copy of plain row {i} ({rj.answer_source})")
                rows.append((j, vid + "_segmented", rj.answer_source, f"twin copy of {i}"))
    if WRITE:
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print(f"\n[{split}] freeedge patch present: {have_fe}; rows")
    if rows:
        print(pd.DataFrame(rows, columns=["id", "vid", "src", "how"]).to_string())
    out[split] = (P, det)

L = labelled(SUB)
for split in ("val", "test"):
    P, det = out[split]
    Ls = L[(L.split == split) & L.index.isin([int(k) for k in P])]
    print(f"labelled {split} rows changed: {len(Ls)}")
    if len(Ls):
        new = np.array([P[str(i)]["value"] for i in Ls.index])
        print(pd.DataFrame({"vid": Ls.video_id, "y": Ls.y, "old": Ls.answer, "new": new, "m_old": Ls.m, "m_new": A.row_mra(new, Ls.y.values)}).to_string())
P, det = out["test"]
if P:
    ch = det.loc[[int(k) for k in P]].copy()
    ch["new"] = [P[str(i)]["value"] for i in ch.index]
    print("\ntest rows changed by category:", ch["cat"].value_counts().to_dict())
    print(ch[["video_id", "question", "answer_source", "answer", "new", "g", "cs"]].round(4).to_string())
    for k, v in P.items():
        print(k, v["why"])
