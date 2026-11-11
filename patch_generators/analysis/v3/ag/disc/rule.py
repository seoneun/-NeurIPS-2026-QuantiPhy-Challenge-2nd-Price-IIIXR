"""disc (audit7 rule 4): a flat disc's diameter is the LONG axis of its image ellipse.

measure_v3 reads every diameter / radius as the SHORT side of the mask's min-area rectangle (ext_px -> rr[:,1], 75th pct).
That is right for balls and tall cylinders, and wrong for a flat disc seen obliquely: a circle of diameter D projects to an
ellipse whose long axis is D (the in-plane direction parallel to the image plane) and whose short axis is D cos(tilt).
rul267 already fixes this for its own noun list (lampshade, plate, coin, lid, carpet, ...). This rule covers what diamx does
not see:
  (a) PRIOR diameters of flat game pieces / pucks / tokens (xiangqi = "chinese chess", checker, draughts, puck, go stone,
      poker chip, token, coaster), not Western chess (bare "chess"), not cylinders / spheres; nouns diamx classifies
      (dclass != None) are left to rul267;
  (b) a SAME-CLASS prior (measure_v3 QP_SAMECLASS, raw meas_grav 'sameclass_N': the prior's own grounding was replaced by
      the other tracked instances) whose instances have median long/short aspect > 1.6, any noun except cylinders/spheres;
  (c) TARGET diameters of the nouns in (a) (none in test / val today).
The long axis is the min-area-rect long side rr[:,0], median over the slowest third of the extent frames (centroid speed),
so motion blur along the travel does not lengthen it. Guards as diamx: >= 3 extent frames, ruler not border-clipped,
long / old in [1.10, 2.6], long >= 15 px; 2D-ratio modes only (2d_ratio, vlm_2d_ratio; the ratio scales linearly).
Every geo value of the video that uses the prior scales by old / long (target diameters by long / old), provided the row's
own scale reproduces the re-measured ruler (prior value / raw meas_mpp within 2 % of old; this also admits rows whose target
track came from another source, merge_src xg_*, since their scale is still the prior ruler). The FINAL answer is
re-fused like submit_v3 (agree within 2x -> weighted gm as the row had it; strong-mode disagreement -> geo; weak-mode motion
disagreement -> CaW; weak size disagreement -> gm (--dis_size_gm all)); rows whose geo already disagrees with CaW by > 2x
and moves further away are left alone (diamx guard). Rows already patched (answer_source patch*) and rows of the rul424
patch are skipped. var_ans_plain copies follow their plain row.
Inputs: tracks (scan.py -> cache/tmp/ag_disc_scan_{split}.csv), parse / prior text, raw geo, CaW values. No labels."""
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzc"
pd.set_option('display.width', 400); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 60)

# diamx's classes (analysis/v3/ag/diamx/rule.py): nouns it owns are deferred to rul267
SPH = r"\b(ball|balls|basketball|football|volleyball|baseball|globe|sphere|orb|marble|bubble|balloon|planet|moon|jupiter|io|europa|ganymede|callisto|orange|apple|egg|bead|pearl)\b"
CYL = r"\b(bottle|neck|tube|cup|mug|glass|jar|can|pipe|pole|trunk|cylinder|handle|vase|pot|bucket|column|candle|log|container|planter|plant|table|rim|stem|leg|post|pillar|tower|chimney|barrel|drum|pocket|pockets|cone)\b"
WHEEL = r"\b(wheel|wheels|tire|tires|tyre|tyres)\b"
DISC = (r"\b(lampshade|lampshades|shade|plate|plates|dish|saucer|coin|coins|disc|discs|disk|disks|cd|frisbee|pizza|pepperoni|pie|"
        r"pancake|cookie|cookies|biscuit|clock|hoop|manhole|lid|tray|carpet|rug|cell|cells|donut|doughnut|bagel|button|medal|"
        r"mirror|target|umbrella|trampoline|tape|cymbal|gear|pulley|fan)\b")
# flat game pieces / pucks / tokens (not Western chess: a bare "chess" piece is an upright turned figure)
FLAT = (r"\b(?:chinese\s+chess|xiangqi|(?:red|black)\s+(?:general|chariot|horse|elephant|advisor|cannon|soldier)\s+chess|"
        r"checkers?|draughts?|pucks?|go\s+stones?|poker\s+chips?|tokens?|coasters?)\b")
STRONG = ("2d_ratio", "3d_calib")
LG2 = np.log10(2.0)
MINF, MAXF, MINPX, ASP = 1.10, 2.6, 15.0, 1.6


def diamx_class(p):
    p = str(p).lower()
    if not p or p == "nan" or re.search(SPH, p) or (re.search(CYL, p) and not re.search(WHEEL, p)):
        return None
    if re.search(WHEEL, p) and "steering" not in p:
        return "wheel"
    if re.search(DISC, p):
        return "disc"
    return None


def excluded(p):
    p = str(p).lower()
    return bool(re.search(SPH, p) or re.search(CYL, p) or re.search(WHEEL, p))


def ruler(r):
    """(old px, long px, source, guard) for the prior's ruler of scan row r"""
    p = str(r.pob)
    if excluded(p):
        return None, None, None, "cyl_sph"
    sc = "sameclass" in str(r.grav) and np.isfinite(r.get("sc_old", np.nan))
    if sc:
        old, new, asp, clip, nfr, src = r.sc_old, r.sc_long, r.sc_asp, bool(r.sc_clip), r.sc_nfr, "same-class instances"
    else:
        old, new, asp, clip, nfr, src = r.p_old, r.p_long, r.p_asp, bool(r.p_clip), r.p_nfr, "prior track"
    flat = bool(re.search(FLAT, p)) and diamx_class(p) is None
    if not flat and not (sc and np.isfinite(asp) and asp > ASP):
        return old, new, src, ("diamx_noun" if diamx_class(p) else "not_flat")
    if np.isfinite(r.get("ppx_used", np.nan)) and np.isfinite(old) and abs(np.log(r.ppx_used / old)) > 0.02:
        return old, new, src, "ppx_mismatch"          # the scan does not reproduce the extent measure_v3 used
    if not (np.isfinite(old) and np.isfinite(new) and old > 0):
        return old, new, src, "nan"
    if nfr < 3:
        return old, new, src, "few_frames"
    if clip:
        return old, new, src, "border"
    f = new / old
    if f < MINF:
        return old, new, src, "small"
    if f > MAXF:
        return old, new, src, "too_big"
    if new < MINPX:
        return old, new, src, "tiny"
    return old, new, src, None


def fuse(row, gnew):
    """submit_v3 fusion of a row whose geo becomes gnew -> (value or None, route)"""
    src = str(row.answer_source); cs = row.cs; g = row.g; ans = row.answer; mode = str(row["mode"]); cat = row["cat"]
    strong = mode in STRONG
    size = row["kind"] == "size"
    have_c = np.isfinite(cs) and cs > 0
    agree = have_c and abs(np.log10(gnew / cs)) <= LG2
    if src in ("geo_cawinvalid", "geo_only_mode", "dis_motion_geo", "agree_geo_kind"):
        return gnew, src
    if cat == "D3" and size:
        return None, "d3_size"                         # D3 sizes take CaW / other routes: not handled here
    if src in ("agree_gm", "agree_wgm", "dis_geo", "dis_size_gm", "dis_caw_routed"):
        if src in ("agree_gm", "agree_wgm") and np.isfinite(g) and abs(np.log10(g / cs)) > 0.01:
            w = float(np.log(ans / cs) / np.log(g / cs))
            if not 0.4 <= w <= 0.8:
                w = 0.65 if (strong and size) else 0.5
        else:
            w = 0.65 if (strong and size) else 0.5
        if agree:
            if row["kind"] in ("dist2", "avgspeed") and src in ("agree_gm", "agree_wgm") and abs(ans - g) / g < 1e-3:
                return gnew, "agree_geo"
            return float(gnew ** w * cs ** (1 - w)), f"agree_w{w:.2f}"
        if strong:
            return gnew, "dis_geo"
        if size:
            return float(np.sqrt(gnew * cs)), "dis_size_gm"
        return float(cs), "dis_caw_routed"             # weak-mode motion disagreement: CaW (no 3D --dis_motion here)
    return None, "unhandled:" + src


def run(split, logs):
    D = pd.read_csv(f"{R}/cache/tmp/ag_disc_scan_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0, usecols=["Unnamed: 0", "merge_src", "geo_value"])
    try:
        st2 = set(json.load(open(f"{R}/cache/tmp/ag_stale2_patch_{split}.json")))
    except Exception:
        st2 = set()
    # per (video, prior text): ruler factor
    vf = {}
    for i, r in D[D.pdia].drop_duplicates(["vid", "praw"]).iterrows():
        old, new, src, gd = ruler(r)
        logs.append(dict(split=split, side="prior", vid=r.vid, phrase=r.pob, src=src, old=old, new=new, guard=gd, grav=r.grav, mode=r["mode"]))
        if gd is None:
            vf[(r.vid, str(r.praw))] = (old, new, f"prior '{r.pob}' diameter {old:.1f}->{new:.1f}px ({src}, ellipse long axis, aspect {new / old:.2f})")
    out, info = {}, {}
    for i, r in D.iterrows():
        fac, why, tfix = 1.0, [], False
        if r.tdia and re.search(FLAT, str(r.tgt).lower()) and diamx_class(r.tgt) is None and not excluded(r.tgt):
            old, new = r.get("t_old", np.nan), r.get("t_long", np.nan)
            ok = np.isfinite(old) and np.isfinite(new) and old > 0 and r.get("t_nfr", 0) >= 3 and not bool(r.get("t_clip", False)) \
                and MINF <= new / old <= MAXF and new >= MINPX
            logs.append(dict(split=split, side="target", vid=r.vid, phrase=r.tgt, old=old, new=new, guard=None if ok else "guard"))
            if ok:
                fac *= new / old; why.append(f"target '{r.tgt}' diameter {old:.1f}->{new:.1f}px"); tfix = True
        if r.pdia and (r.vid, str(r.praw)) in vf:
            old_, new_, w_ = vf[(r.vid, str(r.praw))]
            # this row's scale must be the ruler re-measured here (value / meas_mpp = old extent): otherwise another
            # scale source (gravity, belt, other track) produced its geo
            ppx_row = r.vsi / r.mpp if (np.isfinite(r.mpp) and r.mpp > 0) else np.nan
            if not (np.isfinite(ppx_row) and abs(np.log(ppx_row / old_)) <= 0.02):
                logs.append(dict(split=split, side="row", id=i, vid=r.vid, guard=f"ppx_mismatch({ppx_row:.3g} vs {old_:.3g})")); continue
            fac *= old_ / new_; why.append(w_)
        if abs(fac - 1) < 1e-6 or i not in det.index:
            continue
        row = det.loc[i]
        src = str(row.answer_source)
        if "2d_ratio" not in str(r["mode"]) or str(r["mode"]).startswith("3d"):
            logs.append(dict(split=split, side="row", id=i, vid=r.vid, guard="not_2d_ratio:" + str(r["mode"]))); continue
        if src.startswith("patch") or str(i) in st2:
            logs.append(dict(split=split, side="row", id=i, vid=r.vid, guard="already_patched:" + (src if src.startswith("patch") else "rul424"))); continue
        if src in ("var_ans_plain", "patch_copy"):
            info[i] = ("copy", why); continue
        if tfix and str(raw.merge_src.get(i, "")).startswith("xg"):
            # (rul267) the target's geo came from another track source than the extent re-measured here
            logs.append(dict(split=split, side="row", id=i, vid=r.vid, guard="xg_geo_source")); continue
        if not (np.isfinite(row.g) and row.g > 0) or abs(np.log(row.g / raw.geo_value.get(i, np.nan))) > 1e-3:
            logs.append(dict(split=split, side="row", id=i, vid=r.vid, guard="g_not_raw_geo")); continue
        gnew = float(row.g * fac)
        if np.isfinite(row.cs) and row.cs > 0 and abs(np.log10(row.g / row.cs)) > LG2 and abs(np.log10(gnew / row.cs)) > abs(np.log10(row.g / row.cs)):
            logs.append(dict(split=split, side="row", id=i, vid=r.vid, src=src, g=row.g, gnew=gnew, cs=row.cs, ans=row.answer, guard="pushes_dis")); continue
        val, how = fuse(row, gnew)
        logs.append(dict(split=split, side="row", id=i, vid=r.vid, src=src, g=row.g, gnew=gnew, cs=row.cs, ans=row.answer, new_ans=val, guard=how, fac=fac))
        if val is None or not np.isfinite(val) or val <= 0 or abs(np.log10(val / row.answer)) < 0.002:
            continue
        out[str(int(i))] = {"value": float(val), "why": f"disc: {'; '.join(why)}; geo {row.g:.4g}->{gnew:.4g}; {src}->{how}"}
    # var_ans_plain copies follow the plain recording's row with the same question
    from variant_keys import split_variant
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        p_ = json.loads(l); P[int(p_["id"])] = p_
    vid_ = pd.Series({i: str(P[int(i)]["video_id"]) for i in det.index if int(i) in P})
    base = vid_.map(lambda x: split_variant(x)[0])
    qn = pd.Series({i: P[int(i)]["question"].strip().lower() for i in vid_.index})
    det = det.loc[vid_.index]
    plain = {(base[j], qn[j]): j for j in vid_.index if vid_[j] == base[j]}
    for i in det.index[det.answer_source.isin(["var_ans_plain"])]:
        j = plain.get((base[i], qn[i]))
        if j is not None and str(int(j)) in out and str(int(i)) not in out and str(i) not in st2:
            out[str(int(i))] = {"value": out[str(int(j))]["value"], "why": f"disc: var_ans_plain copy of row {j}: " + out[str(int(j))]["why"][6:]}
    json.dump(out, open(f"{R}/cache/tmp/ag_disc_patch_{split}.json", "w"), indent=1)
    return det, out


if __name__ == "__main__":
    logs = []
    res = {s: run(s, logs) for s in ("test", "val")}
    LG = pd.DataFrame(logs)
    LG.to_csv(f"{R}/cache/tmp/ag_disc_log.csv")
    print("== prior rulers"); print(LG[LG.side == "prior"].drop(columns=["side"]).round(3).to_string())
    print("== rows"); print(LG[LG.side == "row"].drop(columns=["side"]).round(4).to_string())
    for s, (det, out) in res.items():
        print(f"== {s}: patch rows {len(out)}")
        if out:
            o = det.loc[[int(k) for k in out]].copy(); o["new"] = [out[k]["value"] for k in out]
            print("   by cat:", o["cat"].value_counts().to_dict())
            o["q"] = o.question.str[:62]
            print(o[["video_id", "cat", "q", "answer_source", "g", "cs", "answer", "new"]].round(4).to_string())
    # labels: val + all-237 test copies
    import analyze_caw as A
    from lab_rows import labelled
    L = labelled(SUB).reset_index().rename(columns={"index": "id", "Unnamed: 0": "id"})
    L["new"] = [res[s][1].get(str(int(i)), {}).get("value", np.nan) for s, i in zip(L.split, L.id)]
    hit = L[L.new.notna()]
    print(f"== labelled rows {len(L)} (val {int((L.split == 'val').sum())}, all-237 test copies {int((L.split == 'test').sum())}); changed {len(hit)}")
    if len(hit):
        hit = hit.copy(); hit["m_new"] = A.row_mra(hit.new.values, hit.y.values)
        print(hit[["split", "id", "video_id", "cat", "answer", "new", "y", "m", "m_new"]].round(3).to_string())
