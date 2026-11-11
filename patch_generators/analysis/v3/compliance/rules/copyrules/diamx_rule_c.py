"""diamx (audit rule B3): diameter extent of disc-like objects.

The geo pipeline reads a diameter / radius as the MINOR side of the mask's min-area rectangle (75th pct over frames).
That is right for balls (labels: every alternative extent loses on the 92 labelled ball-prior rows and the ball targets)
and for tall cylinders (bottle, neck, cup: the minor side IS the diameter). It is wrong for flat discs seen obliquely:
the projection of a disc is an ellipse whose MAJOR axis is the diameter (the minor axis is foreshortened by cos(tilt)).
  - flat / hanging discs (lampshade, plate, clock, cookie, coin, pizza, tape roll, red blood cell, ...): major axis
    (min-area-rect long side; on real videos a moving disc uses the extent across the motion, so blur does not count);
  - upright wheels / tires: min(major axis, box height). A wheel stands vertical, so its vertical extent is the diameter
    whatever the yaw; the box height is immune to the horizontal merge with the bicycle frame / car body that inflates
    the major axis.
Guards: spheres and tall cylinders excluded; the track must not touch the frame border (>30 % of extent frames) or be
flagged clipped; the change factor must be in [1.10, 2.6]: below 1.10 the mask is near-circular
(seen face-on) and major-vs-minor is segmentation noise / blur, where the labelled balls prefer the minor side; above 2.6
the box is merged / wrong. Wheels must be upright (box h >= 0.8 w). >= 3 extent frames.
A target diameter scales geo by new/old; a disc PRIOR diameter scales every geo value of its video by old/new.
The final answer is re-fused like submit_v3 (agree weights inferred from the row, strong/weak disagreement routing,
D3 sizes -> CaW). Rows whose answer does not depend on geo are not emitted.
Inputs: tracks (via scan.py -> cache/tmp/ag_diamx_scan_{split}.csv), question / prior text, geo / CaW values.

copyrules fixed copy (2026-10-17, agent jumpffg) of analysis/v3/ag/diamx/rule.py. Changes:
  1. The pooling block at the end (segmented / recoloured copies of one recording whose base answers are identical all
     take the geometric mean of the copies' new answers) selects and changes a row by looking at other rows (other
     videos' answers). It is off by default; DIAMX_POOL=1 restores it (used only for the control run that reproduces the
     production patch).
  2. The repo root is derived from QP_ROOT or five levels above this file.
  Gates, guards and the re-fusion are unchanged. Run through analysis/v3/compliance/rules/copyrules/diamx2_c.py."""
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nza"
pd.set_option('display.width', 400); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 40)

SPH = r"\b(ball|balls|basketball|football|volleyball|baseball|globe|sphere|orb|marble|bubble|balloon|planet|moon|jupiter|io|europa|ganymede|callisto|orange|apple|egg|bead|pearl)\b"
CYL = r"\b(bottle|neck|tube|cup|mug|glass|jar|can|pipe|pole|trunk|cylinder|handle|vase|pot|bucket|column|candle|log|container|planter|plant|table|rim|stem|leg|post|pillar|tower|chimney|barrel|drum|pocket|pockets|cone)\b"
WHEEL = r"\b(wheel|wheels|tire|tires|tyre|tyres)\b"
DISC = (r"\b(lampshade|lampshades|shade|plate|plates|dish|saucer|coin|coins|disc|discs|disk|disks|cd|frisbee|pizza|pepperoni|pie|"
        r"pancake|cookie|cookies|biscuit|clock|hoop|manhole|lid|tray|carpet|rug|cell|cells|donut|doughnut|bagel|button|medal|"
        r"mirror|target|umbrella|trampoline|tape|cymbal|gear|pulley|fan)\b")
STRONG = ("2d_ratio", "3d_calib")
LG2 = np.log10(2.0)


def dclass(p):
    p = str(p).lower()
    if not p or p == "nan" or re.search(SPH, p) or (re.search(CYL, p) and not re.search(WHEEL, p)):
        return None
    if re.search(WHEEL, p) and "steering" not in p:
        return "wheel"
    if re.search(DISC, p):
        return "disc"
    return None


def new_ext(r, pre, cls, real):
    old, maj, bh = r[pre + "old"], r[pre + "maj"], r[pre + "bh"]
    if cls == "wheel":
        new = np.nanmin([maj, bh])
    else:
        mover = r.get(pre + "travel_sz", 0) >= 1.0
        new = r[pre + "acr"] if (real and mover) else maj
    return old, new


MINF = float(os.environ.get("DIAMX_MINF", "1.10"))


def guards(r, pre, old, new, cls=None):
    if not (np.isfinite(old) and np.isfinite(new) and old > 0):
        return "nan"
    if r.get(pre + "nfr", 0) < 3:
        return "few_frames"
    if bool(r.get(pre + "clip", False)) or r.get(pre + "bord", 0) > 0.3:
        return "border"
    if cls == "wheel" and r[pre + "bh"] < 0.8 * r[pre + "bw"]:
        return "wheel_not_upright"            # an upright wheel is at least as tall as wide; a wide box = wrong / merged object
    f = new / old
    if f < MINF:
        return "small"
    if f > 2.6:
        return "too_big"
    if new < 15:
        return "tiny"                         # < 15 px: mask quantisation dominates (the small-object pixel correction is unsigned)
    return None


def fuse(row, gnew, cfull, sim):
    """final answer for a row whose geo becomes gnew (None = answer does not follow geo / unknown route)."""
    src = str(row.answer_source); cs = row.cs; g = row.g; ans = row.answer; mode = str(row["mode"]); cat = row["cat"]
    strong = mode in STRONG
    size = row["kind"] == "size"
    agree = np.isfinite(cs) and cs > 0 and abs(np.log10(gnew / cs)) <= LG2
    if src.startswith("patch") or src in ("var_ans_plain", "patch_copy", "caw_nogeo", "caw_c1", "phys_bound", "phys_bound_final", "scene_prior", "xrow_prior", "kin"):
        return None, "skip_src"
    if src == "geo_cawinvalid" or src == "geo_only_mode" or src == "dis_motion_geo" or src == "agree_geo_kind":
        return gnew, src
    if cat == "D3" and size and src in ("d3size_caw", "agree_gm", "dis_geo", "agree_wgm"):
        return None, "d3size_caw"                     # D3 sizes: CaW answers whatever geo is (if geo stays in the agree / strong path)
    if src in ("d3size_unstable_gm",):
        if agree:
            return float(np.sqrt(gnew * cs)), "d3size_unstable_gm"
        return None, "d3_unstable_dis"
    if src in ("agree_gm", "agree_wgm", "dis_geo", "dis_size_gm", "dis_caw_routed"):
        if src in ("agree_gm", "agree_wgm") and np.isfinite(g) and abs(np.log10(g / cs)) > 0.01:
            w = float(np.log(ans / cs) / np.log(g / cs))
            if not 0.4 <= w <= 0.8:                     # the answer is not a plain weighted gm (pooled / later rule): default weight
                w = (0.7 if cat == "D2" else 0.65) if (strong and size) else 0.5
        else:
            w = (0.7 if cat == "D2" else 0.65) if (strong and size) else 0.5
        if agree:
            if cat == "D3" and size:                    # now an agreement -> the D3 size route (full-video CaW, sim x1.1)
                if not np.isfinite(cfull) or cfull <= 0:
                    return None, "d3_nofull"
                return float(cfull * (1.1 if sim and "height" not in str(row.question).lower() else 1.0)), "d3size_caw_new"
            if row["kind"] in ("dist2", "avgspeed") and src in ("agree_gm", "agree_wgm") and abs(ans - g) / g < 1e-3:
                return gnew, "agree_geo"
            return float(gnew ** w * cs ** (1 - w)), f"agree_w{w:.2f}"
        if src == "dis_caw_routed":
            return None, "still_dis_caw"
        if strong:
            return gnew, "dis_geo"
        if size:
            return float(np.sqrt(gnew * cs)), "dis_size_gm"   # weak-mode size disagreement (--dis_size_gm all)
        return None, "dis_weak_motion"
    return None, "unhandled:" + src


rows_out = {}
logs = []
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_diamx_scan_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    ftk = "caw9b" if split == "val" else "caw9b_orig"
    full = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ftk}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs()
    # per-video factor from a disc PRIOR diameter (applies to every geo value of the video that uses that prior)
    vf = {}
    for i, r in D[D.pdia].iterrows():
        cls = dclass(r.pob)
        if cls is None:
            continue
        real = not str(r.vid).startswith(("simulation", "segment"))
        old, new = new_ext(r, "p_", cls, real)
        if not str(r["mode"]).startswith("2d"):
            # 3D: the focal is calibrated per frame on the prior's extent x its stated depth (often given at one time only),
            # so a ratio of aggregated extents does not carry over; only 2D ratio scales are rescaled
            logs.append(dict(split=split, id=i, side="prior", vid=r.vid, phrase=r.pob, cls=cls, old=old, new=new, guard="prior_3d")); continue
        gd = guards(r, "p_", old, new, cls)
        logs.append(dict(split=split, id=i, side="prior", vid=r.vid, phrase=r.pob, cls=cls, old=old, new=new, guard=gd, bw=r.p_bw, bh=r.p_bh, maj=r.p_maj))
        if gd is None:
            vf[(r.vid, str(r.praw))] = (old / new, f"prior {r.pob} diameter {old:.0f}->{new:.0f}px ({cls})")
    out = {}
    for i, r in D.iterrows():
        fac, why = 1.0, []
        if r.tdia:
            cls = dclass(r.tgt)
            if cls is not None:
                real = not str(r.vid).startswith(("simulation", "segment"))
                old, new = new_ext(r, "t_", cls, real)
                gd = guards(r, "t_", old, new, cls)
                same = r.pdia and str(r.get("p_key")) == str(r.get("t_key"))
                logs.append(dict(split=split, id=i, side="target", vid=r.vid, phrase=r.tgt, cls=cls, old=old, new=new, guard=gd if not same else "same_as_prior", bw=r.t_bw, bh=r.t_bh, maj=r.t_maj))
                if gd is None and not same:
                    fac *= new / old; why.append(f"target {r.tgt} {r.qdim} {old:.0f}->{new:.0f}px ({cls})")
        if r.pdia and (r.vid, str(r.praw)) in vf and not (r.tdia and str(r.get("p_key")) == str(r.get("t_key"))):
            f_, w_ = vf[(r.vid, str(r.praw))]; fac *= f_; why.append(w_)
        if abs(fac - 1) < 1e-6 or i not in det.index:
            continue
        row = det.loc[i].copy()
        for c_, v_ in (("mode", r["mode"]), ("kind", r.qk), ("cat", r["cat"]), ("question", r.question)):
            if c_ not in row.index or (isinstance(row[c_], float) and not np.isfinite(row[c_])):
                row[c_] = v_
        if not (np.isfinite(row.g) and row.g > 0):
            logs.append(dict(split=split, id=i, side="fuse", vid=r.vid, guard="no_g")); continue
        gnew = float(row.g * fac)
        if np.isfinite(row.cs) and row.cs > 0 and abs(np.log10(row.g / row.cs)) > LG2 and abs(np.log10(gnew / row.cs)) > abs(np.log10(row.g / row.cs)):
            # geo already disagrees with CaW by > 2x and the fix would push it further: geo is dominated by another error
            logs.append(dict(split=split, id=i, side="fuse", vid=r.vid, phrase=r.tgt, src=row.answer_source, g=row.g, gnew=gnew, cs=row.cs,
                             ans=row.answer, guard="pushes_dis", fac=fac)); continue
        val, how = fuse(row, gnew, full.get(i, np.nan), str(r.vid).startswith(("simulation", "segment")))
        logs.append(dict(split=split, id=i, side="fuse", vid=r.vid, phrase=r.tgt, src=row.answer_source, g=row.g, gnew=gnew, cs=row.cs,
                         ans=row.answer, new_ans=val, guard=how, fac=fac))
        if val is None or not np.isfinite(val) or val <= 0 or abs(np.log10(val / row.answer)) < 0.002:
            continue
        out[str(int(i))] = {"value": float(val), "why": f"diamx: {'; '.join(why)}; geo {row.g:.4g}->{gnew:.4g}; {how}"}
    # recoloured / segmented copies of one recording whose answers were pooled (identical answers to the same question) stay
    # pooled: every copy takes the geometric mean of the copies' new answers (unchanged copies keep their old answer in it)
    POOL = os.environ.get("DIAMX_POOL", "0") == "1"      # copyrules: cross-row pooling off by default
    sys.path.insert(0, f"{R}/analysis/v3")
    from variant_keys import split_variant
    qn = det.get("question", pd.Series(D.question.reindex(det.index).values, index=det.index)).astype(str).str.strip().str.lower()
    key = det.video_id.astype(str).map(lambda x: split_variant(x)[0]) + "||" + qn
    for k in (set(key[[int(x) for x in out]]) if POOL else ()):
        grp = key.index[(key == k) & ~det.answer_source.astype(str).str.startswith("patch") & ~det.answer_source.isin(["var_ans_plain", "patch_copy"])]
        if len(grp) < 2 or det.answer[grp].round(6).nunique() != 1:
            continue
        vals = [out[str(int(j))]["value"] if str(int(j)) in out else det.answer[j] for j in grp]
        pooled = float(np.exp(np.mean(np.log(vals))))
        for j in grp:
            if det.answer_source[j].startswith("patch") or det.answer_source[j] in ("var_ans_plain", "patch_copy"):
                continue
            w_ = out.get(str(int(j)), {}).get("why", f"diamx: pooled copy of {k}")
            out[str(int(j))] = {"value": pooled, "why": w_ + f"; pooled over {len(grp)} copies"}
    json.dump(out, open(f"{R}/cache/tmp/ag_diamx_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] patch rows: {len(out)}")
LG = pd.DataFrame(logs)
LG.to_csv(f"{R}/cache/tmp/ag_diamx_log.csv")
print(LG[LG.side != "fuse"].groupby(["split", "side", "cls", "guard"], dropna=False).size().to_string())
F = LG[LG.side == "fuse"]
print(F.groupby(["split", "guard"]).size().to_string())
