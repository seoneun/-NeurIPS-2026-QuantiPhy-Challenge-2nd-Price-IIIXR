"""A2 roll patch: turn the re-measured geo (analysis/v3/ag/roll/roll_rule.py -> cache/tmp/ag_roll_meas_<split>.csv) into
FINAL answers on top of sub_xt_Nza, recomputing the fusion like submit_v3:
  within 2x of CaW: avgspeed / dist2 -> geo; size in a strong mode (2d_ratio, 3d_calib) -> g^0.65 c^0.35; else gm
  outside 2x: strong mode -> geo; weak-mode size -> gm (--dis_size_gm all); else CaW
  D3 size rows routed to agree / dis_geo -> d3size_caw: full-video CaW x1.05 (sim heights) / x1.1, then the sim width / length
  dimgm hedge (ratio <= 2.6 / 1.6); class physical bounds (kinematic.class_bound, extra3) last.
Rows already changed by an earlier patch, scene/xrow priors and rows whose geo moves < 2% are skipped; var_ans_plain copies
(segmented recordings) take the plain row's new answer.
usage: python build_patch.py <split> [base submission, default sub_xt_Nza]"""
import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/tools")
import kinematic as K_  # noqa: E402
K_.EXTRA_BOUNDS = K_.EXTRA2 = K_.EXTRA3 = True

split = sys.argv[1] if len(sys.argv) > 1 else "test"
base = sys.argv[2] if len(sys.argv) > 2 else "sub_xt_Nza"
D = pd.read_csv(f"{R}/submissions/{base}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
M = pd.read_csv(f"{R}/cache/tmp/ag_roll_meas_{split}.csv", index_col=0)
RAW = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
full = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{'caw9b_orig' if split == 'test' else 'caw9b'}.csv", index_col=0).parsed_value,
                     errors="coerce").abs()
STRONG = ("2d_ratio", "3d_calib")
RXD = {"width": r"\b(width|wide|wingspan)\b", "length": r"\b(length|long)\b", "height": r"\b(height|tall)\b"}


def fuse(i, g, mode):
    r = D.loc[i]; cs = float(r.cs); kind = r.kind; ql = str(r.question).lower()
    strong = mode in STRONG
    if np.isfinite(cs) and cs > 0:
        rt = g / cs
        if 0.5 <= rt <= 2.0:
            if kind in ("avgspeed", "dist2"):
                v, s = g, "agree_geo_kind"
            elif kind == "size" and strong:
                v, s = g ** 0.65 * cs ** 0.35, "agree_wgm"
            else:
                v, s = float(np.sqrt(g * cs)), "agree_gm"
        elif strong:
            v, s = g, "dis_geo"
        elif kind == "size":
            v, s = float(np.sqrt(g * cs)), "dis_size_gm"
        else:
            v, s = cs, "dis_caw_routed"
    else:
        v, s = g, "geo_nocaw"
    if r["cat"] == "D3" and kind == "size" and s in ("agree_gm", "agree_wgm", "dis_geo"):
        fv = full.get(i, np.nan)
        v = float(fv) if (np.isfinite(fv) and fv > 0) else cs; s = "d3size_caw"
        if str(r.video_id).startswith("simulation"):
            v *= 1.05 if re.search(RXD["height"], ql) else 1.1
            for dm_, rmax_ in (("width", 2.6), ("length", 1.6)):
                if re.search(RXD[dm_], ql) and not re.search(RXD["height"], ql) and abs(np.log10(g / v)) <= np.log10(rmax_):
                    v, s = float(np.sqrt(v * g)), "d3size_dimgm"
    b = K_.class_bound(str(r.question), str(r.ground_truth_prior) if "ground_truth_prior" in D.columns else str(RAW.ground_truth_prior.get(i)), kind)
    if b is not None:
        f_ = K_.q_si(str(r.question)); ins = lambda x: np.isfinite(x) and x > 0 and b[0] <= x * f_ <= b[1]
        if not ins(v):
            c_ = [x for x in (g, cs) if ins(x)]
            if len(c_) == 1:
                v, s = c_[0], "phys_bound"
    return float(v), s


patch, rows = {}, []
for i, m in M.iterrows():
    if i not in D.index:
        continue
    d = D.loc[i]; src = str(d.answer_source)
    if src.startswith("patch") or src in ("scene_prior", "xrow_prior", "var_ans_plain"):
        rows.append(dict(id=i, skip=src)); continue
    if not (np.isfinite(m.base) and np.isfinite(m.rule) and m.base > 0 and m.rule > 0 and np.isfinite(m.geo_raw)):
        rows.append(dict(id=i, skip="no_geo")); continue
    k = m.rule / m.base
    if abs(np.log(k)) < np.log(1.02):
        rows.append(dict(id=i, skip=f"geo_moves_{k:.3f}")); continue
    g2 = float(m.geo_raw) * k
    mode2 = ("vlm_" if str(m.mode_raw).startswith("vlm_") else "") + str(m.rule_mode)
    v, s = fuse(i, g2, mode2)
    why = (f"roll: rolled video, heights on image x (f/W {m.base_f:.3f}->{m.rule_f:.3f})" if m.rule_mode != m.base_mode or
           (np.isfinite(m.rule_f) and m.rule_f > 0) else f"roll: lying-body height prior = major axis (mpp x{k:.3f})")
    why += f"; geo {m.geo_raw:.4g}->{g2:.4g} {mode2}; {s}"
    rows.append(dict(id=i, vid=d.video_id, cat=d["cat"], q=str(d.question)[:70], old=d.answer, old_src=src, g_old=m.geo_raw, g_new=g2,
                     cs=d.cs, new=v, new_src=s))
    if abs(np.log(v / float(d.answer))) > 1e-6:
        patch[int(i)] = dict(value=v, why=why)
# var_ans_plain copies: the segmented recording repeats the plain recording's answer
for i in [j for j in D.index if str(D.answer_source[j]) == "var_ans_plain"]:
    pv = re.sub(r"_segmented$", "", str(D.video_id[i]))
    if pv == D.video_id[i]:
        continue
    src_ = D.index[(D.video_id == pv) & (D.question == D.question[i])]
    for j in src_:
        if int(j) in patch:
            patch[int(i)] = dict(value=patch[int(j)]["value"], why=f"copy of {int(j)} (var_ans_plain); " + patch[int(j)]["why"])
            rows.append(dict(id=i, vid=D.video_id[i], cat=D.cat[i], q=str(D.question[i])[:70], old=D.answer[i], old_src="var_ans_plain",
                             new=patch[int(j)]["value"], new_src=f"copy_{int(j)}"))
json.dump(patch, open(f"{R}/cache/tmp/ag_roll_patch_{split}.json", "w"), indent=1)
T = pd.DataFrame(rows)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 72)
print(T.to_string())
print(f"[{split}] patch rows {len(patch)} by cat", D.cat.reindex([int(x) for x in patch]).value_counts().to_dict(),
      "->", f"cache/tmp/ag_roll_patch_{split}.json")
