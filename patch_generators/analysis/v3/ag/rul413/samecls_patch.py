"""B1 same-class ruler mover (simulations only), audit6 synthesis rule B1.
Trigger (per row, from cache/tmp/ag_samecls_rows_{split}.csv written by scan.py):
  simulation video, size prior (not gravity), target head noun == prior head noun (tc.head_noun / heads_compatible),
  target track (the track source whose px quantity reproduces the raw geo, else v2sc entity) covers >= 60 % of the
  tracked frames, box-area CV < CVMAX, quadratic fit of the centroid path R^2 > 0.98, and the asked time t (if any)
  inside the target track's observed time span (+-0.05 s).
Variant (SC_VARIANT):
  spec : any class (the rule as written in the synthesis)
  ball : the ruler is a round rigid object - prior dim diameter/radius and the head noun a ball-like word
         (a sphere's projected diameter does not depend on its orientation or pose; a person's height, a car's length
         or a raccoon's length do)
Action on the FINAL answer (rows whose answer_source starts with 'patch' are skipped):
  speed / accel rows whose source is agree_* (geo within 2x of CaW)        -> geo alone
  rows answered by CaW (caw_*, dis_caw_*, dis_size_gm) with geo outside 2x  -> geo
  var_ans_plain rows (a degraded variant copying its plain video's answer) -> the plain row's new answer
Run scan.py first. Writes cache/tmp/ag_samecls_patch_{test,val}.json (variant ball, border-free area CV < 0.10 by default;
SC_OUT renames) and prints the label evaluation. Label result (sub_xt_Nza):
  spec (any class, border-free CV<0.10): 90 test rows; labels val 8 rows -0.40, test copies 14 rows -1.00 (sim_0171
        astronaut: a deforming body's box is not a ruler) -> rejected
  ball (default): 19 test rows (all S2); labels sim_0163 val 2 rows +0.40, test copies 2 rows +0.40, no labelled loss
  (the all-frame CV < 0.25 variant changes the same 19 rows; all-frame CV < 0.10 drops sim_0162/0163/0256 -> 13 rows)
usage: python analysis/v3/ag/rul413/scan.py; SC_VARIANT=ball|spec SC_CVMAX=0.25 python analysis/v3/ag/rul413/samecls_patch.py [sub_name]"""
import os, sys, json, re, math
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
import analyze_caw as A
from variant_keys import split_variant
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nza"
VAR = os.environ.get("SC_VARIANT", "ball")
CVMAX = float(os.environ.get("SC_CVMAX", "0.10"))     # the synthesis' 0.10, applied to the border-free frames:
CVCOL = os.environ.get("SC_CVCOL", "area_cv_in")      # a ball entering / leaving the frame is cut by the border, which alone
# inflates the all-frame CV to 0.16-0.23 on otherwise constant-size tracks (sim_0162 / 0163 / 0256: interior sqrt-area 32-33.5 px)
COVMIN, R2MIN = 0.6, 0.98
TMARGIN = float(os.environ.get("SC_TMARGIN", "0.05"))
OUTTAG = os.environ.get("SC_OUT", "ag_samecls")
BALL = re.compile(r"(?:ball|sphere|puck|marble|orb|globe|bead|pellet)$")
CAW_SRC = re.compile(r"^(?:caw_(?!nogeo)|dis_caw|dis_size_gm)")


def trigger(d):
    ok = (d.status == "ok") & d.video_id.str.startswith("simulation") & (d["cov"] >= COVMIN) & (d[CVCOL] < CVMAX) & (d.r2 > R2MIN)
    # the asked time must lie inside the target's observed span (a ball entering the frame after t has no measurement at t)
    tin = d.qt.isna() | ((d.qt >= d.t_lo - TMARGIN) & (d.qt <= d.t_hi + TMARGIN))
    ok &= tin
    if VAR == "ball":
        ok &= d.dim_prior.isin(["diameter", "radius"]) & d.ht.astype(str).str.contains(BALL) & d.hp.astype(str).str.contains(BALL)
    return ok


def decide(r):
    src, g, cs, k = str(r.src), r.g, r.cs, r.qkind
    if src.startswith("patch") or not (pd.notna(g) and g > 0):
        return None, None
    have_c = pd.notna(cs) and cs > 0
    inside = have_c and 0.5 <= g / cs <= 2.0
    if src.startswith("agree") and k in ("speed", "accel") and abs(math.log(r.answer / g)) > 1e-6:
        return float(g), f"same-class ruler, clean track ({r.tsrc}: cov {r['cov']:.2f}, areaCV {r[CVCOL]:.3f}, R2 {r.r2:.3f}); {src} -> geo alone"
    if CAW_SRC.match(src) and have_c and not inside:
        return float(g), f"same-class ruler, clean track ({r.tsrc}: cov {r['cov']:.2f}, areaCV {r[CVCOL]:.3f}, R2 {r.r2:.3f}); {src} geo/CaW {g / cs:.2f} -> geo"
    return None, None


def run(split):
    d = pd.read_csv(f"{R}/cache/tmp/ag_samecls_rows_{split}.csv")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    d["fire"] = trigger(d)
    patch = {}
    for _, r in d[d.fire].iterrows():
        v, why = decide(r)
        if v is not None:
            patch[int(r.id)] = dict(value=v, why=why)
    # variant copies follow their plain row
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    q = {int(i): P[int(i)]["question"].strip().lower() for i in det.index if int(i) in P}
    base = {int(i): split_variant(str(P[int(i)]["video_id"]))[0] for i in det.index if int(i) in P}
    plain = {(base[i], q[i]): i for i in q if str(P[i]["video_id"]) == base[i]}
    for i in det.index[det.answer_source == "var_ans_plain"]:
        i = int(i); j = plain.get((base.get(i), q.get(i)))
        if j is not None and j in patch and i not in patch:
            patch[i] = dict(value=patch[j]["value"], why=f"variant copy of plain row {j}: " + patch[j]["why"])
    for i in list(patch):
        if str(det.answer_source.get(i, "")).startswith("patch"):
            del patch[i]
    return d, det, patch


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 48)
    from lab_rows import labelled
    allp = {}
    for split in ("val", "test"):
        d, det, patch = run(split)
        allp[split] = patch
        json.dump({str(k): v for k, v in sorted(patch.items())}, open(f"{R}/cache/tmp/{OUTTAG}_patch_{split}.json", "w"), indent=1)
        print(f"== {split} [{VAR}, {CVCOL}<{CVMAX}]: candidates {len(d)}, fire {int(d.fire.sum())}, rows changed {len(patch)}")
        if patch:
            o = det.loc[[int(k) for k in patch]].copy()
            o["new"] = [patch[int(k)]["value"] for k in o.index]
            print("   by cat:", o["cat"].value_counts().to_dict())
            if split == "test":
                o["q"] = o.question.str[:58]
                print(o[["video_id", "cat", "q", "answer_source", "g", "cs", "answer", "new"]].round(3).to_string())
    L = labelled(SUB).reset_index().rename(columns={"index": "id", "Unnamed: 0": "id"})
    L["new"] = [allp[s].get(int(i), {}).get("value", np.nan) for s, i in zip(L.split, L.id)]
    hit = L[L.new.notna()].copy()
    print(f"== labelled rows (val + all-237 test copies) {len(L)}, changed {len(hit)}")
    if len(hit):
        hit["m_new"] = A.row_mra(hit.new.values, hit.y.values)
        print(hit[["split", "id", "video_id", "cat", "answer_source", "answer", "new", "y", "m", "m_new"]].round(3).to_string())
        for s in ("val", "test"):
            h = hit[hit.split == s]
            if len(h):
                print(f"   {s}: n={len(h)} MRA sum before {h.m.sum():.2f} after {h.m_new.sum():.2f} (delta {h.m_new.sum() - h.m.sum():+.2f})")
        for s in ("val", "test"):
            Ls = L[L.split == s].copy()
            mn = Ls.m.copy(); sel = Ls.new.notna(); mn[sel] = A.row_mra(Ls.new[sel].values, Ls.y[sel].values)
            print(f"   {s} macro: before {A.macro(Ls.m, Ls['cat'])[0]:.4f} after {A.macro(mn, Ls['cat'])[0]:.4f}")
