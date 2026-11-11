"""rul062: undo consistency.c1_mask dup_value for a same-class pair that grounds to the moving prior object.

c1_mask drops geo on both rows when two size questions of one video with different target phrases get identical geo
(assumed: both phrases grounded to one object, so neither can be trusted). When that shared track is the MOVING PRIOR
instance itself, the flag is wrong for the row that names that instance: its size and the prior speed come from the same
track, the most self-consistent geo there is (no depth or calibration difference between scale and target).

Gates (code only, all required):
  1. simulation video;  2. two kind=size rows, different target phrases, identical SI geo (the c1 dup_value pair);
  3. equal head nouns;  4. prior kind speed/accel and the prior object's head noun equals that head noun;
  5. answer_source == caw_c1;  6. not in an applied patch (sr84.sh --patch_json).
Tier A: the row whose target phrase carries a motion participle (rolling, moving, ...) = the moving prior instance.
Tier B: the other row (a second instance of the same class, no size qualifier in either phrase): sim templates place
        one asset several times (labelled sim_0193 moons all 0.137), so the moving instance's size is the best estimate.
Value = the build's fusion for an unflagged row (as rul014): CaW within 2x -> g^w c^(1-w) (w 0.7 D2 size, 0.65 else)
in strong modes / gm in weak; outside 2x -> geo in strong modes, no change in weak modes.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul062/rul062.py [build] [tiers: A|AB (default AB)]
writes cache/tmp/ag_c42dupsame_patch_{test,val}.json (tiers as given) and ag_c42dupsameA_patch_{test,val}.json (tier A).
"""
import sys, os, re, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo")
from rules import UNIT_SI
BUILD = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzy"
TIERS = sys.argv[2] if len(sys.argv) > 2 else "AB"
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
STRONG = ("2d_ratio", "3d_calib"); RATIO = 2.0
MOTION = {"rolling", "moving", "sliding", "falling", "bouncing", "running", "walking", "flying"}
SIZEQ = {"big", "bigger", "biggest", "small", "smaller", "smallest", "large", "larger", "largest", "little", "tiny",
         "giant", "huge", "mini", "tall", "taller", "tallest", "short", "shorter", "shortest", "long", "longer", "longest"}


def words(p):
    return re.findall(r"[a-z]+", str(p or "").lower())


def head(p):
    w = words(p); return w[-1] if w else ""


def fuse(g, c, mode, kind, cat):
    """the build's fusion for an unflagged row (recipe d_r2_gm_c1w3 + agree_w size:0.65:strong + D2:agree_wgm:size:w0.7)"""
    if not (np.isfinite(g) and g > 0):
        return np.nan, "nogeo"
    if not (np.isfinite(c) and c > 0):
        return g, "geo_cawinvalid"
    r = max(g, c) / min(g, c)
    strong = mode in STRONG
    if r <= RATIO + 1e-12:
        if kind == "size" and strong:
            w = 0.7 if cat == "D2" else 0.65
            return g ** w * c ** (1 - w), f"agree_wgm{w}"
        return np.sqrt(g * c), "agree_gm"
    return (g, "dis_geo") if strong else (np.nan, "dis_caw_weak")


sh = open(f"{R}/cache/tmp/sr84.sh").read()
tags = re.search(r"--patch_json\s+\"?([^ \"]+)", sh).group(1).split(",")
for split in ["test", "val"]:
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv")
    if "Unnamed: 0" in raw.columns: raw = raw.set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{BUILD}_{'detail' if split == 'test' else 'valdetail'}.csv")
    if "Unnamed: 0" in det.columns: det = det.set_index("Unnamed: 0")
    cawc = pd.to_numeric(det["cs"] if "cs" in det.columns else det["c"], errors="coerce")
    patched = set()
    for t in tags:
        fp = f"{R}/cache/tmp/ag_{t}_patch_{split}.json"
        if os.path.exists(fp): patched |= {int(k) for k in json.load(open(fp))}
    par = {}
    for i, l in enumerate(open(f"{R}/geo/out/parse_{split}_v2.jsonl")):
        j = json.loads(l); par[int(j.get("id", i))] = j
    info = {}
    for i in raw.index:
        p = par.get(int(i), {}); q = p.get("q") or {}; pr = (p.get("priors") or [{}])[0]
        info[i] = dict(kind=q.get("kind"), unit=q.get("unit"), tph=(p.get("target") or {}).get("phrase"), traw=q.get("target"),
                       pkind=pr.get("kind"), pobj=(p.get("prior_obj") or {}).get("phrase") or pr.get("obj"))
    I = pd.DataFrame.from_dict(info, orient="index")
    g = pd.to_numeric(raw.geo_value, errors="coerce")
    gs = g * I.unit.map(lambda u: UNIT_SI.get(u, np.nan))
    out, outA = {}, {}
    for v, idx in raw.groupby("video_id").groups.items():
        if not str(v).startswith("simulation"): continue
        ix = [i for i in idx if np.isfinite(gs[i]) and I.kind[i] == "size"]
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if I.tph[i] == I.tph[j] or not I.tph[i] or not I.tph[j]: continue
                if abs(gs[i] - gs[j]) > 1e-6 * max(abs(gs[i]), 1e-12): continue
                h = head(I.tph[i])
                if not h or h != head(I.tph[j]) or I.pkind[i] not in ("speed", "accel") or head(I.pobj[i]) != h: continue
                mot = {k: bool(MOTION & (set(words(I.tph[k])) | set(words(I.traw[k])))) for k in (i, j)}
                sizeq = any(SIZEQ & (set(words(I.tph[k])) | set(words(I.traw[k]))) for k in (i, j))
                for k in (i, j):
                    if k not in det.index or k in patched or str(det.answer_source[k]) != "caw_c1": continue
                    tier = "A" if mot[k] else "B"
                    if tier == "B" and (sizeq or mot[i] == mot[j]):   # B needs an identified moving partner, no size qualifier
                        continue
                    c = float(cawc[k]); gv = float(g[k])
                    val, how = fuse(gv, c, str(det["mode"][k]), "size", str(det["cat"][k]))
                    if not np.isfinite(val): continue
                    why = (f"rul062{tier}: '{I.tph[k]}' shares the moving prior '{I.pobj[k]}' track with '{I.tph[j if k == i else i]}'; "
                           f"c1 dup flag undone, {how} g={gv:.4g} c={c:.4g}")
                    rec = {"value": float(val), "why": why}
                    if tier in TIERS: out[str(int(k))] = rec
                    if tier == "A": outA[str(int(k))] = rec
                    print(split, k, v, det["cat"][k], tier, repr(I.tph[k]), "| old", round(float(det.answer[k]), 4), "new", round(val, 4),
                          "geo", round(gv, 4), "caw", round(c, 4), how)
    json.dump(out, open(f"{R}/cache/tmp/ag_c42dupsame_patch_{split}.json", "w"), indent=1)
    json.dump(outA, open(f"{R}/cache/tmp/ag_c42dupsameA_patch_{split}.json", "w"), indent=1)
    print(split, "rows", len(out), "(tier A only:", len(outA), ")", "tiers", TIERS)
