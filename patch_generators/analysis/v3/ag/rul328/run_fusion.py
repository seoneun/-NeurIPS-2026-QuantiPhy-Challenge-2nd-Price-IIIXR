"""rul328: run the pipeline's fusion (analysis/v3/submit_v3.py, unmodified) exactly as round 71 builds xt_Nzl
(cache/tmp/sr71.sh: Nzk + 65 patches), but with the geo raw rows matched by the rul328 rule (analysis/v3/ag/rul328/rule.py)
set to the pipeline's own no-geo state, i.e. what geo/measure_v3.py writes for a part_target row
(meas_why=part_target, geo/meas NaN, meas_valid False, answer_source vlm, parsed_value = vlm_value).
The fusion then routes them exactly as any no-geo row (caw_nogeo + caw_shrink/probe scaling, var_norm, twin pooling).
Every file the submit would write under preds/ or submissions/ is redirected into analysis/v3/ag/rul328/out/.
usage: python run_fusion.py raw   -> write cache/tmp/ag_partattr_raw_{split}.csv
       python run_fusion.py fuse  -> run the fusion (tag ag_partattr_pa)"""
import os, sys, shlex, runpy, json
import numpy as np
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
OD = f"{R}/analysis/v3/ag/rul328/out"
os.makedirs(OD, exist_ok=True)
sys.path.insert(0, f"{R}/analysis/v3/ag/rul328")
import rule
mode = sys.argv[1]

if mode == "raw":
    for sp in ("test", "val"):
        out = f"{R}/cache/tmp/ag_partattr_raw_{sp}.csv"
        if os.path.exists(out):
            os.remove(out)
        raw = pd.read_csv(f"{R}/preds/{sp}_{GEO}_raw.csv").set_index("Unnamed: 0")
        P = {}
        for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
            x = json.loads(l); P[int(x["id"])] = x
        hits = {}
        for i in raw.index:
            m = rule.match(raw.question[i], P.get(int(i)))
            if m:
                hits[int(i)] = m
        n = 0
        for i, m in hits.items():
            g0 = raw.geo_value.get(i)
            print(sp, i, raw.video_id[i], repr(raw.question[i]), m, "geo", g0, "why", raw.meas_why.get(i))
            if pd.isna(g0) and not bool(raw.meas_valid.get(i)):
                continue          # already no-geo (e.g. part_target / scene_target)
            for c in ("geo_value", "meas_value", "meas_mode", "meas_f", "meas_mpp", "meas_grav", "meas_gap",
                      "meas_owlvote", "meas_nseg", "meas_colswap", "merge_src"):
                if c in raw.columns:
                    raw.loc[i, c] = np.nan
            raw.loc[i, "meas_valid"] = False
            raw.loc[i, "meas_why"] = "part_target"
            raw.loc[i, "answer_source"] = "vlm"
            raw.loc[i, "parsed_value"] = raw.vlm_value[i]
            n += 1
        if n:
            raw.to_csv(out)
        json.dump({str(k): v for k, v in hits.items()}, open(f"{R}/cache/tmp/ag_partattr_hits_{sp}.json", "w"))
        print(sp, "rule hits", len(hits), "raw rows invalidated", n)
    sys.exit(0)

OUT = "ag_partattr_pa"
PATCHES = ("dim,lane,kin,hgt,rul264,jump,second,rul001,rul439,wing,twin,roll,rul280,rul265,rul285,rul298,rul413,rul006,wlen,"
           "rul267,rul412,rul303,rul005,rul428,rul424,orbit,rul305,rul282,disc,rul269,rul323,rul432,rul324,p1pf,rul436,"
           "rul438,rul301,rul297,rul284,rul270,rul454,rul423,rul268,rul313,rul314,rul329,rul417,rul300,rul433,"
           "rul327,rul219,gap,rul333,rul406,rul421,rul319,rul321,rul318,rul322,rul320,rul315,rul420,rul291,"
           "rul425,rul435")
FLB = " --scene_prior_ext dimpre --var_norm --disp_bound 4 --std_court --geo_only_modes 2d_gself --depth_bound --patch_json " + PATCHES
line = [l for l in open(f"{R}/submissions/sub_v30_command.txt") if " x30 --caw" in l][0].strip()
line = line.replace(" x30 ", f" {OUT} ").replace("geo_v4ntrj5pkmsrngsr_d_vw", GEO) + FLB
toks = shlex.split(line)
i = toks.index("python")
for e in toks[:i]:
    k, v = e.split("=", 1); os.environ[k] = v
script = toks[i + 1]
sys.argv = [f"{R}/{script}"] + toks[i + 2:]
_read, _to = pd.read_csv, pd.DataFrame.to_csv


def _redir(p):
    ps = str(p)
    if ps.startswith(f"{R}/preds/") or ps.startswith(f"{R}/submissions/{OUT}"):
        return os.path.join(OD, os.path.basename(ps))
    return ps


def read_csv(p, *a, **k):
    ps = str(p)
    for sp in ("test", "val"):
        alt = f"{R}/cache/tmp/ag_partattr_raw_{sp}.csv"
        if ps == f"{R}/preds/{sp}_{GEO}_raw.csv" and os.path.exists(alt):
            ps = alt
    if ps.startswith(f"{R}/submissions/{OUT}"):
        ps = _redir(ps)
    return _read(ps, *a, **k)


def to_csv(self, p=None, *a, **k):
    if isinstance(p, str):
        q = _redir(p)
        if q != p:
            print(f"[rul328] write redirected: {p} -> {q}", file=sys.stderr)
        p = q
    return _to(self, p, *a, **k)


pd.read_csv = read_csv
pd.DataFrame.to_csv = to_csv
os.chdir(R)
sys.path.insert(0, os.path.dirname(f"{R}/{script}"))
runpy.run_path(f"{R}/{script}", run_name="__main__")
