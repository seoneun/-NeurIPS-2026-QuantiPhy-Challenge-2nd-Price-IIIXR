"""rul329: run the pipeline's fusion (analysis/v3/submit_v3.py, unmodified) exactly as round 66 builds xt_Nzg
(cache/tmp/sr66.sh), but with the geo raw rows the rul329 rule re-measured (cache/tmp/ag_partword_raw_{split}.csv);
every file the submit would write under preds/ or submissions/ is redirected into analysis/v3/ag/rul329/out/.
usage: python run_fusion.py raw   -> write the modified raw csv(s)
       python run_fusion.py fuse  -> run the fusion (tag ag_partword_pw)"""
import os, sys, shlex, runpy, json
import numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
OD = f"{R}/analysis/v3/ag/rul329/out"
os.makedirs(OD, exist_ok=True)
mode = sys.argv[1]
import pandas as pd

if mode == "raw":
    for sp in ("test", "val"):
        mp = f"{R}/cache/tmp/ag_partword_meas_{sp}.json"
        out = f"{R}/cache/tmp/ag_partword_raw_{sp}.csv"
        if os.path.exists(out):
            os.remove(out)
        if not os.path.exists(mp):
            print(sp, "no re-measured rows"); continue
        raw = pd.read_csv(f"{R}/preds/{sp}_{GEO}_raw.csv").set_index("Unnamed: 0")
        M = json.load(open(mp))["res"]; n = 0
        for k, x in M.items():
            i = int(k)
            g0 = raw.geo_value.get(i)
            new = x["new"]
            same = (pd.isna(g0) and not np.isfinite(new)) or (np.isfinite(new) and pd.notna(g0) and abs(new / g0 - 1) < 1e-6)
            if same:
                continue
            raw.loc[i, "geo_value"] = new if np.isfinite(new) else np.nan
            raw.loc[i, "meas_mode"] = x["mode"] if np.isfinite(new) else np.nan
            raw.loc[i, "meas_f"] = x["f"] if x["f"] is not None else np.nan
            raw.loc[i, "meas_why"] = x["why"] if x["why"] else np.nan
            # geo/arbitrate_v2.py: a valid measurement within 2 decades of the prior copy (h1) is the geo answer ("meas"),
            # and the fusion (caw/analyze_v2caw.methods2) reads geo only on rows with answer_source == "meas"
            h1 = abs(float(raw.h1_value.get(i)))
            ok = bool(np.isfinite(new) and new > 0 and abs(np.log10(new / h1)) <= 2)
            raw.loc[i, "meas_valid"] = ok
            raw.loc[i, "meas_value"] = new
            if ok:
                raw.loc[i, "answer_source"] = "meas"; raw.loc[i, "parsed_value"] = new
            else:
                raw.loc[i, "geo_value"] = np.nan
            n += 1
            print(sp, i, "geo", g0, "->", new, x["mode"])
        if n:
            raw.to_csv(out)
        print(sp, "raw rows replaced", n)
    sys.exit(0)

OUT = "ag_partword_pw"
PATCHES = ("dim,lane,kin,hgt,rul264,jump,second,rul001,rul439,wing,twin,roll,rul280,rul265,rul285,rul298,rul413,rul006,wlen,"
           "rul267,rul412,rul303,rul005,rul428,rul424,orbit,rul305,rul282,disc,rul269,rul323,rul432,rul324,p1pf,rul436,"
           "rul438,rul301,rul297")
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
        alt = f"{R}/cache/tmp/ag_partword_raw_{sp}.csv"
        if ps == f"{R}/preds/{sp}_{GEO}_raw.csv" and os.path.exists(alt):
            ps = alt
    if ps.startswith(f"{R}/submissions/{OUT}"):
        ps = _redir(ps)
    return _read(ps, *a, **k)


def to_csv(self, p=None, *a, **k):
    if isinstance(p, str):
        q = _redir(p)
        if q != p:
            print(f"[rul329] write redirected: {p} -> {q}", file=sys.stderr)
        p = q
    return _to(self, p, *a, **k)


pd.read_csv = read_csv
pd.DataFrame.to_csv = to_csv
os.chdir(R)
sys.path.insert(0, os.path.dirname(f"{R}/{script}"))
runpy.run_path(f"{R}/{script}", run_name="__main__")
