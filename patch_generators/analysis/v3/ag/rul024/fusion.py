"""rul024: run the pipeline fusion (analysis/v3/submit_v3.py, UNMODIFIED) exactly as round 74 builds xt_Nzo
(cache/tmp/sr74.sh: sub_v30 x30 line + FLB + the Nzo --patch_json list), with the geo raw replaced by
cache/tmp/ag_c22rngparse_raw_{split}.csv when TAG ends in _fix.  Every file the submit would write under preds/ or
submissions/<TAG> is redirected into analysis/v3/ag/rul024/out/.
usage: python fusion.py ag_c22rngparse_base|ag_c22rngparse_fix"""
import os, sys, shlex, runpy, re
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
OD = f"{R}/analysis/v3/ag/rul024/out"
os.makedirs(OD, exist_ok=True)
OUT = sys.argv[1]
FIX = OUT.endswith("_fix")
sr = open(f"{R}/cache/tmp/sr74.sh").read()
PATCHES = re.search(r'"Nzo\|\|--patch_json ([^"]+)"', sr).group(1)
print(f"[rul024] {len(PATCHES.split(','))} patches", file=sys.stderr)
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
    if ps.startswith(f"{R}/preds/") or ps.startswith(f"{R}/submissions/{OUT}") or ps.startswith("preds/") or ps.startswith(f"submissions/{OUT}"):
        return os.path.join(OD, os.path.basename(ps))
    return ps


def read_csv(p, *a, **k):
    ps = str(p)
    if FIX:
        for sp in ("test", "val"):
            alt = f"{R}/cache/tmp/ag_c22rngparse_raw_{sp}.csv"
            if ps in (f"{R}/preds/{sp}_{GEO}_raw.csv", f"preds/{sp}_{GEO}_raw.csv") and os.path.exists(alt):
                print(f"[rul024] read redirected: {ps} -> {alt}", file=sys.stderr)
                ps = alt
    if f"submissions/{OUT}" in ps:
        ps = _redir(ps)
    return _read(ps, *a, **k)


def to_csv(self, p=None, *a, **k):
    if isinstance(p, str):
        q = _redir(p)
        if q != p:
            print(f"[rul024] write redirected: {p} -> {q}", file=sys.stderr)
        p = q
    return _to(self, p, *a, **k)


pd.read_csv = read_csv
pd.DataFrame.to_csv = to_csv
os.chdir(R)
sys.path.insert(0, os.path.dirname(f"{R}/{script}"))
runpy.run_path(f"{R}/{script}", run_name="__main__")
