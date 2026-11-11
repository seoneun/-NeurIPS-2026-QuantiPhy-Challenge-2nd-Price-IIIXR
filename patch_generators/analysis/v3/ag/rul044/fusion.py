"""rul044 step 4: run the pipeline fusion (analysis/v3/submit_v3.py, UNMODIFIED) exactly as round 77 builds xt_Nzr
(cache/tmp/sr77.sh: sub_v30 x30 line + FLB + the Nzr --patch_json list), with optional changes:
  C30W_RAW=1     geo raw replaced by cache/tmp/ag_c30win_raw_{split}.csv (mkraw.py)
  C30W_DROP=a,b  patch tags left out (rul424: its two rows are the ones rul044 re-answers)
Every file the submit would write under preds/ or submissions/<TAG> is redirected into analysis/v3/ag/rul044/out/.
usage: [env] python fusion.py <TAG>"""
import os, sys, shlex, runpy, re
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
OD = f"{R}/analysis/v3/ag/rul044/out"
os.makedirs(OD, exist_ok=True)
OUT = sys.argv[1]
RAW = os.environ.get("C30W_RAW") == "1"
DROP = set(x for x in os.environ.get("C30W_DROP", "").split(",") if x)
sr = open(f"{R}/cache/tmp/sr77.sh").read()
PATCHES = re.search(r'"Nzr\|\|--patch_json ([^"]+)"', sr).group(1)
PATCHES = ",".join(p for p in PATCHES.split(",") if p not in DROP)
print(f"[rul044] {len(PATCHES.split(','))} patches (dropped {sorted(DROP)}), raw={RAW}", file=sys.stderr)
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
    if RAW:
        for sp in ("test", "val"):
            alt = f"{R}/cache/tmp/ag_c30win_raw_{sp}.csv"
            if ps in (f"{R}/preds/{sp}_{GEO}_raw.csv", f"preds/{sp}_{GEO}_raw.csv") and os.path.exists(alt):
                ps = alt
    if f"submissions/{OUT}" in ps:
        ps = _redir(ps)
    return _read(ps, *a, **k)


def to_csv(self, p=None, *a, **k):
    if isinstance(p, str):
        p = _redir(p)
    return _to(self, p, *a, **k)


pd.read_csv = read_csv
pd.DataFrame.to_csv = to_csv
os.chdir(R)
sys.path.insert(0, os.path.dirname(f"{R}/{script}"))
runpy.run_path(f"{R}/{script}", run_name="__main__")
