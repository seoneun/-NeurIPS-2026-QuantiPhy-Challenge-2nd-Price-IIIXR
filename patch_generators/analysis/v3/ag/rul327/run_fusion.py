"""rul327: run the pipeline fusion (analysis/v3/submit_v3.py, UNMODIFIED) exactly as round 67 builds xt_Nzh (cache/tmp/sr67.sh),
with kinematic.class_bound wrapped by handbound.install (adds the hand/finger close-up speed bound). Every file the
submit would write under preds/ or submissions/ is redirected into analysis/v3/ag/rul327/out/.
usage: python run_fusion.py [tag]   (tag default ag_parse3_hb; tag ending in _base runs WITHOUT the wrap, reproducibility check)"""
import os, sys, shlex, runpy
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
OD = f"{R}/analysis/v3/ag/rul327/out"
os.makedirs(OD, exist_ok=True)
OUT = sys.argv[1] if len(sys.argv) > 1 else "ag_parse3_hb"
PATCHES = ("dim,lane,kin,hgt,rul264,jump,second,rul001,rul439,wing,twin,roll,rul280,rul265,rul285,rul298,rul413,rul006,wlen,"
           "rul267,rul412,rul303,rul005,rul428,rul424,orbit,rul305,rul282,disc,rul269,rul323,rul432,rul324,p1pf,rul436,"
           "rul438,rul301,rul297,rul284,rul270,rul454,rul423,rul268,rul313,rul314,rul329")
FLB = " --scene_prior_ext dimpre --var_norm --disp_bound 4 --std_court --geo_only_modes 2d_gself --depth_bound --patch_json " + PATCHES
line = [l for l in open(f"{R}/submissions/sub_v30_command.txt") if " x30 --caw" in l][0].strip()
line = line.replace(" x30 ", f" {OUT} ").replace("geo_v4ntrj5pkmsrngsr_d_vw", GEO) + FLB
toks = shlex.split(line)
i = toks.index("python")
for e in toks[:i]:
    k, v = e.split("=", 1); os.environ[k] = v
script = toks[i + 1]
sys.argv = [f"{R}/{script}"] + toks[i + 2:]
sys.path.insert(0, os.path.dirname(f"{R}/{script}"))
sys.path.insert(0, f"{R}/analysis/v3/ag/rul327")
if not OUT.endswith("_base"):
    import kinematic, handbound
    handbound.install(kinematic)
    print("[rul327] kinematic.class_bound wrapped with the hand bound", file=sys.stderr)
_read, _to = pd.read_csv, pd.DataFrame.to_csv


def _redir(p):
    ps = str(p)
    if ps.startswith(f"{R}/preds/") or ps.startswith(f"{R}/submissions/{OUT}") or ps.startswith(f"submissions/{OUT}") or ps.startswith("preds/"):
        return os.path.join(OD, os.path.basename(ps))
    return ps


def read_csv(p, *a, **k):
    ps = str(p)
    if f"submissions/{OUT}" in ps:
        ps = _redir(ps)
    return _read(ps, *a, **k)


def to_csv(self, p=None, *a, **k):
    if isinstance(p, str):
        q = _redir(p)
        if q != p:
            print(f"[rul327] write redirected: {p} -> {q}", file=sys.stderr)
        p = q
    return _to(self, p, *a, **k)


pd.read_csv = read_csv
pd.DataFrame.to_csv = to_csv
os.chdir(R)
runpy.run_path(f"{R}/{script}", run_name="__main__")
