"""P1 part 3: re-measure the P1 videos with geo/measure_v3.py exactly as the rj5sQ chain does (same env / args, read-only
import), restricted to the selected videos, once with the production person-focal table (ref, must reproduce production)
and once with the gated person focal added for the P1 videos (new).

P1 videos = prior identity collapse (collapse.py) AND gate fired (gate.py). For them FPERSON gets the gated focal; the
existing measure_v3 logic then replaces the moving-prior calibration when it disagrees with the person focal by > 1.25x.
usage: python analysis/v3/ag/p1pf/meas.py <split> <m|v|mu|vu> <ref|new>
writes cache/tmp/ag_p1pf_meas_<split>_<variant>_<mode>.csv (measure_v3's per-row output)"""
import os, sys, json
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
split, var, mode = sys.argv[1:4]
TRK = {"m": "v2sc", "v": "v3ntmv", "mu": "v2scu", "vu": "v3ntmvu"}[var]
ENV = dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", RL_SELF_GRAV="1", RL_SELF_GRAV_MODE="2d_gself", QP_FPERSON_MINN="2",
           QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1", QP_TSHIFT="1")
os.environ.update(ENV)
os.environ.pop("QP_FPERSON", None)          # the table is filled below instead of from geo/out/person_focal_<split>.json
sys.path.insert(0, f"{R}/geo")
import measure_v3 as MV  # noqa: E402

COL = json.load(open(f"{R}/cache/tmp/ag_p1pf_collapse_{split}.json"))
GATE = json.load(open(f"{R}/cache/tmp/ag_p1pf_gate_{split}.json")) if os.path.exists(f"{R}/cache/tmp/ag_p1pf_gate_{split}.json") else {}
P1 = sorted(v for v in COL if GATE.get(v, {}).get("ok"))
SEL = set(P1) | set(os.environ.get("P1PF_EXTRA_VIDS", "").split(",")) - {""}
PF = json.load(open(f"{R}/geo/out/person_focal_{split}.json"))
MV.FPERSON.update(PF)                       # production table (QP_FPERSON=1, MINN=2)
if mode == "new":
    for v in P1:
        g = GATE[v]
        MV.FPERSON[v] = dict(PF.get(v, {}), f_over_W=float(g["f_over_W"]), n=max(2, int(g["n"])), gate=g["why"])
print(f"[p1pf] {split} {var} ({TRK}) {mode}: P1 videos {P1}; measured videos {sorted(SEL)}; focal",
      {v: round(MV.FPERSON[v]["f_over_W"], 3) for v in P1 if v in MV.FPERSON}, flush=True)

_lst = MV.load_split_tracks


def _filt(split_, cfg):
    byv, vjs = _lst(split_, cfg)
    return {k: v for k, v in byv.items() if k in SEL}, vjs


MV.load_split_tracks = _filt
_to_csv = pd.DataFrame.to_csv


def _redir(self, path=None, *a, **k):
    if isinstance(path, str) and path.startswith(f"{R}/geo/out/"):
        path = f"{R}/cache/tmp/ag_p1pf_" + os.path.basename(path)
    return _to_csv(self, path, *a, **k)


pd.DataFrame.to_csv = _redir
tag = f"_{var}_{mode}"
sys.argv = ["measure_v3.py", "--split", split, "--fix", "abcdef", "--tag", tag, "--fdef_json", f"{R}/geo/out/focal_medians_test.json",
            "--moge", "--moge_opts", "fbrej", "--tracks", TRK]
MV.main()
src = f"{R}/cache/tmp/ag_p1pf_meas_{split}_v2{tag}.csv"
os.replace(src, f"{R}/cache/tmp/ag_p1pf_meas_{split}_{var}_{mode}.csv")
print("wrote", f"cache/tmp/ag_p1pf_meas_{split}_{var}_{mode}.csv")
