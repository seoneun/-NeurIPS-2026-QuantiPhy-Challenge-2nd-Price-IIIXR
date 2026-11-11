"""grpC_strict.py: per-question re-fusion helper for the grpC re-fusion rules (rul328 rul099 rul044 rul329 rul327
rul024; notes/check_repair_grpC.md).

The production rules took the answer of a whole non-strict fusion run (production recipe with var / sib / lab_pool / twin
pooling) on a geo raw where the rule's rows were changed. Here the fusion is the clean fusion of build_clean4.sh step 1:
  GEO_T=<gt> bash analysis/v3/compliance/strict_build.sh chkrep_<tag> NONE --d3lab_focal "854x480=0.5959,853x480=0.5959:1.3:gm:gm"
run on copies of the clean geo tables preds/{test,val}_geo_v4nt<GEO_T>{,u}srngsr_d_vw_raw.csv (GEO_T env, default ffmv)
with the rule's rows changed (edit callback), written under the unique table name GEO_T=ck<tag> inside QP_ROOT.
Run it only inside a rerun_clean.sh shadow (QP_ROOT = shadow): strict_build.sh writes submissions/{,sub_}xt_chkrep_<tag>*.csv
(shadow copies) and, through the shadow's symlinks, logs/{xt,mg_xt}_chkrep_<tag>{,u}.log and
caw/out/v2caw/val124/xt_chkrep_<tag>{,u}.csv in the real repo (unique names).

run(tag, edit, env=None, python=None, both=False) -> dict split -> (new xt detail, base xt detail, base sub detail, changed ids)
  edit(split, u, raw) changes the DataFrame raw (index = row id) in place and returns the changed row ids; it is called for
  the main table only (u = ''), as the production rules changed only the main geo raw, unless both=True (u = 'u' too).
  The patch value is the MAIN fusion answer (submissions/xt_chkrep_<tag>_detail.csv), as in production, where the rule's
  fusion run was the main fusion and its patch rows are never touched by tools/mo_guard.py (patch sources are not
  agree / dis); the u fusion and mo_guard still run inside strict_build.sh.
  env: extra environment for strict_build.sh; python: value of PYTHON (scripts/common.sh: PY=$PYTHON).
  base = submissions/xt_<B>_{detail,valdetail}.csv (main fusion) and <BASE>_... (merged), BASE = argv[1] = sub_xt_<B>
  (default sub_xt_clean4f) = the strict fusion of clean4.
  Prints, per split, the number of rows outside the changed set whose answer differs from the base (expected 0: the
  strict fusion of a row reads only that row; this also checks that the shadow run reproduces the base).
gate(new, base, bsub, rows) -> rows whose main answer changed vs the main base (rtol 1e-6, as the production mkpatch.py
  files) and whose base answer_source (main or merged) does not start with 'patch' (clean4f has none: KEEP=NONE)."""
import os, re, sys, subprocess
import numpy as np, pandas as pd
R = os.environ.get("QP_ROOT")
assert R and "/shadow" in R, "run inside a rerun_clean.sh shadow (QP_ROOT=<OUTDIR>/shadow)"
GT0 = os.environ.get("GEO_T", "ffmv")
SPEC = "854x480=0.5959,853x480=0.5959:1.3:gm:gm"
BASE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("sub_xt_") else "sub_xt_clean4f"
SUF = {"test": "_detail.csv", "val": "_valdetail.csv"}


def raw_path(sp, gt, u):
    return f"{R}/preds/{sp}_geo_v4nt{gt}{u}srngsr_d_vw_raw.csv"


def load_raw(sp, u="", gt=None):
    return pd.read_csv(raw_path(sp, gt or GT0, u)).set_index("Unnamed: 0")


def detail(name, sp):
    return pd.read_csv(f"{R}/submissions/{name}{SUF[sp]}").set_index("Unnamed: 0")


def run(tag, edit, env=None, python=None, both=False):
    gt = "ck" + re.sub(r"[^A-Za-z0-9]", "", tag)
    name = f"chkrep_{tag}"
    chg = {}
    for sp in ("test", "val"):
        chg[sp] = set()
        for u in ("", "u"):
            raw = load_raw(sp, u)
            ids = edit(sp, u, raw) or [] if (u == "" or both) else []
            chg[sp] |= {int(i) for i in ids}
            raw.reset_index().to_csv(raw_path(sp, gt, u), index=False)
    e = dict(os.environ, GEO_T=gt, QP_ROOT=R)
    e.update(env or {})
    if python:
        e["PYTHON"] = python
    p = subprocess.run(["bash", f"{R}/analysis/v3/compliance/strict_build.sh", name, "NONE", "--d3lab_focal", SPEC],
                       cwd=R, env=e, capture_output=True, text=True)
    print(p.stdout[-1500:]); print(p.stderr[-3000:], file=sys.stderr)
    if p.returncode != 0:
        raise SystemExit(f"strict_build.sh {name} failed rc {p.returncode}")
    out = {}
    for sp in ("test", "val"):
        n, b, bs = detail(f"xt_{name}", sp), detail(BASE[4:], sp), detail(BASE, sp)
        n = n.reindex(b.index); bs = bs.reindex(b.index)
        d = ~np.isclose(n.answer.values, b.answer.values, rtol=1e-9, equal_nan=True)
        outside = [int(i) for i in b.index[d] if int(i) not in chg[sp]]
        print(f"[{tag} {sp}] geo rows changed {sorted(chg[sp])}; main answers differing from {BASE[4:]}: {int(d.sum())}; "
              f"of them outside the changed rows: {len(outside)} {outside[:10]}")
        out[sp] = (n, b, bs, sorted(chg[sp]))
    return out


def gate(n, b, bs, rows):
    keep = []
    for i in rows:
        if i not in b.index:
            continue
        if str(b.answer_source[i]).startswith("patch") or str(bs.answer_source[i]).startswith("patch"):
            continue
        if not np.isclose(float(n.answer[i]), float(b.answer[i]), rtol=1e-6):
            keep.append(i)
    return keep


def write_patch(tag, sp, P):
    import json
    p = f"{R}/cache/tmp/ag_{tag}_patch_{sp}.json"
    json.dump({str(int(k)): v for k, v in P.items()}, open(p, "w"), indent=0)
    print(f"[{tag} {sp}] wrote {len(P)} rows -> {p}")


def ownmeas(split, var, vids, suf, env=None, mode="ownt"):
    """per-row own-text measurement (grpC_pre_ownmeas.py = owntracks/meas.py + hooks) of the videos vids; returns
    (DataFrame by row id, diag dict). Output cache/tmp/ag_ownt_{meas,diag}_<split>_<var>_<mode><suf>.{csv,json} (shadow)."""
    import json
    e = dict(os.environ, QP_ROOT=R, OWNT_SUF=suf)
    e.update(env or {})
    p = subprocess.run(["python", f"{R}/analysis/v3/compliance/check_repair/grpC_pre_ownmeas.py", split, var, mode, ",".join(sorted(vids))],
                       cwd=R, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"grpC_pre_ownmeas.py {split} {var} {suf} failed rc {p.returncode}")
    d = pd.read_csv(f"{R}/cache/tmp/ag_ownt_meas_{split}_{var}_{mode}{suf}.csv").set_index("id")
    g = json.load(open(f"{R}/cache/tmp/ag_ownt_diag_{split}_{var}_{mode}{suf}.json"))
    return d, g
