"""tedge2_c.py [SUB]: runnable builder for tag rul432 (border-cut target speed, free edge in place of the mask centroid).

Why this file exists (copyrules task, agent rigidmo, 2026-10-17):
  - analysis/v3/compliance/s_builders.tsv maps rul432 to analysis/v3/ag/tlight/tlight_rule.py.  That script is the D3
    traffic-light height rule; it writes cache/tmp/ag_tlight_patch_*.json and never ag_tedge2_patch_*.json.  The real
    rul432 chain (scripts/pipeline/05_patch_rules.sh lines 169-172) is
        analysis/v3/ag/tedge/diag.py val ; diag.py test ; analysis/v3/ag/tedge/rule.py sub_xt_Nze ;
        analysis/v3/ag/derive/derive_tedge2.py   (tedge rows minus rows whose base answer source is sruler_geo)
    The chain must run inside one process because rerun_shadow.sh resets the shadow cache/tmp after every TSV line.
  - diag.py and rule.py read the geo raw table preds/<split>_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv (the table of the
    production bases).  rule.py gate f_g requires the base detail's g to equal that table's geo_value.  A strict base
    built by strict_build.sh on GEO_T=sil7 (sub_xt_cand14f) carries the sil7 geo_value, which differs from the rj5sQ2
    geo_value on many copy rows (e.g. test row 2977: 68.098 vs 68.968), so f_g failed there for a reason that is a
    property of the base's geo table, not of the row.  This wrapper reads the raw table of the base instead:
    env TE_RAW (e.g. v4ntsil7srngsr) or, when unset, the candidate table (v4ntrj5sQ2srngsr, v4ntsil7srngsr) whose
    geo_value equals the base detail's g on the most test rows (counts are printed).
  - derive_tedge2.py hard-codes sub_xt_Nze for the sruler_geo exclusion; here the exclusion reads SUB.
Changes to the exec'd production sources (diag.py, rule.py): only the raw-table name string.  Gates, fits and fusion
are unchanged.  Default SUB sub_xt_Nze + table rj5sQ2 = the production run (control).
Writes cache/tmp/ag_tedge_diag_{val,test}.csv, cache/tmp/ag_tedge_patch_{test,val}.json and
cache/tmp/ag_tedge2_patch_{test,val}.json under QP_ROOT (run it only through analysis/v3/compliance/rerun_shadow.sh).
"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nze"
OLD = "v4ntrj5sQ2srngsr"
CAND = ["v4ntrj5sQ2srngsr", "v4ntsil7srngsr"]


def pick_raw():
    if os.environ.get("TE_RAW"):
        return os.environ["TE_RAW"], "env TE_RAW"
    det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv", index_col=0)
    cnt = {}
    for c in CAND:
        g = pd.read_csv(f"{R}/preds/test_geo_{c}_d_vw_raw.csv", index_col=0, usecols=["Unnamed: 0", "geo_value"]).geo_value.reindex(det.index)
        cnt[c] = int(((det.g - g).abs() <= 1e-9 * g.abs().clip(lower=1)).sum())
    best = max(CAND, key=lambda c: cnt[c])
    return best, f"exact g matches on test rows {cnt}"


RAW, how = pick_raw()
print(f"tedge2_c: SUB={SUB} raw table={RAW} ({how})", flush=True)


EXEC = ("import sys; p, old, new = sys.argv[1:4]; s = open(p).read(); assert s.count(old) == 1, (p, s.count(old)); "
        "s = s.replace(old, new); sys.argv = [p] + sys.argv[4:]; "
        "exec(compile(s, p, 'exec'), {'__name__': '__main__', '__file__': p})")


def run_src(path, args):
    """run a production script in its own process, with only the raw-table name replaced (separate processes, as the
    original pipeline ran diag val / diag test / rule as separate commands, so no module state is shared)"""
    subprocess.run([sys.executable, "-c", EXEC, path, OLD, RAW] + args, check=True)


import subprocess
AG = f"{R}/analysis/v3/ag"
for sp in ("val", "test"):
    run_src(f"{AG}/tedge/diag.py", [sp])
run_src(f"{AG}/tedge/rule.py", [SUB])
# derive_tedge2 with the base SUB: tedge rows minus rows whose base answer source is sruler_geo
for s, det in (("test", f"{SUB}_detail"), ("val", f"{SUB}_valdetail")):
    D = pd.read_csv(f"{R}/submissions/{det}.csv").set_index("Unnamed: 0")
    t = json.load(open(f"{R}/cache/tmp/ag_tedge_patch_{s}.json"))
    o = {k: v for k, v in t.items() if str(D.answer_source.get(int(k), "")).replace("mo_guard:", "") != "sruler_geo"}
    json.dump(o, open(f"{R}/cache/tmp/ag_tedge2_patch_{s}.json", "w"), indent=0)
    print("rul432", s, len(t), "->", len(o), sorted(o, key=int))
