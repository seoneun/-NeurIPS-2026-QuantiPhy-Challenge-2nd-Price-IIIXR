"""uncut2r100_chain (copyrules, 2026-10-17): the rul441 chain in one process, for rerun_shadow.sh (convention argv1).
  1. analysis/v3/ag/rul439/diag.py test, then val (unchanged production script; label-free, base-free: reads the raw
     geo table's meas_* columns and the track files) -> cache/tmp/ag_uncut2_diag_{test,val}.csv;
  2. analysis/v3/compliance/rules/copyrules/uncut2r100_c.py SUB with env GEO_T = the base's geo table (uncut_geot.geo_t)
     -> cache/tmp/ag_uncut2r100_patch_{test,val}.json.
Why a wrapper: the strict re-run of rul441 failed with FileNotFoundError cache/tmp/ag_uncut2_diag_test.csv (no
production copy of the diag csv exists in cache/tmp), and rerun_shadow.sh resets the shadow cache/tmp after every TSV line.
usage: python analysis/v3/compliance/rules/copyrules/uncut2r100_chain.py [SUB=sub_xt_Nz]"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(HERE, "../../../../.."))
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nz"
sys.path.insert(0, HERE)
from uncut_geot import geo_t  # noqa: E402

gt, n = geo_t(R, SUB)
print(f"base {SUB}: geo table {gt} (rows with detail g == table geo: {n})", flush=True)
env = dict(os.environ, GEO_T=gt, QP_ROOT=R, OMP_NUM_THREADS="2")
for split in ("test", "val"):
    p = subprocess.run([sys.executable, f"{R}/analysis/v3/ag/rul439/diag.py", split], env=env, cwd=R)
    assert p.returncode == 0, f"diag.py {split} rc={p.returncode}"
KEEP = os.environ.get("COPYRULES_KEEP")              # analysis only: keep a copy of the diag csvs outside the shadow
if KEEP:
    import shutil
    os.makedirs(KEEP, exist_ok=True)
    for split in ("test", "val"):
        shutil.copy(f"{R}/cache/tmp/ag_uncut2_diag_{split}.csv", f"{KEEP}/ag_uncut2_diag_{split}.csv")
p = subprocess.run([sys.executable, f"{HERE}/uncut2r100_c.py", SUB], env=env, cwd=R)
sys.exit(p.returncode)
