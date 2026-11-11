"""slopechord_loose_c: runs the production rul421 builder analysis/v3/ag/rul421/rul421.py in its
'loose' variant, which is how patches/ag_slopechord_loose_patch_*.json were made (README rule 76: rul421 for slope
ends keyed with a suffix, keys containing _far / _near): env SC_KEYS=loose SC_TAG=rul422 SC_SUB=<base>,
once per split (test, val).  Writes cache/tmp/ag_slopechord_loose_patch_{test,val}.json.
Why a wrapper (2026-10-17, copyrules helper labd3): s_builders.tsv lists rul422 with no builder ('-'),
because the variant is selected by environment variables, not by its own file.  No rule code is changed.
usage (through rerun_shadow.sh only): python slopechord_loose_c.py sub_xt_<BASE>"""
import os, sys, subprocess
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
base = sys.argv[1]
env = dict(os.environ, SC_KEYS="loose", SC_TAG="rul422", SC_SUB=base)
for split in ("test", "val"):
    print("== run rul421.py", split, base, "SC_KEYS=loose", flush=True)
    rc = subprocess.call([sys.executable, f"{R}/analysis/v3/ag/rul421/rul421.py", split], env=env)
    if rc:
        sys.exit(rc)
