"""pendacc_up_chain: runs the production chain unchanged on one base: analysis/v3/ag/pendacc/pendacc.py (writes
cache/tmp/ag_pendacc_patch_{test,val}.json, plus pendaccspec / pendacccap and scan CSVs), then
analysis/v3/ag/derive_late/rul332.py (reads ag_pendacc_patch_*, writes ag_pendacc_up_patch_*).
Why a wrapper (2026-10-17, copyrules helper labd3): rul332.py alone fails with FileNotFoundError because
rerun_shadow.sh resets the shadow cache/tmp after every TSV line, so the producer and the rule must run in one line.
No rule code is changed.  usage (through rerun_shadow.sh only): python pendacc_up_chain.py sub_xt_<BASE>"""
import os, sys, subprocess
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
base = sys.argv[1]
for f in ("analysis/v3/ag/pendacc/pendacc.py", "analysis/v3/ag/derive_late/rul332.py"):
    print("== run", f, base, flush=True)
    rc = subprocess.call([sys.executable, f"{R}/{f}", base])
    if rc:
        sys.exit(rc)
