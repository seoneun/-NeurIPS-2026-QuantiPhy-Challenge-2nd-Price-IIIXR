"""tedge3_rep.py [SUB REP_TOL OLD_TOL TAG MODE]: analysis/v3/ag/rul433/rule.py run unchanged (check_repair grpB, 2026-10-17)
with the skip-if-patched gate of its base fed from the shadow's re-run patches (grpB_ownpool.install_skip, round list P_NZH:
production base sub_xt_Nzh, scripts/pipeline/05_patch_rules.sh:233-234). Pre-stages earlier in the same run:
tedge3_pre_diag.py {tedge,own} {val,test} (diag files) and grpA_pre_srlist.py 67 Nzh P_NZH (cache/tmp/sr67.sh, read only by
the evaluation printout at rule.py's end). Writes cache/tmp/ag_tedge3_patch_{test,val}.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul433/rule.py", sys.argv[1:], skip=("rul433", "P_NZH"))
