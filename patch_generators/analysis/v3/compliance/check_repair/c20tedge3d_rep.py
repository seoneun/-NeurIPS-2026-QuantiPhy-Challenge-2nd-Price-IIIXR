"""c20tedge3d_rep.py [SUB FRAC RUNTOL TAG VAL]: analysis/v3/ag/rul015/rule.py run unchanged (check_repair grpB, 2026-10-17)
with the skip-if-patched gate of its base fed from the shadow's re-run patches (grpB_ownpool.install_skip, round list P_NZN:
production base sub_xt_Nzn, 05_patch_rules.sh:292-298). Pre-stages earlier in the same run: c20tedge3d_pre_diag3d.py
{val,test} and grpA_pre_srlist.py 73 Nzn P_NZN (cache/tmp/sr73.sh, read only by the evaluation printout).
Writes cache/tmp/ag_c20tedge3d_patch_{test,val}.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul015/rule.py", sys.argv[1:], skip=("rul015", "P_NZN"))
