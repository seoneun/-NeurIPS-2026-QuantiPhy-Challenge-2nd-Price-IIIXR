"""c69segrun_rep.py [SUB SLOPE_TOL TAG]: analysis/v3/ag/rul127/rule.py run unchanged (check_repair grpB, 2026-10-17)
with the skip-if-patched gate of its base fed from the shadow's re-run patches (grpB_ownpool.install_skip, round list P_NZZ18:
default base sub_xt_Nzz18 at rule.py:24; rul127 is a rule after 112, so the list is inferred from that default).
Pre-stage earlier in the same run: c69segrun_pre_diag.py {val,test}. Writes cache/tmp/ag_c69segrun_patch_{test,val}.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul127/rule.py", sys.argv[1:], skip=("rul127", "P_NZZ18"))
