"""c79limbedgef_rep.py [SUB]: analysis/v3/ag/rul146/rul146.py run unchanged under grpB_ownpool.py (check_repair grpB, 2026-10-17).
The builder reads the row's target track with measure_v3.Video(vj, "e").get(own target); grpB_ownpool restricts the lookup
to the row's own-text queries (colour_assign over other questions' phrases removed). Other inputs as the original: base
detail SUB (argv 1), own raw geo row, rul433's own-track diag cache/tmp/ag_tedge3_diagown_{test,val}.csv (pre-stage
tedge3_pre_diag.py own, earlier in the same run). Skip-if-patched gate (grpB_ownpool.install_skip; inferred: the round list of its default base sub_xt_Nzz24 is not defined in scripts/fusion_recipe.sh; the tags before rul146 in P_NZZ48A are used).
Writes cache/tmp/ag_c79limbedgef_patch_{test,val}.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul146/rul146.py", sys.argv[1:], skip=("rul146", "P_NZZ48A<"))
