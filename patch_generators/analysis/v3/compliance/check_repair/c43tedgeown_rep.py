"""c43tedgeown_rep.py [SUB]: analysis/v3/ag/rul066/rul066.py without its evaluation tail (check_repair grpB, 2026-10-17).
The original writes its patches (rul066.py:62) and then evaluates against labels and the fuzzy-copy list
cache/tmp/ag_a42s0_fuzzycopies.csv (rul066.py:66-78), which does not exist (checked 10-17: absent in the repo
cache/tmp) -> FileNotFoundError, rc 1. This copy executes the original source up to the line
'# evaluation: val + all-237 + fuzzy copies' (must occur once); every gate and value line is the original's.
Inputs: base detail SUB (argv 1), tedge diag cache/tmp/ag_tedge_diag_{split}.csv and rul433 own-track diag
cache/tmp/ag_tedge3_diagown_{split}.csv (pre-stage tedge3_pre_diag.py, earlier in the same run), the row's own parse record
and own raw geo row. Skip-if-patched gate: grpB_ownpool.install_skip with round list P_NZZ
(rows that an earlier tag of the base sub_xt_Nzz writes in the shadow's re-run patches count as already patched). Writes cache/tmp/ag_c43tedgeown_patch_{test,val}.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
p = f"{R}/analysis/v3/ag/rul066/rul066.py"
src = open(p).read()
mk = "\n# evaluation: val + all-237 + fuzzy copies\n"
assert src.count(mk) == 1, "evaluation marker not found exactly once"
sys.argv = [p] + sys.argv[1:]
OP.install_skip("rul066", "P_NZZ")   # production base sub_xt_Nzz (rul066.py:17): rows already patched there are skipped
G = {"__name__": "__main__", "__file__": p}
try:
    exec(compile(src.split(mk)[0], p, "exec"), G)
finally:
    OP.report()
