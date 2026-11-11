"""c69segrun_pre_diag.py SPLIT: pre-stage of rul127 for the clean re-run (check_repair grpB, 2026-10-17).
Runs analysis/v3/ag/rul127/diag.py SPLIT unchanged under grpB_ownpool.py (own-text track lookup) ->
cache/tmp/ag_c69segrun_diag_<SPLIT>.csv. rul127 is not in scripts/pipeline/05_patch_rules.sh (rule after 112); the stage
is inferred from rul127/rule.py:121 (reads that file) and diag.py's docstring. Reads rul433's own-track diag
cache/tmp/ag_tedge3_diagown_<SPLIT>.csv (tedge3_pre_diag.py own SPLIT, earlier in the same run). Writes no patch."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul127/diag.py", [sys.argv[1]])
