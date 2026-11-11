"""c20tedge3d_pre_diag3d.py SPLIT: pre-stage of rul015 for the clean re-run (check_repair grpB, 2026-10-17).
Runs analysis/v3/ag/rul015/diag3d.py SPLIT unchanged (05_patch_rules.sh:296-297) under grpB_ownpool.py (own-text
track lookup) -> cache/tmp/ag_c20tedge3d_diag_<SPLIT>.csv. Reads rul433's own-track diag cache/tmp/ag_tedge3_diagown_<SPLIT>.csv
(written by tedge3_pre_diag.py own SPLIT earlier in the same run), the row's parse record (own depth keys) and own raw geo row."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
OP.run("analysis/v3/ag/rul015/diag3d.py", [sys.argv[1]])
