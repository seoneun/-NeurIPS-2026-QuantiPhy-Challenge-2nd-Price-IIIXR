"""tedge3_pre_diag.py WHICH SPLIT: pre-stage of the tedge family for the clean re-run (check_repair grpB, 2026-10-17).
WHICH = tedge: analysis/v3/ag/tedge/diag.py SPLIT -> cache/tmp/ag_tedge_diag_<SPLIT>.csv (scripts/pipeline/05_patch_rules.sh:169-170)
WHICH = own:   env TEDGE3_OWN=1 TEDGE3_TAG=own analysis/v3/ag/rul433/diag_own.py SPLIT -> cache/tmp/ag_tedge3_diagown_<SPLIT>.csv
               (05_patch_rules.sh:230-231)
Both run unchanged under grpB_ownpool.py: each row's target / prior-object track lookup uses only the row's own-text queries
(colour_assign of measure_v3.Video restricted to them). Inputs per row: its parse record (geo/out/parse_<split>_v2.jsonl),
its own raw geo row (preds/<split>_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv, = the clean geo ffmv in the shadow), the track of its
own target phrase key (and of its own prior object) in the video's track file. Writes no patch."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpB_ownpool as OP  # noqa: E402
which, split = sys.argv[1], sys.argv[2]
if which == "tedge":
    OP.run("analysis/v3/ag/tedge/diag.py", [split])
elif which == "own":
    OP.run("analysis/v3/ag/rul433/diag_own.py", [split], env=dict(TEDGE3_OWN="1", TEDGE3_TAG="own"))
else:
    sys.exit("WHICH must be tedge or own")
