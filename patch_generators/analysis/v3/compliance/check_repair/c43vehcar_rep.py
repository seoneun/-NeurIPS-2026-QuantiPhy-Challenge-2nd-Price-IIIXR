"""c43vehcar_rep (check_repair grpD, 2026-10-17): runs analysis/v3/ag/rul068/rul068.py's run() unchanged (imported
read-only) for test and val, without the original __main__ block.
Why: the original __main__ reads cache/tmp/ag_a42s0_fuzzycopies.csv (rul068.py:104) before calling run(); that file does
not exist, so the clean re-run stopped with FileNotFoundError before any patch was written. The file is used only for the
evaluation printout (rul068.py:118-130: label lookup of fuzzy copies) and is not read by run() (rul068.py:73-96),
which builds and writes cache/tmp/ag_c43vehcar_patch_{test,val}.json. The evaluation printout (labels, fuzzy copies,
same-question labels) is dropped.
Base: env C43_SUB (read by rul068.py:29 when rule_dim is imported).
usage: C43_SUB=sub_xt_<base> python analysis/v3/compliance/check_repair/c43vehcar_rep.py"""
import os, sys
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/analysis/v3/ag/rul068")
import rul068 as C43                      # module top level: imports rule_dim with argv [C43_SUB]; writes nothing

for sp in ("test", "val"):
    out, skipped, S, PR, ext = C43.run(sp)    # writes cache/tmp/ag_c43vehcar_patch_<sp>.json
    print(f"== {sp}: base {C43.RD.SUB}; {len(out)} rows changed; skipped {len(skipped)}")
    Si = S.set_index("id")
    for k, e in out.items():
        print(f"  CHANGE {k} {Si.vid[int(k)]} {e['value']:.4g} | {e['why']}")
    for i, w in skipped.items():
        print(f"  SKIP {i} {w}")
