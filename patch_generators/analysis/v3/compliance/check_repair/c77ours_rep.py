"""c77ours_rep (check_repair grpD, 2026-10-17): analysis/v3/ag/rul143/build_c77ours.py build() unchanged (imported
read-only), with kin_plan() reading the parse from the repo's geo/out instead of the deleted worktree copy.
Why: build_c77ours.kin_plan() execs the teammate's kin2d.py prefix with __file__ in analysis/v3/ag/rul143/wt/scripts/ours,
so kin2d's rows() opens analysis/v3/ag/rul143/wt/geo/out/parse_<split>_v2.jsonl (kin2d.py:37, :120). That file is gone.
The teammate branch (commit d057b1d) holds the parse as intermediates/geo/out/parse_<split>_v2.jsonl.gz; its gunzipped
content has the same md5 as the repo's geo/out/parse_<split>_v2.jsonl (test 06e4920c..., val 9a8b1353...; checked
2026-10-17 with git show | gunzip | md5sum). Here kin2d's global R is set to the repo root after the exec, so rows() reads
<root>/geo/out/parse_<split>_v2.jsonl. kin_plan only routes rows (motion / pair / ends); values come from the frozen module
outputs in analysis/v3/ag/rul143/wt/geo/out/{scene3d_*,kin2d_*} as before.
Base: env C77_BASE (answer_source and answer: gate only).
usage: C77_BASE=sub_xt_<base> python analysis/v3/compliance/check_repair/c77ours_rep.py [test|val ...]
writes cache/tmp/ag_c77ours_patch_{test,val}.json"""
import os, sys
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SPLITS = sys.argv[1:] or ("test", "val")
sys.path.insert(0, f"{R}/analysis/v3/ag/rul143")
import build_c77ours as B                    # module top level: execs scene3d.py's plan(); writes nothing


def kin_plan(split):
    """build_c77ours.kin_plan with kin2d's R = repo root (parse file geo/out/parse_<split>_v2.jsonl)"""
    ks = open(f"{B.W}/scripts/ours/kin2d.py").read()
    sys.argv = ["kin2d.py", "--stage", "measure", "--split", split]
    kn = {"__file__": f"{B.W}/scripts/ours/kin2d.py"}
    exec(ks[:ks.index("# ----------------------------------------------------------------------------- locate (GPU)")], kn)
    kn["R"] = R
    out = {}
    for r in kn["rows"]():
        objs, ends, pairs = kn["plan"](r)
        out[r["id"]] = "motion" if objs else "pair" if pairs else "ends" if ends else None
    return out


B.kin_plan = kin_plan
print("base", B.BASE)
for s in SPLITS:
    B.build(s)
