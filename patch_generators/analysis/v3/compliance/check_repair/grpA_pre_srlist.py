"""grpA_pre_srlist.py NUM ROUND VAR: pre-stage for the clean re-run (check_repair grpA, 2026-10-17).
Writes cache/tmp/sr<NUM>.sh inside the shadow root (QP_ROOT) with the patch list of round sub_xt_<ROUND>, i.e. the list
variable VAR of scripts/fusion_recipe.sh, in the format of scripts/pipeline/05_patch_rules.sh (wsr(), line 292; sr70 at
line 262). The rule scripts rul435 (sr70 = Nzk), rul021 (sr74 = Nzo), rul045 (sr78 = Nzs) and rul056 (sr82 = Nzw,
inferred from rul056.py's default SUB sub_xt_Nzw and rul057's entry in P_NZX = P_NZW + rul057) read this file
only to skip rows that another applied patch already sets (same row ids). Writes no patch.
Refuses to run outside a rerun_clean.sh shadow (QP_ROOT must end in /shadow)."""
import os, sys, subprocess
R = os.environ.get("QP_ROOT", "")
if not R or os.path.basename(os.path.normpath(R)) != "shadow":
    sys.exit("QP_ROOT is not a rerun_clean.sh shadow root: %r" % R)
num, rnd, var = sys.argv[1], sys.argv[2], sys.argv[3]
lst = subprocess.run(["bash", "-c", 'source scripts/fusion_recipe.sh >/dev/null 2>&1; printf "%s" "${' + var + '}"'],
                     cwd=R, capture_output=True, text=True, check=True).stdout.strip()
if not lst:
    sys.exit("empty list for %s" % var)
dst = os.path.join(R, "cache", "tmp", "sr%s.sh" % num)
if os.path.islink(dst) or not os.path.isdir(os.path.dirname(dst)) or os.path.islink(os.path.dirname(dst)):
    sys.exit("refusing to write through a symlink: %s" % dst)
with open(dst, "w") as f:
    f.write('# patch list of sub_xt_%s (our round script sr%s.sh), read by the rule scripts of the next round\n'
            'V_="%s||--patch_json %s"\n' % (rnd, num, rnd, lst))
print("wrote", dst, "round", rnd, "var", var, "tags", len(lst.split(",")))
