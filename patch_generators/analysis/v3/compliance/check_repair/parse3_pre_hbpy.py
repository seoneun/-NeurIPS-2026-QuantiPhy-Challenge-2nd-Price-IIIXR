"""parse3_pre_hbpy.py: python launcher used as PYTHON by parse3_rep.py for strict_build.sh (scripts/common.sh: PY=$PYTHON).
For analysis/v3/submit_v3.py it wraps kinematic.class_bound with the rul327 hand bound (analysis/v3/ag/rul327/handbound.py,
imported read-only; install() = the production run_fusion.py wrap) and runs the script unmodified; every other command
(tools/mo_guard.py, 'python -' with a here-document) is executed with plain python."""
import os, sys, runpy
args = sys.argv[1:]
if not args or not args[0].endswith("submit_v3.py"):
    os.execvp("python", ["python"] + args)
script = os.path.abspath(args[0])
R = os.environ["QP_ROOT"]
sys.argv = [script] + args[1:]
sys.path.insert(0, os.path.dirname(script))
sys.path.insert(0, f"{R}/analysis/v3/ag/rul327")
import kinematic, handbound  # noqa: E402
handbound.install(kinematic)
print("[parse3_pre_hbpy] kinematic.class_bound wrapped with the rul327 hand bound", file=sys.stderr)
runpy.run_path(script, run_name="__main__")
