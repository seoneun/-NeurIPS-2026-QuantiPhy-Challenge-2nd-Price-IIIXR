import os, sys, json, subprocess, contextlib, io
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
FF = json.load(open(f'{R}/geo/out/file_fps_test.json'))
vids = sorted(k for k, e in FF.items() if k.startswith('internet') and not k.endswith('_segmented')
              and (e['csv'] - e['file']) / e['file'] >= 0.10)
out = f'{R}/cache/tmp/rul375/repro'; os.makedirs(out, exist_ok=True)
subprocess.run([sys.executable, f'{HERE}/framepattern.py', *vids], env={**os.environ, 'FP_OUT': out}, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(0, HERE)
import rule_fileclock as RF
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    d = RF.compute(f'{out}/fp_summary.csv')
for r in d.itertuples():
    print(r.id0, r.new)
