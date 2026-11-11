import contextlib, io, json, os, runpy
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(f'{R}/analysis/v3/ag/rul396/route2.py', run_name='__main__')
patch = json.load(open(f'{R}/cache/tmp/ag_rul396_patch_test.json'))
for k, v in patch.items():
    print(k, v['value'])
