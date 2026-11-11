import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")


def ok(x):
    return x is not None and np.isfinite(x) and x > 0


d = pd.read_csv(f'{R}/submissions/{SUB}_detail.csv').set_index('Unnamed: 0')
d['s0'] = d.answer_source.astype(str).str.replace('mo_guard:', '', regex=False)
d['mog'] = d.answer_source.astype(str).str.startswith('mo_guard:')
out = {}
for i, r in d.iterrows():
    q = str(r['question']).lower()
    if r['cat'] == 'S2' and r.s0 in ('agree_wgm', 'agree_gm') and r['kind'] in ('speed', 'accel', 'disp') and ok(r.g) and not NONRIGID.search(q):
        if not r.mog and str(r['video_id']).startswith('simulation'):
            out[str(i)] = {'value': float(r.g), 'why': f'rul275: S2 rigid-target {r["kind"]} agreement row ({r.s0}) -> geo {r.g:.4g} (was {r.answer:.4g})'}
json.dump(out, open(f'{R}/cache/tmp/ag_rul275_patch_test.json', 'w'), indent=1)
print('')
