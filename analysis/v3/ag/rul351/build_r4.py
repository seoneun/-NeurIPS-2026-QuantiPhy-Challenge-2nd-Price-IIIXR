import argparse, json, os, subprocess, sys
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
KEYS = ['rul368', 'rul371', 'rul355', 'rul364', 'rul366',
        'rul353', 'rul361', 'rul363', 'rul374',
        'rul354', 'rul362', 'rul372', 'rul373',
        'rul352', 'rul356', 'rul365', 'rul367', 'rul370',
        'rul369', 'r4_main']
OUT = os.path.join(R, 'cache/tmp/r4_fromcode')


def main():
    env = dict(os.environ, R4_COMPUTED='1')
    os.makedirs(OUT, exist_ok=True)
    vals = {}
    for k in KEYS:
        d = os.path.join(R, 'analysis/v3/ag', k)
        subprocess.run([sys.executable, os.path.join(d, 'reproduce.py')], cwd=R, capture_output=True, text=True, env=env, check=True)
        o = pd.read_csv(os.path.join(d, 'reproduce_out.csv'), float_precision='round_trip')
        for i, v in zip(o.id0.astype(int), o.value.astype(float)):
            vals.setdefault(i, (k, v))
    X = pd.read_csv(os.path.join(R, 'analysis/v3/calib/r4/collect_rows.csv'))
    sel = X[X.sel]
    want = dict(zip(sel.id0.astype(int), sel['cat']))
    spec = ['name\tpred_t014', 'base\tpred_t007']
    for c in ['S2', 'D2', 'D3']:
        rows = {str(i): {'value': vals[i][1], 'why': vals[i][0]} for i in sorted(want) if want[i] == c}
        p = os.path.join(OUT, f'ag_r4_comp_{c}_patch_test.json')
        json.dump(rows, open(p, 'w'), indent=0)
        spec.append(f'block\tr4_{c}\t{c}\tjson:{os.path.relpath(p, R)}')
    open(os.path.join(OUT, 'pred_t014.tsv'), 'w').write('\n'.join(spec) + '\n')


if __name__ == '__main__':
    main()
