#!/usr/bin/env python
import glob, json, os, re, subprocess, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ORDER = ['S2', 'D2', 'S3', 'D3']
RUNPY = {
    'rul391': "g = runpy.run_path(P, run_name='r5_rebuild'); V = {str(k): v for k, v in g['main']()[0].items()}",
    'rul384': "g = runpy.run_path(P, run_name='r5_rebuild2'); V = {str(k): v['value'] for k, v in g['main']().items()}",
    'rul400': "g = runpy.run_path(P, run_name='__main__'); V = {k: v['value'] for k, v in g['out'].items()}",
    'rul398': "g = runpy.run_path(P, run_name='__main__'); V = {k: v['value'] for k, v in g['patch'].items()}",
}
NUM = re.compile(r'^\s*(\d+)\s*[, ]\s*(?:np\.float64\()?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?(?=\s|,|#|$)')


def run_sources(od, keys):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    procs = {}
    for k in keys:
        P = f'{R}/analysis/v3/ag/{k}/reproduce.py'
        if k in RUNPY:
            code = ('import json, os, runpy, sys; P, J = sys.argv[1], sys.argv[2]; sys.argv = [P]; '
                    'sys.path.insert(0, os.path.dirname(P)); ' + RUNPY[k] + '; json.dump(V, open(J, "w"))')
            cmd = [sys.executable, '-c', code, P, f'{od}/{k}_values.json']
        else:
            cmd = [sys.executable, P]
        procs[k] = subprocess.Popen(cmd, cwd=R, env=env, stdout=open(f'{od}/{k}.out', 'w'), stderr=open(f'{od}/{k}.err', 'w'))
    bad = [k for k, p in procs.items() if p.wait() != 0]
    if bad:
        sys.exit('')


def values(od, k, ids):
    if k in RUNPY:
        v = {int(i): float(x) for i, x in json.load(open(f'{od}/{k}_values.json')).items()}
        return {i: v[i] for i in ids if i in v}
    v = {}
    for line in open(f'{od}/{k}.out'):
        m = NUM.match(line)
        if m and int(m.group(1)) in ids:
            i, x = int(m.group(1)), float(m.group(2))
            if i in v and v[i] != x:
                sys.exit('')
            v[i] = x
    return v


def main():
    out = os.path.abspath(sys.argv[1])
    od = f'{out}/r5c'; os.makedirs(od, exist_ok=True)
    sel = pd.read_csv(f'{R}/analysis/v3/calib/r5/pred_t010_blocks.csv', usecols=['id0', 'cat', 'source'])
    run_sources(od, sorted(sel.source.unique()))
    V = {k: values(od, k, set(g.id0)) for k, g in sel.groupby('source')}
    sel['value'] = [V[k].get(i, np.nan) for k, i in zip(sel.source, sel.id0)]
    lines = ['name\tpred_t010', 'base\tpred_t009']
    for c, g in sel.groupby('cat'):
        jp = f'{od}/ag_r5_pred_t010_{c}_patch_test.json'
        json.dump({str(i): {'value': v, 'why': s} for i, v, s in zip(g.id0, g.value, g.source)}, open(jp, 'w'), indent=1)
        lines.append(f'block\tr5_{c}\t{c}\tjson:{os.path.relpath(jp, R)}')
    sp = f'{od}/pred_t010.tsv'
    open(sp, 'w').write('\n'.join(lines) + '\n')
    open(f'{od}/spec.txt', 'w').write(sp + '\n')


if __name__ == '__main__':
    main()
