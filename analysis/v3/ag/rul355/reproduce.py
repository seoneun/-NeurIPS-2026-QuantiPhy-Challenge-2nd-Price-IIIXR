import io, json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
NG = f'{R}/analysis/v3/ag/rul316'
FROZEN = os.path.join(D, 'frozen', 'rul316_internet_0041.json')
VID = 'internet_0041'


def gpu_tracks():
    path = f'{NG}/rul316_trk.py'
    src = open(path).read()
    key = 'OUT = f"{R}/cache/tmp/rul316_tracks_{split}"; os.makedirs(OUT, exist_ok=True)'
    assert key in src
    tmp = tempfile.mkdtemp()
    src = src.replace(key, f'OUT = {tmp!r}')
    sys.path[:0] = [NG, f'{R}/geo']
    sys.argv = [path, '--only', VID]
    exec(compile(src, path, 'exec'), dict(__name__='__main__', __file__=path))
    T = reduce_tracks(json.load(open(f'{tmp}/{VID}.json')))
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(T, open(FROZEN, 'w'))
    print('wrote', FROZEN)


def reduce_tracks(T):
    keep = ('c', 'box', 'rr', 'border', 'area')
    inst = [dict({k: v for k, v in m.items() if k not in ('st', 'poly')},
                 st=[{k: s[k] for k in keep if k in s} if s else None for s in m['st']])
            for m in T['inst'] if m['phrase'] in ('duck', 'arch')]
    return dict({k: v for k, v in T.items() if k != 'inst'}, inst=inst)


def collected(x):
    return float(pd.read_csv(io.StringIO(f'v\n{float(x)!r}\n')).v.iloc[0])


def main():
    sys.path.insert(0, NG)
    import rul316_lib as L
    import rul316_meas as NM
    T = json.load(open(FROZEN))
    p = [json.loads(l) for l in open(f'{R}/geo/out/parse_test_v2.jsonl')]
    pr = [x for x in next(q for q in p if q['id'] == 3055)['priors'] if x.get('value_si')][0]
    m, why = NM.pick_inst(T, str(pr.get('obj') or '').lower())
    e, n = L.ext_px(L.ser_inst(T, m), pr.get('dim'))
    duck_px = float('%.1f' % e)
    arch = []
    for m in L.instances(T, 'arch'):
        st = [s for s in m['st'] if s]
        b = np.array([s['box'] for s in st], float)
        w, h, cy = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1]), np.median((b[:, 1] + b[:, 3]) / 2)
        if w < 200:
            arch.append((cy, w, h))
    up = [a for a in arch if a[0] < 180]
    gr = [a for a in arch if a[0] >= 180]
    h_up = float(np.median([a[2] for a in up])); h_gr = float(np.median([a[2] for a in gr]))
    mpp = pr['value_si'] / duck_px
    r1 = h_up * mpp * 100
    print('')
    with open(os.path.join(D, 'reproduce_out.csv'), 'w') as fo:
        fo.write('id0,value\n')
        fo.write(f'3058,{collected(round(r1, 2))!r}\n')
    print('3058 unrounded', repr(r1), 'written', repr(collected(round(r1, 2))))


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_tracks(); sys.exit(0)
    if '--rerun' in sys.argv:
        subprocess.run(['flock', f'{R}/cache/tmp/r4_gpu.lock', os.path.expanduser('~/anaconda3/envs/baba/bin/python'),
                        os.path.abspath(__file__), '--gpu'], check=True)
    main()
