import importlib.util, json, sys
import numpy as np, pandas as pd

R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
D = R + '/analysis/v3/ag/rul384/'
def block_a():
    spec = importlib.util.spec_from_file_location('gf', R + '/analysis/v3/ag/rul241/gflb_lib_copy.py')
    gf = importlib.util.module_from_spec(spec); a = sys.argv; sys.argv = ['x', 'pred_t111']; spec.loader.exec_module(gf); sys.argv = a
    taps = {}
    for v in ['captured_0015', 'captured_0015s', 'captured_0015x']:
        J = json.load(open(f'{R}/geo/out/tracks_test_v2sc/{v}.json'))
        oid = [q.get('obj', 1) for q in J['queries'] if q['key'].startswith('toy')][0]
        o = J['objs'][str(oid)]; f = [i for i, s in enumerate(o) if s and s.get('st')]
        T = np.array(J['fidx'])[f] / float(J['fps']); Y = np.array([o[i]['st']['c'][1] for i in f])
        arcs = [(T[a], T[b], -p[1] / (2 * p[0])) for a, b, p, r2, rms in gf.fit_arcs(T, Y)]
        arc = [x for x in arcs if x[0] > 1.5][0]
        taps['sam_' + v] = arc[2]
    for f, ph in [('test_captured_0015_stuffed-toy_owlv2.csv', 'owlv2'), ('test_captured_0015_toy_gdino.csv', 'gdino')]:
        d = pd.read_csv(D + f)
        d = d[(d.y1 - d.y0) < 120].sort_values('score', ascending=False).groupby('frame').head(1)
        d = d[(d.frame >= 46) & (d.frame <= 63)]
        p = np.polyfit(d.frame.values / 24.0, (d.y0 + d.y1).values / 2, 2)
        taps['det_' + ph] = -p[1] / (2 * p[0])
    tap = float(np.median(list(taps.values())))
    return 100 * 9.8 * (2.6 - tap)


def main():
    F = pd.read_csv(T5 + '/free_rows.csv').set_index('id0')
    out = {}
    v = block_a()
    for i in (2269, 2274, 2279):
        out[i] = dict(value=v)
    for i, plain in ((1987, 833), (1988, 834), (1990, 836), (1991, 837)):
        out[i] = dict(value=float(F.loc[plain, 'c26c']))
    for i in sorted(out):
        print(i, round(out[i]['value'], 6))
    return out


if __name__ == '__main__':
    main()
