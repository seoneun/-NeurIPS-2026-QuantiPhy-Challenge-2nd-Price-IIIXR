import json, numpy as np, pandas as pd
R = '@QP_ROOT@/'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}cache/tmp/r5'
A = 4.05
def compute(verbose=False):
    d = json.load(open(R + 'geo/out/tracks_test_v3ntmv/captured_0002.json'))
    c = {k: e['st']['c'] for k, e in enumerate(d['objs']['0']) if isinstance(e, dict) and e.get('st') and e['st'].get('c')}
    ks = sorted(c); cy = np.array([c[k][1] for k in ks]); cx = np.array([c[k][0] for k in ks])
    rest = np.median([c[k][1] for k in ks if 10 <= k <= 24])
    step = {k: np.hypot(c[k][0] - c[k - 1][0], c[k][1] - c[k - 1][1]) for k in ks if k - 1 in c}
    k_on = min(k for k in ks if k > 24 and c[k][1] - rest > 1.5)
    k_post = min(k for k in ks if k > k_on + 3 and k in step and k - 1 in step and step[k] < 0.6 * step[k - 1])
    k_end = k_post - 1
    fit_k = [k for k in ks if k_on <= k <= k_end]
    sq = np.sqrt([c[k][1] - rest for k in fit_k]); p = np.polyfit(fit_k, sq, 1); k0 = -p[1] / p[0]
    t0 = k0 / 24.0; t_end = k_end / 24.0
    v1 = A * (t_end - t0); v1_lo = A * (t_end - 0.5 / 24 - t0); v1_hi = A * (t_end + 0.5 / 24 - t0)
    L = pd.read_csv(T5 + '/labelled_rows.csv').set_index('id0').loc[2146, 'y'] / 100.0
    v2 = np.sqrt(2 * A * L)
    val = np.sqrt(v1 * v2) * 100
    if verbose:
        print('')
        print('')
        print('')
    return {2149: val}
if __name__ == '__main__':
    compute(True)
