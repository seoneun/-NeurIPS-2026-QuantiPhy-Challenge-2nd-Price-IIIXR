import json, os, re
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
AUD = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv').set_index('id0')


def num_m(text, word):
    m = re.search(word + r'[^=]*=\s*([0-9.]+)\s*m\b', str(text))
    return float(m.group(1))


def dragon_994():
    js = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0283.json'))
    fps = js['fps']
    m = pd.read_csv(f'{R}/geo/out/meas_test_v2_v3v_rntrj5sQ.csv')
    mpp = float(m[m.id == 994].mpp.iloc[0])
    F, C, H = [], [], []
    for pos, fr in enumerate(js['objs']['0']):
        if fr and fr.get('st') and not fr['st'].get('border'):
            F.append(js['fidx'][pos]); C.append(fr['st']['c']); H.append(fr['st']['box'][3] - fr['st']['box'][1])
    F = np.array(F); C = np.array(C, float); H = np.array(H, float)
    assert np.all(np.diff(F) == 1)
    h = H - H.mean()
    ac = [np.sum(h[:-k] * h[k:]) / np.sum(h * h) for k in range(1, 25)]
    P = int(np.argmax(ac[5:]) + 6)
    t0 = int(round(1.0 * fps))
    ker = np.ones(P) / P
    cs = np.stack([np.convolve(C[:, k], ker, mode='valid') for k in range(2)], 1)
    fc = F[:len(cs)] + (P - 1) / 2.0
    i = np.argmin(np.abs(fc - t0))
    v1 = np.linalg.norm(cs[i] - cs[i - 1]) * mpp * fps
    sel = (F >= t0 - P) & (F <= t0 + P)
    A = np.vstack([F[sel] / fps, np.ones(sel.sum())]).T
    sx = np.linalg.lstsq(A, C[sel, 0], rcond=None)[0][0]; sy = np.linalg.lstsq(A, C[sel, 1], rcond=None)[0][0]
    v2 = np.hypot(sx, sy) * mpp
    print('')
    return v1, v2, '%.4f'


def yellowcar_1541():
    tire_m = num_m(AUD.loc[1541, 'prior'], 'tire of the yellow car')
    js = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0152.json'))
    fps = js['fps']

    def tr(o):
        out = {}
        for pos, fr in enumerate(js['objs'][str(o)]):
            if fr and fr.get('st'):
                out[js['fidx'][pos]] = fr['st']
        return out
    car, tire = tr(2), tr(1)
    full = [f for f in range(28, 38) if f in tire and not tire[f]['border']
            and abs((tire[f]['box'][2] - tire[f]['box'][0]) - (tire[f]['box'][3] - tire[f]['box'][1])) <= 3]
    dpx = np.median([0.5 * ((tire[f]['box'][2] - tire[f]['box'][0]) + (tire[f]['box'][3] - tire[f]['box'][1])) for f in full])
    k = tire_m / dpx * fps
    F = np.array([f for f in range(20, 29) if f in car and not car[f]['border']])
    x2 = np.array([car[f]['box'][2] for f in F], float)
    p = np.polyfit(F / fps, x2, 2)
    v_fit = np.polyval(np.polyder(p), 1.0) / fps
    v1 = v_fit * k
    T = np.array(full, float)
    cx = np.array([0.5 * (tire[f]['box'][0] + tire[f]['box'][2]) for f in full])
    q = np.polyfit(T / fps, cx, 2)
    a2 = 2 * q[0] * tire_m / dpx
    lin = np.polyfit(T / fps, cx, 1)[0] * tire_m / dpx
    v2 = lin - a2 * (T.mean() / fps - 1.0) if a2 > 0 else lin
    print('')
    return v1, v2, '%.3f'


def star_1631():
    hgt = num_m(AUD.loc[1631, 'prior'], 'height')
    js = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0312.json'))
    bw, bh = [], []
    for pos, fr in enumerate(js['objs']['0']):
        if fr and fr.get('st') and not fr['st'].get('border'):
            b = fr['st']['box']; bw.append(b[2] - b[0]); bh.append(b[3] - b[1])
    r = np.array(bw, float) / np.array(bh, float)
    rm = np.array([np.median(r[max(0, i - 2): i + 3]) for i in range(len(r))])
    W1 = hgt * np.percentile(rm, 90)
    W2 = hgt * 2 * np.sin(np.radians(72)) / (1 + np.cos(np.radians(36)))
    print('')
    return W1, W2, '%.4f'


if __name__ == '__main__':
    out = []
    for i, fn in ((994, dragon_994), (1541, yellowcar_1541), (1631, star_1631)):
        v1, v2, fmt = fn()
        p1, p2 = float(fmt % v1), float(fmt % v2)
        assert abs(p1 - p2) / max(p1, p2) <= 0.15
        out.append((i, round(p1, 4), v1))
    with open(os.path.join(D, 'reproduce_out.csv'), 'w') as f:
        f.write('id0,value\n')
        for i, v, raw in out:
            f.write(f'{i},{float(v)!r}\n')
    for i, v, raw in out:
        print('')
