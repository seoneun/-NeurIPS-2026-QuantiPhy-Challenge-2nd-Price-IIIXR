import sys, json, io, contextlib, numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
TOOL5 = f'{R}/analysis/v3/ag/rul389/frozen' if R5FZ else f'{R}/cache/tmp/r5_tool'
CT5 = f'{R}/analysis/v3/ag/rul389/frozen' if R5FZ else f'{R}/cache/tmp'
OUT = f'{R}/analysis/v3/ag/rul389'
sys.path.insert(0, OUT)
from smoothstep_fit import fit, kin_model, track

FPS, DT = 24.0, 0.369

def cart():
    res = {}
    for cam, tags in {'B (0152/0153)': ['s152ft', 's153ft', 's153rt'], 'A (0151)': ['s151ft', 's151rt']}.items():
        vs, as_, sizes = [], [], []
        fixed = None
        for tag in tags:
            video = 'simulation_0' + tag[1:4]
            k, x, dpx = track(tag, video, drop=(7, 8, 9) if tag == 's151ft' else ())
            e = pd.read_csv(f'{TOOL5}/tracks_test_{tag}/{video}_extents.csv')
            e = e[e.box_h.notna() & (e.border.fillna(0) == 0)]
            e = e[np.abs(e.box_h / np.median(e.box_h) - 1) < 0.12]
            sizes += [float(np.median(e.box_h)), float(np.median(e.box_w))]
            x = np.sign(x[-1] - x[0]) * x
            if tag == 's151rt':
                b = fit(k, x, fix=fixed)
            else:
                b = fit(k, x)
                if tag == 's151ft':
                    fixed = (b[1], b[2])
            v, a = kin_model(b, 24)
            vs.append(v); as_.append(a)
        s = DT / np.mean(sizes)
        res[cam] = (np.mean(vs) * FPS * s, np.mean(as_) * FPS * FPS * s, np.mean(sizes))
    v1 = float(np.mean([r[0] for r in res.values()])); a1 = float(np.mean([r[1] for r in res.values()]))
    return round(v1, 4), round(a1, 4)

def car0149():
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(open(f'{R}/analysis/v3/ag/rul392/garage_r5.py').read(), ns)
    k, s = np.asarray(ns['k49'], float), np.asarray(ns['s49'], float)
    b = fit(k, s); v12, a12 = kin_model(b, 12)
    return round(a12 * FPS * FPS, 4)

def main():
    F = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
    A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv').set_index('id0')
    v1, a1 = cart()
    a05 = car0149()
    rock = float(json.load(open(f'{CT5}/ag_rul395_patch_test.json'))['1170']['value'])
    fork = float(F.loc[1449, 'c26c'])
    plat = float(np.mean([A.loc[52, 'g'], A.loc[61, 'g']]))
    for i, new in ((1539, a1), (2014, a1), (1544, a1), (1538, v1), (2013, v1), (1522, a05), (1169, rock), (1751, rock),
                   (1489, fork), (1371, plat)):
        print(f'{i},{new}')


if __name__ == '__main__':
    main()
