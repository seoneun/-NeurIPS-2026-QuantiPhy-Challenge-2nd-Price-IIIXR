import json, os
import numpy as np, pandas as pd

R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
CT = f'{R}/analysis/v3/ag/r5_left/frozen' if R5FZ else f'{R}/cache/tmp/r5_left'
AUD = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv').set_index('id0')


def trk(split_set, video):
    return json.load(open(f'{R}/geo/out/tracks_test_{split_set}/{video}.json'))


def st(tj, o, f):
    L = tj['objs'].get(str(o))
    return None if (L is None or f >= len(L) or L[f] is None) else L[f]['st']


def i74():
    V = 'internet_0074'; FPS_FILE = 23.976
    TV = trk('v3ntmv', V)

    def scale(f):
        s = st(TV, 0, f); return 6.1 / (0.5 * (min(s['rr'][:2]) + (s['box'][2] - s['box'][0])))

    def gap(a, b):
        dx = max(0, max(a[0], b[0]) - min(a[2], b[2])); dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return float(np.hypot(dx, dy))

    def d2(tj, oa, ob, f):
        a, b = st(tj, oa, f), st(tj, ob, f)
        cen = float(np.hypot(a['c'][0] - b['c'][0], a['c'][1] - b['c'][1])); g = gap(a['box'], b['box'])
        return (g if g < 0.5 * cen else cen), cen, g

    def net_y(f, x):
        s = st(TV, 3, f); cx, cy = s['c']; ang = s['rr'][2]
        slope = np.tan(np.radians(ang + 90.0)) if abs(ang) > 45 else np.tan(np.radians(ang))
        return cy + slope * (x - cx)

    res = {}
    f = int(round(1.0 * FPS_FILE)); k = scale(f)
    bx, by = st(TV, 2, f)['c']; p1 = abs(by - net_y(f, bx))
    res[3284] = dict(new=p1 * k)
    f = int(round(2.0 * FPS_FILE)); k = scale(f)
    c = {o: d2(TV, 2, o, f) for o in (1, 4)}; o = min(c, key=lambda q: c[q][1]); v, cen, g = c[o]
    res[3285] = dict(new=v * k)
    f = int(round(4.5 * FPS_FILE)); k = scale(f)
    v, cen, g = d2(TV, 1, 4, f)
    res[3288] = dict(new=v * k)
    return res


def s148():
    d = pd.read_csv(f'{CT}/det/test_simulation_0148_train-window_owlv2.csv'); d['w'] = d.x1 - d.x0; d['h'] = d.y1 - d.y0
    win = []
    for f in (8, 11, 14, 17):
        c = d[(d.frame == f) & (d.x0 > 1) & (d.x1 < 852) & (d.w > 100) & (d.w < 200) & (d.w > d.h)].sort_values('score', ascending=False)
        if len(c): win.append(float(c.w.iloc[0]))
    e = pd.read_csv(f'{CT}/s148win_extents.csv'); e = e[(e.area > 0) & (e.border == 0) & (e.box_w > 150)]
    win += [float(x) for x in e.box_w]
    wpx = float(np.median(win)); sc = 1.8 / wpx
    TM, TV = trk('v2sc', 'simulation_0148'), trk('v3ntmv', 'simulation_0148')
    ph = [s['box'][3] - s['box'][1] for tj, o in ((TM, 7), (TV, 3)) for f in range(26) for s in [st(tj, o, f)] if s]
    ppx = float(np.median(ph)); bound = ppx * sc
    a = AUD.loc[346]; vlm = float(np.sqrt(a.cs * a.qwen))
    return dict(new=float(np.sqrt(bound * vlm)))




def compute():
    a = i74(); b = s148()
    return {3284: a[3284]['new'], 3285: a[3285]['new'], 3288: a[3288]['new'], 346: b['new']}


