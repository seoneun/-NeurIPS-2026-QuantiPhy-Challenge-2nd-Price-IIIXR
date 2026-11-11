import json, os, numpy as np, sys
sys.path.insert(0, os.path.dirname(__file__))
from anim_check import homog
R = '@QP_ROOT@'

def trk(video, phrase, ts, need_nb=False):
    d = json.load(open(f'{R}/geo/out/tracks_test_{ts}/{video}.json'))
    q = [x for x in d['queries'] if x['phrase'] == phrase and x.get('obj') is not None][0]
    T, C, B = [], [], []
    for fi, e in zip(d['fidx'], d['objs'][str(q['obj'])]):
        if not e or not e.get('st'): continue
        st = e['st']
        if need_nb and st.get('border', 0): continue
        T.append(fi / d['fps']); bx = st['box']; C.append(st['c']); B.append([(bx[0] + bx[2]) / 2, st['bottom']])
    return np.array(T), np.array(C, float), np.array(B, float)

def slope(T, P, lo, hi):
    m = (T >= lo - 1e-9) & (T <= hi + 1e-9)
    return np.array([np.polyfit(T[m], P[m, k], 1)[0] for k in (0, 1)]), m.sum()

def dog():
    out = {}
    for v in ('simulation_0213', 'simulation_0214', 'simulation_0217'):
        for ts in ('v2sc', 'v3ntmv'):
            Tc, Cc, _ = trk(v, 'black cat', ts, need_nb=True)
            mpp = 0.94 / np.linalg.norm(slope(Tc, Cc, -1, 99)[0])
            Td, Cd, _ = trk(v, 'dog', ts)
            r = [np.linalg.norm(slope(Td, Cd, t - 0.25, t + 0.25)[0]) * mpp for t in (1.0, 1.5)]
            out[(v, ts)] = r
            print('')
    for v in ('simulation_0214', 'simulation_0217'):
        TA, _, BA = trk('simulation_0212', 'dog', 'v2sc'); TB, CB, BB = trk(v, 'dog', 'v2sc')
        ok = (TA >= TB.min()) & (TA <= TB.max())
        pb = np.c_[np.interp(TA[ok], TB, CB[:, 0]), np.interp(TA[ok], TB, CB[:, 1])]
        a = BA[ok]; H = fitH(a, pb)
        pr = (H @ np.c_[a, np.ones(len(a))].T).T; pr = pr[:, :2] / pr[:, 2:3]
        Tc, Cc, _ = trk(v, 'black cat', 'v2sc', need_nb=True); mpp = 0.94 / np.linalg.norm(slope(Tc, Cc, -1, 99)[0])
        Tb, _, Bb = trk(v, 'dog', 'v2sc')
        r = [np.linalg.norm(slope(TA[ok], pr, t - 0.25, t + 0.25)[0]) * mpp for t in (1.0, 1.5)]
        Tb, Cb, _ = trk(v, 'dog', 'v2sc'); r0 = [np.linalg.norm(slope(Tb, Cb, t - 0.25, t + 0.25)[0]) * mpp for t in (1.0, 1.5)]
        print('')
    return out

def fitH(a, b):
    def norm(p):
        m = p.mean(0); s = np.sqrt(2) / np.mean(np.linalg.norm(p - m, axis=1))
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    Ta, Tb = norm(a), norm(b)
    ah = (Ta @ np.c_[a, np.ones(len(a))].T).T; bh = (Tb @ np.c_[b, np.ones(len(b))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(ah, bh):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    return np.linalg.inv(Tb) @ H @ Ta

def box(video, phrase, a_stated, t0, ts='v3ntmv', w=0.125):
    T, C, _ = trk(video, phrase, ts)
    tt = T[(T >= T.min() + w) & (T <= T.max() - w)]
    v = np.array([np.linalg.norm(slope(T, C, t - w, t + w)[0]) for t in tt])
    f = T <= 0.30 + 1e-9
    P = C[f] - C[f].mean(0); u = np.linalg.svd(P)[2][0]; p = P @ u
    aff = abs(2 * np.polyfit(T[f], p, 2)[0])
    ff = 9.8 / aff
    m = tt >= t0
    apx = np.polyfit(tt[m], v[m], 1)[0]
    mpp = a_stated / apx
    vi = lambda t: np.interp(t, tt, v) * mpp
    print('')
    print(f'    stated-acc scale: t_end {T.max():.3f}  a_px[{t0},end] {apx:.1f} px/s2  m/px {mpp:.5f}  v(1.0) {vi(1.0):.3f}  v(1.5) {vi(1.5):.3f}'
          f'  | v_px(t) ' + ' '.join(f'{t:.2f}:{x:.0f}' for t, x in zip(tt[::3], v[::3])))
    return vi(1.0), vi(1.5)

if __name__ == '__main__':
    dog()
    for v, p, a, t0 in [('simulation_0183', 'tool box', 1.58, 1.25), ('simulation_0184', 'tool box', 1.58, 1.25),
                        ('simulation_0185', 'gift box', 3.18, 1.0), ('simulation_0186', 'gift box', 3.18, 1.0),
                        ('simulation_0181', 'box', 2.338, 0.75), ('simulation_0182', 'box', 2.338, 0.75)]:
        box(v, p, a, t0)
