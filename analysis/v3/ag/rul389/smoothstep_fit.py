import io, contextlib, numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
TOOL5 = f'{R}/analysis/v3/ag/rul389/frozen' if R5FZ else f'{R}/cache/tmp/r5_tool'
FPS = 24.0

def S(u):
    u = np.clip(u, 0, 1); return 3 * u ** 2 - 2 * u ** 3

def fit(k, x, k0s=np.arange(-30, 12.01, 0.25), Ks=np.arange(24, 160.01, 0.5), fix=None):
    best = None
    grid = [fix] if fix else [(a, b) for a in k0s for b in Ks]
    for k0, K in grid:
        A = np.c_[np.ones_like(k), S((k - k0) / K)]
        p, *_ = np.linalg.lstsq(A, x, rcond=None)
        r = np.sqrt(np.mean((A @ p - x) ** 2))
        if best is None or r < best[0]:
            best = (r, k0, K, p)
    return best

def kin_model(b, kq):
    r, k0, K, p = b
    xs = p[0] + p[1] * S((np.array([kq - 2, kq - 1, kq]) - k0) / K)
    return xs[2] - xs[1], (xs[2] - xs[1]) - (xs[1] - xs[0])

def kin_poly(k, x, kq, deg):
    pp = np.polyfit(k, x, deg); xs = np.polyval(pp, [kq - 2, kq - 1, kq])
    return xs[2] - xs[1], (xs[2] - xs[1]) - (xs[1] - xs[0]), np.sqrt(np.mean((np.polyval(pp, k) - x) ** 2))

def track(tag, video, lo=0, hi=99, drop=()):
    e = pd.read_csv(f'{TOOL5}/tracks_test_{tag}/{video}_extents.csv')
    e = e[e.box_h.notna() & (e.border.fillna(0) == 0)]
    med = np.median(e.box_h); e = e[np.abs(e.box_h / med - 1) < 0.12]
    e = e[(e.frame >= lo) & (e.frame <= hi) & ~e.frame.isin(drop)]
    return e.frame.to_numpy(float), ((e.x0 + e.x1) / 2).to_numpy(float), float(np.median(e.box_h))

def report():
    print('')
    cart = {}
    for tag, video, lo, hi, drop in [('s153ft', 'simulation_0153', 0, 99, ()), ('s152ft', 'simulation_0152', 0, 99, ()),
                                     ('s153rt', 'simulation_0153', 0, 99, ()), ('s151rt', 'simulation_0151', 0, 99, ()),
                                     ('s151ft', 'simulation_0151', 0, 99, (7, 8, 9))]:
        k, x, dpx = track(tag, video, lo, hi, drop)
        sgn = np.sign(x[-1] - x[0]); x = sgn * x
        b = fit(k, x)
        cart[tag] = (k, x, dpx, b)
        for scale_name, s in [('own', 0.369 / dpx), ('common', 0.369 / 49.5)]:
            v24, a24 = kin_model(b, 24); v12, a12 = kin_model(b, 12)
            print('')
        for deg in (3, 4):
            if len(k) > 8:
                v24, a24, rr = kin_poly(k, x, 24, deg)
                print('')
    b153 = cart['s153ft'][3]
    for tag in ('s151rt', 's151ft'):
        k, x, dpx, bfree = cart[tag]
        bf = fit(k, x, fix=(b153[1], b153[2]))
        s = 0.369 / dpx; v24, a24 = kin_model(bf, 24)
        print('')

    print('')
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(open(f'{R}/analysis/v3/ag/rul392/garage_r5.py').read(), ns)
    for name, kk, ss in [('0150 composite', ns['k50'], ns['s50']), ('0149 3D front tire', ns['k49'], ns['s49'])]:
        kk = np.asarray(kk, float); ss = np.asarray(ss, float)
        b = fit(kk, ss)
        out = [f'{name}: frames {int(kk.min())}-{int(kk.max())} n {len(kk)} smoothstep k0 {b[1]:.2f} K {b[2]:.1f} D {b[3][1]:.2f} m rms {b[0]:.3f} m']
        for kq in (12, 24):
            v, a = kin_model(b, kq); out.append(f'k{kq}: v {v*FPS:.3f} a {a*FPS*FPS:.3f}')
        print(' | '.join(out))
        for deg in (2, 3, 4):
            r = [kin_poly(kk, ss, kq, deg) for kq in (12, 24)]
            print('')


if __name__ == '__main__':
    report()
