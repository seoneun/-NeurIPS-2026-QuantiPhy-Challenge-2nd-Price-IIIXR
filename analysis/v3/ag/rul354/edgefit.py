import numpy as np
from trk import load
W, H = 853, 480


def edge_fit(df, axis='x', t0=0, t1=99):
    c = 'bx' if axis == 'x' else 'by'; s = 'bw' if axis == 'x' else 'bh'; lim = W if axis == 'x' else H
    lo = df[c] - df[s] / 2; hi = df[c] + df[s] / 2
    fwd = (df[c].iloc[-1] - df[c].iloc[0]) > 0
    e = hi if fwd else lo
    ok = (e > 3) & (e < lim - 3) & (df.t >= t0) & (df.t <= t1)
    t = df.t[ok].values; y = e[ok].values
    p = np.polyfit(t, y, 2)
    rms = float(np.sqrt(np.mean((np.polyval(p, t) - y) ** 2)))
    return dict(a=abs(2 * p[0]), v0=p[1] * (1 if fwd else -1), n=int(ok.sum()), rms=rms, p=p, fwd=fwd,
                tr=(round(float(t.min()), 2), round(float(t.max()), 2)))


def show(v, ph, axis='x', t0=0, t1=99, A=None, tq=()):
    (fps, d), o = load(v)
    df = o[ph]
    f = edge_fit(df, axis, t0, t1)
    msg = f'{v} "{ph}" lead-edge({axis}) n={f["n"]} t={f["tr"]} a_px={f["a"]:.1f} v0_px={f["v0"]:.1f} v0/a={f["v0"] / f["a"]:.3f}s rms={f["rms"]:.2f}'
    print(msg)
    return f, df


f, df = show('simulation_0262', 'red cup', 'y', 0, 0.70)
s = 9.8 / f['a']
inside = df[(df.by - df.bh / 2 > 3) & (df.by + df.bh / 2 < H - 3)]
print('')
for tq in [0.25, 0.5, 0.625]:
    print('')
for v in ['simulation_0157', 'simulation_0158']:
    ft, dt = show(v, 'turtle', 'x')
    st = 2.639 / ft['a']
    print('')
    if v == 'simulation_0158':
        for (a, b) in [(0, 99), (0, 1.0), (0, 0.8), (0.4, 1.2)]:
            fs, ds = show(v, 'shark', 'x', a, b)
            print('')
        ins = ds[(ds.bx - ds.bw / 2 > 3) & (ds.bx + ds.bw / 2 < W - 3)]
        print('')
        lo = ds.bx - ds.bw / 2; hi = ds.bx + ds.bw / 2
        print('   shark per-frame t/x0/x1:', ' '.join(f'{t:.2f}:{a:.0f}-{b:.0f}' for t, a, b in zip(ds.t.values[::3], lo.values[::3], hi.values[::3])))
(fps, d), o = load('simulation_0159')
for ph, d0, d1 in [('turtle', 15.232, 14.968), ('shark', 16.474, 15.744)]:
    df = o[ph]
    print(f'0159 {ph}:', ' '.join(f'{t:.2f}:{x0:.0f}-{x1:.0f}/{h:.0f}' for t, x0, x1, h in zip(df.t.values[:16:2], (df.bx - df.bw / 2).values[:16:2], (df.bx + df.bw / 2).values[:16:2], df.bh.values[:16:2])))
