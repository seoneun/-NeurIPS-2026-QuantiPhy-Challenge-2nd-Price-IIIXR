import contextlib, io, sys
import numpy as np
R = '@QP_ROOT@'
sys.path.insert(0, f'{R}/analysis/v3/ag/rul354')
with contextlib.redirect_stdout(io.StringIO()):
    from trk import load, fitpar
    from edgefit import edge_fit


def phx(v):
    (fps, d), o = load(v); df = o['phoenix']
    fe = edge_fit(df, 'x'); p0, v0, ac, rms = fitpar(df.t, -(df.cx - df.cx.iloc[0]))
    s = float(np.sqrt((2 / fe['a']) * (2 / ac)))
    return float(df.bh.rolling(3, center=True).median().max() * s)


if __name__ == '__main__':
    print(949, repr(phx('simulation_0269')))
