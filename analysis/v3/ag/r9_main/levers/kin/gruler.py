import numpy as np, pandas as pd
G, FPS = 9.8, 24.0
def est(vid):
    d = pd.read_csv(f'output/work/kin/otrk_{vid}.csv')
    top, bot, ar = d.top.values.astype(float), d.bot.values.astype(float), d.area.values.astype(float)
    held = np.median(ar[:11]); base = np.median(top[:11])
    on = int(np.argmax(top - base > 8))
    c = on
    while c + 1 < len(bot) and bot[c + 1] >= bot[c] - 5: c += 1
    ap = c + 1
    while ap + 1 < len(bot) and bot[ap + 1] <= bot[ap] + 1: ap += 1
    f = np.arange(c + 1, min(ap + 3, len(bot)))
    q = np.polyfit(f, bot[f], 2); apx = 2 * q[0]
    s_f = apx / (G / FPS ** 2)
    D = np.sqrt(ar); good = ar >= 0.7 * held
    idx = np.arange(len(D)); D = np.interp(idx, idx[good], D[good])
    D_f = D[f].mean()
    s = s_f * D / D_f
    steps = np.arange(0, c)
    hb = float(np.sum(np.diff(bot[0:c + 1]) / (0.5 * (s[steps] + s[steps + 1]))))
    return round(100 * hb, 1)
