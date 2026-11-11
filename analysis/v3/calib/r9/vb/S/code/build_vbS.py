import sys, numpy as np, pandas as pd
long_px = np.mean([np.hypot(1190 - 900, 371 - 161), np.hypot(1082.5 - 811, 496 - 290)])
short_px = np.mean([np.hypot(900 - 811, 290 - 161), np.hypot(1190 - 1082.5, 496 - 371)])
s = 27.9 / long_px
pen_px = 182.7; cont_px = 131.5
tip = np.loadtxt(sys.argv[1])
xy = tip[:, 1:3]; fps = 30.0003; k0 = 12
v = np.mean([np.hypot(*(xy[k0 + k] - xy[k0 - k])) / (2 * k / fps) for k in (1, 2)])
rows = [(2965, cont_px * s), (2966, 27.9), (2967, short_px * s), (2968, pen_px * s), (2969, v * s)]
pd.DataFrame(rows, columns=['id0', 'value']).to_csv(sys.argv[2], index=False)
