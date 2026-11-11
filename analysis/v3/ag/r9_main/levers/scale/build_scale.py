import sys, numpy as np, pandas as pd
s = 5.4 / 215.8
BL, BR, FL, FR = np.array([2887, 506]), np.array([3806, 1187]), np.array([2587, 931]), np.array([3494, 1594])
kb_long = np.mean([np.linalg.norm(BR - BL), np.linalg.norm(FR - FL)]); kb_short = np.mean([np.linalg.norm(FL - BL), np.linalg.norm(FR - BR)])
tip = np.loadtxt(sys.argv[1])
fps = 22.0; k0 = int(round(0.4 * fps)); xy = tip[:, 1:3]
v_px = np.mean([np.linalg.norm(xy[k0 + k] - xy[k0 - k]) / (2 * k / fps) for k in (1, 2)])
seg_scale = 27.9 / np.mean([np.hypot(1190 - 900, 371 - 161), np.hypot(1082.5 - 811, 496 - 290)])
seg_cont = np.mean([122.3, 123.5]) * seg_scale
rows = [(2084, kb_long * s), (2085, kb_short * s), (2087, v_px * s), (2965, seg_cont)]
pd.DataFrame(rows, columns=['id0', 'value']).to_csv(sys.argv[2], index=False)
