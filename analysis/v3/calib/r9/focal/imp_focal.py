import sys, pandas as pd
base, out = sys.argv[1], sys.argv[2]
pool = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
f = pd.read_csv('analysis/v3/calib/r9/focal/in/focal_moge2.csv').set_index('video_id').f_moge2
n = pd.read_csv(base)
rows = []
for i in (1062, 1308, 1320, 1065, 1540):
    r = n.iloc[i]
    rows.append((i, r.parsed_value * (pool[r.video_source] / f[r.video_id]) ** 0.5))
pd.DataFrame(rows, columns=['id0', 'value']).to_csv(out, index=False)
