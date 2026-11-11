import json, numpy as np, pandas as pd
R = '@QP_ROOT@'
M = pd.read_csv(f'{R}/geo/out/meas_test_v2_v3m_rrj5sQ.csv')

def len_along_flight(vid, ids):
    d = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/{vid}.json'))
    b = np.array([x['st']['box'] + [x['st'].get('border', 0)] for x in d['objs']['0'] if x and x.get('st')], float)
    ok = (b[:, 4] == 0) & (b[:, 1] > 2) & (b[:, 3] < d['H'] - 3)
    h = np.median(b[ok, 3] - b[ok, 1]); mpp = M[M.id.isin(ids)].mpp.median()
    return h * mpp, h, mpp, int(ok.sum())

a = len_along_flight('simulation_0114', [232, 233, 234]); b = len_along_flight('simulation_0114_segmented', [1815, 1816, 1817])
new = (a[0] + b[0]) / 2
vals = {233: new, 1816: new}
for k, v in vals.items(): print(f'{k},{v:.6f}')
