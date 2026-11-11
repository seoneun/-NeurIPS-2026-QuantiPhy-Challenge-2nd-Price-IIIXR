import sys, numpy as np
sys.path.insert(0, '@QP_ROOT@/analysis/v3/ag/r5_hard')
import slope_scale as S
tb = S.table()
a, b = tb[('simulation_0185', 'v3ntmv')], tb[('simulation_0186', 'v3ntmv')]
d0, d05 = S.DEPTH['simulation_0186']; dlin = lambda t: d0 + (d05 - d0) * t / 0.5
tfall = b['land'] / 2 / S.FPS
R1 = a['v1'] * a['s_slope']; R2 = b['v1'] * b['s_fall'] * dlin(1.0) / dlin(tfall)
v1 = float(np.sqrt(R1 * R2)); K = v1 + 0.5 * 3.18; R1b = a['v15'] * a['s_slope']; v15 = float(np.sqrt(K * R1b))
print(f'1434 {v1:.4f}'); print(f'1435 {v15:.4f}')
