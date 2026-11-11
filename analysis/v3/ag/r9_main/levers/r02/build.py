import sys, numpy as np, pandas as pd
base, prev, out = sys.argv[1:4]
D = 'analysis/v3/ag/r9_main/levers/r02/in'
AVG = {'agree_gm', 'agree_wgm', 'dis_size_gm', 'dis_size_low_gm', 'rul263_unstable_gm', 'rul263_dimgm', 'd3lab_gm',
       'agree_calib', 'dis_calib', 'c2a_mix'}
LEFT_OUT = {255, 3114}
b = pd.read_csv(base); b['id0'] = b.id - 1; cur = b.set_index('id0').parsed_value
p = pd.read_csv(prev); p['id0'] = p.id - 1; v0 = p.set_index('id0').parsed_value
changed = set(cur.index[~np.isclose(cur.values, v0.reindex(cur.index).values, rtol=1e-9, atol=0)])
kimi = set(pd.read_csv(f'{D}/kimi_rows.csv').id0)
F = pd.read_csv(f'{D}/free_rows.csv')
for c in ['g', 'cs', 'cfull', 'ans']: F[c] = pd.to_numeric(F[c], errors='coerce')
F['src'] = F.source.astype(str).str.split(':').str[0]
with np.errstate(all='ignore'):
    F['w'] = np.log(F.ans / F.cs) / np.log(F.g / F.cs)
    F['wf'] = np.log(F.ans / F.cfull) / np.log(F.g / F.cfull)
    okc = (F.src != 'rul263_dimgm') & ((np.abs(F.w - 0.5) < 0.02) | (np.abs(F.w - 0.65) < 0.02) | (np.abs(F.w - 0.7) < 0.02))
    okf = (F.src != 'rul263_dimgm') & ((np.abs(F.wf - 0.5) < 0.02) | (np.abs(F.wf - 0.65) < 0.02) | (np.abs(F.wf - 0.7) < 0.02))
    F['c_eff'] = np.where(okc, F.cs, np.where(okf, F.cfull, F.ans ** 2 / F.g))
    F['w_eff'] = np.where(okc, F.w, np.where(okf, F.wf, 0.5))
    F['lo'] = np.minimum(F.g, F.c_eff); F['hi'] = np.maximum(F.g, F.c_eff); F['r'] = F.hi / F.lo
F['avg'] = F.src.isin(AVG) & F.g.gt(0) & pd.Series(F.c_eff, index=F.index).gt(0) & (F.w_eff >= 0.2) & (F.w_eff <= 0.8)
F['cur'] = cur.reindex(F.id0).values
F['same'] = (np.abs(F.cur / F.ans - 1) < 1e-6) | np.any([np.abs(F.cur - F.ans.round(k)) < 1e-9 for k in (1, 2, 3, 4)], axis=0)
Q = F[F.avg & (F.r >= 1.5) & F.same & ~F.id0.isin(changed) & ~F.id0.isin(kimi) & (F.r <= 2.005) & ~F.id0.isin(LEFT_OUT)]
pd.DataFrame({'id0': Q.id0, 'value': Q.lo.map(lambda v: float('%.6g' % v))}).to_csv(out, index=False)
