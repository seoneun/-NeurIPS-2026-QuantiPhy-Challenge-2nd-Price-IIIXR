"""H4 check: what the probe3 kinoffset block (84 rows; best18v a*t -> a*(t-1/fps)) should have read on the portal under
several label conventions, against the reading (probe3 - best17v: D2 .848-.851, D3 .808-.809). best18v = best17v on all
scored rows (BRIEF_r1). y conventions (k = round(t*fps), tk = k/fps):
  H0 y = a t;  Hhalf y = a (tk - 0.5/fps);  Hfull y = a (t - 1/fps);  Hffbd y = a (tk - t0 - 0.5/fps), t0 = f0/fps from our
  own onset fit (kinoffset why strings, tracks v3ntmv/v2sc).  Mixtures: drop clips 0198-0209 under one, the rest under another."""
import json, re, sys
import numpy as np, pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
sys.path.insert(0, f'{R}/analysis/v3/calib'); import fusion_calib as FC
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv')
T = pd.read_parquet(f'{R}/data/test/test_dataset.parquet')
b18 = pd.read_csv(f'{R}/submissions/sub_xt_best18v_id1.csv').parsed_value.values
p3 = pd.read_csv(f'{R}/submissions/sub_xt_probe3_id1.csv').parsed_value.values
sc = M.scored.values.astype(bool); cat = M['cat'].values
d = ~np.isclose(b18, p3, rtol=1e-9) & sc
print('probe3 vs best18v changed scored rows per cat:', pd.Series(cat[d]).value_counts().to_dict())
ko = json.load(open(f'{R}/cache/tmp/ag_kinoffset_patch_test.json'))
ACC = re.compile(r"acc[a-z]*\s+(?:of\s+(?:the\s+)?(.+?))?\s*(?:(?:before|at)\s+[\d.]+\s*s\s*)?=\s*(-?[\d.]+)\s*m/s", re.I)
TQ = re.compile(r"(?:at\s+(?:time\s+)?|time\s+)([\d.]+)\s*s", re.I)
rows = []
for k, v in ko.items():
    i = int(k); q = A.question[i]; pr = A.prior[i]; fps = float(T.fps.iloc[i])
    t = float(TQ.search(q).group(1)); m = re.search(r'a \(t - 1/fps\) = ([\d.]+) x', v['why'])
    a = float(m.group(1)) if m else float(ACC.search(str(pr)).group(2))
    f0 = re.search(r'f0 (-?[\d.]+) frames', v['why']); f0 = float(f0.group(1)) if f0 else np.nan
    rows.append(dict(id0=i, video=A.video[i], cat=cat[i], t=t, fps=fps, a=a, f0=f0, b18=b18[i], p3=p3[i],
                     drop=A.video[i][:15] in [f'simulation_0{n}' for n in range(198, 210)]))
K = pd.DataFrame(rows); K['tk'] = np.round(K.t * K.fps) / K.fps
print('rows per cat:', K.groupby(['cat', 'drop']).size().to_dict(), '| b18==a*t:', np.allclose(K.b18, K.a * K.t, rtol=1e-3))
print('f0 (frames) per drop video:', K[K['drop']].groupby('video').f0.first().round(2).to_dict())
print('f0 of other rows: n=%d nan=%d median %.2f min %.2f max %.2f' % ((~K['drop']).sum(), K[~K['drop']].f0.isna().sum(),
      K[~K['drop']].f0.median(), K[~K['drop']].f0.min(), K[~K['drop']].f0.max()))
f0 = K.f0.fillna(0).values
Y = dict(H0=K.a * K.t, Hhalf=K.a * (K.tk - 0.5 / K.fps), Hfull=K.a * (K.t - 1 / K.fps),
         Hffbd=K.a * (K.tk - f0 / K.fps - 0.5 / K.fps), Hffbd_t0pos=K.a * (K.tk - np.clip(f0, 0, None) / K.fps - 0.5 / K.fps))
N = dict(D2=1109, D3=881)
def delta(y):
    g = FC.row_mra(K.p3.values, y.values) - FC.row_mra(K.b18.values, y.values)
    return {c: round(float(g[K.cat.values == c].sum()), 2) for c in ('D2', 'D3')}
print('\nobserved (rounded portal): D2 -0.003 -> row-MRA in [-4.4,-2.2]; D3 -0.001 -> [-1.8, 0.0]')
for h, y in Y.items():
    dd = delta(y); print(f'all rows {h:12s} row-MRA {dd}  -> cat pts', {c: round(dd[c] / N[c], 4) for c in dd})
for hd in ('Hhalf', 'Hffbd', 'Hfull'):
    for ho in ('H0', 'Hhalf'):
        y = Y[ho].where(~K['drop'], Y[hd]); dd = delta(y)
        print(f'drop rows {hd:6s} + other rows {ho:6s}: row-MRA {dd} -> cat pts', {c: round(dd[c] / N[c], 4) for c in dd})
K.to_csv(f'{R}/analysis/v3/ag/r3_simconv/h4_kinoffset_rows.csv', index=False)
