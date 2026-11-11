import sys, csv
import pandas as pd
I, H1, W = sys.argv[1:4]
z = {int(r['id0']): float(r['z1s2c']) for r in csv.DictReader(open(H1))}
H = pd.read_csv(I + '/pred_t070_id1.csv'); h = {int(i) - 1: float(v) for i, v in zip(H.id, H.parsed_value)}
d = pd.read_csv(f'{I}/pred_t029_detail.csv', index_col=0)
d['answer'] = [h[int(i)] for i in d.index]
def src(i, r):
    a, g, c = r.answer, r.g, r.cs
    if i in z and not (abs(a / z[i] - 1) < 1e-6): return 'patch:h23'
    if not (g > 0): return 'caw_nogeo'
    if abs(a / g - 1) < 1e-3: return 'geo'
    if c > 0 and abs(a / c - 1) < 1e-3: return 'caw_c1'
    if c > 0 and min(g, c) < a < max(g, c): return 'agree_wgm'
    return 'fusion'
d['answer_source'] = [src(i, r) for i, r in d.iterrows()]
d.to_csv(f'{W}/pred_t030_detail.csv')
d = pd.read_csv(f'{W}/pred_t030_detail.csv', index_col=0); d.index.name = None; d.to_csv(f'{W}/pred_t030_detail.csv', index_label='')
d = pd.read_csv(f'{W}/pred_t030_detail.csv', index_col=0); f = pd.read_csv(f'{I}/pred_t170_detail.csv', index_col=0)
for c in f.columns:
    if c not in d.columns: d[c] = f[c].reindex(d.index)
d.index.name = None; d.to_csv(f'{W}/pred_t031_detail.csv', index_label='')
