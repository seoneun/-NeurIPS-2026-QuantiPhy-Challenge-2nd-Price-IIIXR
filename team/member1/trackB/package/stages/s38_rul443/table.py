import csv, sys
import pandas as pd

sub, rows, h37, out, parts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
RUN = {'pred_t059': 'pred_t194', 'pred_t114': 'pred_t193', 'pred_t113': 'pred_t193'}
ans = {r: pd.read_csv(f'{sub}/{r}_detail.csv').set_index('Unnamed: 0').answer for r in set(RUN.values())}
with open(h37, encoding='utf-8', newline='') as f:
    h = list(csv.reader(f))
k, ki = h[0].index('parsed_value'), h[0].index('id')
cur = {int(r[ki]) - 1: r[k] for r in h[1:]}
tab = {}
for t in pd.read_csv(rows).itertuples():
    v = repr(float(ans[RUN[t.writer]][t.id0]))
    if v != cur[t.id0]:
        tab[t.id0] = v
for p in parts:
    with open(p, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            tab[int(r['id0'])] = r['value']
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['id0', 'value'])
    w.writerows([i, tab[i]] for i in sorted(tab))
