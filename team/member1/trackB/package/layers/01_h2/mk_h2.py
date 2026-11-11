import csv, sys
h1, ver, z, out = sys.argv[1:5]
rd = lambda p: list(csv.DictReader(open(p, newline='', encoding='utf-8')))
H = {int(r['id0']): r for r in rd(h1)}
V = {int(r['id0']): r['verdict'] for r in rd(ver)}
Z = {int(r['id']) - 1: r['parsed_value'] for r in rd(z)}
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule'])
    for i, r in sorted(H.items()):
        if r['reason'] == 'S' and V.get(i) == 'X':
            w.writerow([i, repr(float(Z[i])), 'S_rows_verdict=X -> z1s2c'])
