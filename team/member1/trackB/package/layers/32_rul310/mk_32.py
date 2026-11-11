import csv, sys
h1, a, b, out = sys.argv[1:5]
rd = lambda p: list(csv.DictReader(open(p)))
H = {int(r['id0']): r for r in rd(h1)}
L1 = [(int(r['id0']), H[int(r['id0'])]['p17'], 'lenient_ownvalue:' + r['rule']) for fn in (a, b) for r in rd(fn)]
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); w.writerows(L1)
