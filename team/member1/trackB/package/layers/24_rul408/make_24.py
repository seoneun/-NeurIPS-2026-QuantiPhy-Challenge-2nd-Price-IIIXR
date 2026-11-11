import csv, sys
rows, h1, out = sys.argv[1:4]
H = {int(r['id0']): r for r in csv.DictReader(open(h1, encoding='utf-8'))}
with open(out, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule'])
    for r in csv.DictReader(open(rows)):
        i = int(r['id0']); w.writerow([i, repr(float(H[i]['p17'])), r['rule']])
