import csv, sys
src, out = sys.argv[1:3]
rows = list(csv.DictReader(open(src, newline='', encoding='utf-8')))
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule'])
    for r in rows: w.writerow([r['id0'], r['value'], 'sr:' + r['rule']])
