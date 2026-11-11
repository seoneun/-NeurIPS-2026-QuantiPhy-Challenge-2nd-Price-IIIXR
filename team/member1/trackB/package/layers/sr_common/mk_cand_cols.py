import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='')))
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\r\n' if sys.argv[3:] == ['crlf'] else '\n'); w.writerow(['id0', 'value', 'rule'])
    for r in rows: w.writerow([r['id0'], r['value'], r['rule']])
