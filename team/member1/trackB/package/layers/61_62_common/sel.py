import csv, json, sys
work, ids, lay = sys.argv[1:4]
P, rows = {}, []
for r in csv.DictReader(open(ids)):
    i, t = int(r['id0']), r['rule']
    if t not in P:
        P[t] = json.load(open(f'{work}/ov/cache/tmp/ag_{t}_patch_test.json'))
    rows.append((i, float(P[t][str(i)]['value']), t))
with open(lay, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); [w.writerow([i, repr(v), t]) for i, v, t in rows]
