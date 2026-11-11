import csv, json, math, sys
J, H10, H1, OUT = sys.argv[1:5]
H = {int(r['id0']) for r in csv.DictReader(open(H1, encoding='utf-8'))}
cur = {int(r['id']) - 1: float(r['parsed_value']) for r in csv.DictReader(open(H10, newline='', encoding='utf-8'))}
up = {}
for l in open(J):
    d = json.loads(l)
    if d['split'] == 'test' and d['cond'] == 'up' and not d['alone']: up[int(d['original_id'])] = d['parsed_value']
rows = []
for i in sorted(up):
    u = up[i]; c = cur[i]
    new = math.sqrt(c * u) if u is not None and math.isfinite(u) and u > 0 else c
    if i in H: rows.append([i, repr(new), 'rul409_gm_upright_caw'])
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
print(len(rows), 'rows ->', OUT)
