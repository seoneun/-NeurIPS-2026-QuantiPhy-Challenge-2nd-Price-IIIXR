import json, csv, sys
PF, I, OUT = sys.argv[1:4]
base = {int(r['id']) - 1: float(r['parsed_value']) for r in csv.DictReader(open(I + '/pred_t070_id1.csv', newline='', encoding='utf-8'))}
src = {int(r['id0']): r for r in csv.DictReader(open(I + '/pred_t070_id1_src.csv'))}
out = []
for k, v in json.load(open(PF)).items():
    i = int(k); x = float(v['value'])
    if src[i]['src'] == 'base_p17' or abs(x / base[i] - 1) < 1e-9 or src[i]['cat'] != 'S2': continue
    out.append((i, x, 'bindmoverS3'))
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for i, x, t in sorted(out): w.writerow([i, repr(float(x)), t])
