import csv, json, re, math, sys
HS, H28, H28S, LS, H1, OUT = sys.argv[1:7]
cur = {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(H28, newline='', encoding='utf-8'))}
src = {int(r['id0']): r['src'] for r in csv.DictReader(open(H28S, encoding='utf-8'))}
H = {int(r['id0']): r for r in csv.DictReader(open(H1, encoding='utf-8'))}
C = {}
d = json.load(open(HS))
for k, v in sorted(d.items(), key=lambda x: int(x[0])):
    i = int(k); m = re.search(r'\(r ([0-9.]+) vs', v['why']); w = re.search(r'was ([0-9.]+)', v['why'])
    r = float(m.group(1)); was = float(w.group(1)); new = v['value']
    lim = abs(math.log(r)) + math.log(1.25); fire = abs(math.log(new / was)) > lim
    if fire and 'rul302' in src[i]:
        C[i] = (was, 'rul302_gate: output/pre-rule %.3f beyond own scale r=%.2f x1.25 -> pre-rule value' % (new / was, r))
for r in csv.DictReader(open(LS)):
    if r['sp'] != 'test': continue
    i = int(r['id']); n = int(r['npass'])
    if i not in H: continue
    if 'lane_vp' in src.get(i, '') and n < 6:
        C[i] = (float(H[i]['z1s2c']), 'lane_vp_seedgate: own RANSAC passes %d/8 seeds (<6) -> own z1s2c' % n)
    elif n >= 6 and abs(float(r['Wmed']) / float(cur[i]) - 1) > 0.005:
        C[i] = (float(r['Wmed']), 'lane_vp_seedgate: own RANSAC passes %d/8 seeds (>=6) -> median W' % n)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for i, (v, rule) in sorted(C.items()): w.writerow([i, repr(float(v)), rule])
print(len(C), 'rows ->', OUT)
