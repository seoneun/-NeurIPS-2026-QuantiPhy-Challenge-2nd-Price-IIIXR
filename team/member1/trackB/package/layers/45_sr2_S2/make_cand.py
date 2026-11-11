import json, sys, csv
H19F, SRCF, DETF, OUT = sys.argv[1:5]
H19 = {int(r['id']) - 1: float(r['parsed_value']) for r in csv.DictReader(open(H19F, encoding='utf-8'))}
S = {int(r['id0']): r for r in csv.DictReader(open(SRCF))}
DET = {}
with open(DETF) as f:
    rd = csv.reader(f); hdr = next(rd); ai = hdr.index('answer')
    for r in rd: DET[int(r[0])] = float(r[ai]) if r[ai] not in ('', 'nan') else float('nan')
keep = []
for a in sys.argv[5:]:
    tag, p = a.split('=', 1)
    for k, e in json.load(open(p)).items():
        i = int(k)
        if i not in S or S[i]['cat'] != 'S2': continue
        cur = H19[i]; new = float(e['value']); inp = DET[i]
        own = not S[i]['src'].startswith('base_p17')
        ok = own or abs(inp - cur) <= 1e-6 * abs(cur)
        if ok and abs(new - cur) > 1e-9 * abs(cur): keep.append((i, new, tag))
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for i, v, t in sorted(keep): w.writerow([i, repr(v), t])
