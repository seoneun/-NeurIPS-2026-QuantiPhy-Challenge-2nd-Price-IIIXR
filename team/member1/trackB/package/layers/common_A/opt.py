import csv, re, sys, collections
P17, Z1, H1, H8, H8SRC, OUTA, OUTB = sys.argv[1:8]
def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8-sig')))
def fl(x):
    try: return float(x)
    except ValueError: return None
T = rd(P17); P = {int(r['id']) - 1: r['parsed_value'] for r in T}; vid = {int(r['id']) - 1: r['video_id'] for r in T}
Z = {int(r['id']) - 1: r['parsed_value'] for r in rd(Z1)}
H = {int(r['id0']): r for r in rd(H1)}
F = {int(r['id']) - 1: r['parsed_value'] for r in rd(H8)}
S = {int(r['id0']): r['src'] for r in rd(H8SRC)}
def rec(v): return re.sub(r'(_segmented|[abxsX]+)$', '', v)
def sig(x): return len(repr(abs(x)).lower().split('e')[0].replace('.', '').lstrip('0').rstrip('0'))
by_rec = collections.defaultdict(list)
for i, v in vid.items(): by_rec[rec(v)].append(i)
A, B = [], []
for i in sorted(F):
    v = fl(F[i]); eqZ = v == fl(Z[i])
    if S[i].startswith('base_h1chain') and not eqZ and v != fl(P[i]):
        B.append([i, Z[i], 'h1chain->z1s2c:' + H[i]['reason']])
    if eqZ or v is None: continue
    j = next((j for j in by_rec[rec(vid[i])] if j != i and vid[j] != vid[i] and fl(P[j]) == v), None)
    if j is not None and float(Z[i]) != float(Z[j]) and sig(v) >= 4:
        A.append([i, Z[i], 'copyfix:h8==sibling p17 (' + vid[j] + '), own z1s2c differs'])
for fn, rows in ((OUTA, A), (OUTB, B)):
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
