import csv, sys, os
D = os.path.dirname(os.path.abspath(__file__)); I = os.path.join(D, 'inputs')
LAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(I, 'layers.txt')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'pred_t016_id1.csv')
SRC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(D, 'h8_src_rebuilt.csv')
def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
rows = list(csv.reader(open(os.path.join(I, 'pred_t003_id1.csv'), newline='', encoding='utf-8')))
vi = rows[0].index('parsed_value')
P = {int(r[0]) - 1: r[vi] for r in rows[1:]}
Z = {int(r['id']) - 1: r['parsed_value'] for r in rd(os.path.join(I, 'pred_t004_id1.csv'))}
H = {int(r['id0']): r for r in rd(os.path.join(I, 'h1_rows.csv'))}
def f(x): return float(x)
val, src = {}, {}
for i in P:
    if i not in H: val[i] = None; src[i] = 'unscored_z1s2c'; continue
    h = H[i]
    if h['status'] != 'S': val[i] = f(Z[i]); src[i] = 'base_z1s2c:' + h['status']
    elif h['reason'] == 'S': val[i] = f(P[i]); src[i] = 'base_p17'
    else: val[i] = f(h['h1']); src[i] = 'base_h1chain:' + h['reason']
    assert f(h['h1']) == val[i]
for line in open(LAY, encoding='utf-8'):
    line = line.split('#')[0].strip()
    if not line: continue
    fn, mode = line.split()[:2]
    p = fn if os.path.isabs(fn) else os.path.join(I, 'cands', fn)
    n = 0
    for r in rd(p):
        i = int(r['id0'])
        if i not in H: print(''); continue
        if mode == 'if_base' and not src[i].startswith('base_'): continue
        val[i] = f(r['value']); src[i] = fn + '|' + r['rule']; n += 1
    print(f'{fn:28s} {mode:8s} applied {n}')
for r in rows[1:]:
    i = int(r[0]) - 1
    if src[i] == 'unscored_z1s2c': r[vi] = Z[i]
    elif src[i].startswith('base_'):
        if val[i] != f(P[i]): r[vi] = repr(val[i])
    else: r[vi] = repr(val[i])
with open(OUT, 'w', newline='', encoding='utf-8') as fo: csv.writer(fo, lineterminator='\n').writerows(rows)
with open(SRC, 'w', newline='', encoding='utf-8') as fo:
    w = csv.writer(fo, lineterminator='\n'); w.writerow(['id0', 'cat', 'src', 'value'])
    for i in sorted(P): w.writerow([i, H[i]['cat'] if i in H else '', src[i], Z[i] if val[i] is None else repr(val[i])])
print('wrote', OUT, SRC)
