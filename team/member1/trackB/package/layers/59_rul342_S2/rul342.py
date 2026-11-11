import csv, math, sys
ROWS, H24, Z1, H1, CAND, OUT = sys.argv[1:7]
def load(p): return {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(p, newline='', encoding='utf-8'))}
H = {int(r['id0']): r for r in csv.DictReader(open(H1, encoding='utf-8'))}
base = load(H24); Z = load(Z1)
def f(x):
    try: v = float(x); return v if v > 0 and math.isfinite(v) else None
    except: return None
rows = []
for r in csv.DictReader(open(ROWS)):
    if r['sp'] != 'test': continue
    i = int(r['id']); g = f(r['s1_g']); q = f(r['q3']); cur = f(base[i])
    if i not in H or g is not None or q is None or cur is None: continue
    if base[i] != Z[i]: continue
    if r.get('q3why', '').startswith('prior_scene'): continue
    if max(q / cur, cur / q) <= 2: rows.append((i, math.sqrt(q * cur), 'rul342_gm'))
with open(CAND, 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
R = [r for r in csv.DictReader(open(CAND)) if int(r['id0']) in H and H[int(r['id0'])]['cat'] == 'S2']
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    [w.writerow([r['id0'], r['value'], 'rul342_gm (no own GvP1 geo; second per-question geo within 2x)']) for r in R]
print('cand_h24', len(rows), 'layer S2', len(R))
