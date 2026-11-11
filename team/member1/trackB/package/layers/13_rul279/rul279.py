import csv, sys
B6, B11, H1, OUT = sys.argv[1:5]
def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
def sub(p): return {int(r['id']) - 1: r['parsed_value'] for r in rd(p)}
H = {int(r['id0']): r for r in rd(H1)}
b6 = sub(B6); b11 = sub(B11)
rows = [[i, H[i]['z1s2c'], 'rul279'] for i in sorted(H) if b6[i] != b11[i] and H[i]['reason'] == 'S']
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule']); [w.writerow(r) for r in rows]
print(len(rows), 'rows ->', OUT)
