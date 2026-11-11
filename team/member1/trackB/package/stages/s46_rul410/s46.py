import csv, sys
vf_path, z_path, b89_path, out = sys.argv[1:5]
def sub(p): return {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(p, encoding='utf-8', newline=''))}
vf = sub(vf_path); z = sub(z_path)
b89 = {int(r['id0']): r['value'] for r in csv.DictReader(open(b89_path, encoding='utf-8'))}
A = (66, 518, 526, 654, 762, 1213, 1217, 1218, 1823, 2623)
B = (1232, 1268, 1279, 1292, 1305, 1329, 1346)
C = (2154, 2159, 2239, 2261, 2791, 2843, 2855, 2858, 2979)
D = (2834, 2846)
val = {i: vf[i] for i in A + B}
val.update({i: repr(float(z[i])) for i in C})
val.update({i: '%.3f' % float(b89[i]) for i in D})
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value']); w.writerows([i, val[i]] for i in sorted(val))
