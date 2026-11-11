import csv, collections, sys
H1, AUD, VR, H32, H32S, OUT = sys.argv[1:7]
H = {int(r['id0']): r for r in csv.DictReader(open(H1, encoding='utf-8'))}; p17 = {i: H[i]['p17'] for i in H}
def eq(x, y): x = float(x); y = float(y); return abs(x - y) <= 1e-9 * max(1, abs(y))
A = {int(r['id0']): r for r in csv.DictReader(open(AUD))}
V = {int(r['id0']) for r in csv.DictReader(open(VR))}
RS = ('base_p17', '35_border', '24_rul408', '31_rul002', '32_rul310', '33_len_jointtrack', '34_rul311',
      '78_rul405.csv|r8_q3:audit_owngate_keep')
out = []; C = collections.Counter()
v = {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(H32, newline='', encoding='utf-8'))}
src = {int(r['id0']): r['src'] for r in csv.DictReader(open(H32S))}
for i in sorted(A):
    if H[i]['status'] == 'X_LAB_VAL' or A[i]['team_line'] == 'allowed' or not eq(v[i], p17[i]) or i in V: continue
    s = src.get(i, '')
    if not s.startswith(RS): continue
    z = H[i]['z1s2c']; c = A[i]['pred_t018']
    if not eq(z, p17[i]): val, how = z, 'own z1s2c'
    elif not eq(c, p17[i]): val, how = c, 'strict pred_t018'
    else: C['both==p17'] += 1; continue
    C[s.split('|')[0]] += 1
    out.append((i, val, 'rul427:%s (member2 audit %s %s; was %s)' % (how, A[i]['team_line'], A[i]['audit_verdict'], s[:60].replace(',', ';'))))
print(len(out), C.most_common())
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for r in out: w.writerow(r)
