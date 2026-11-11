import csv, sys
STEP, H1, OUT = sys.argv[1], sys.argv[2], sys.argv[-1]
def rd(p): return list(csv.DictReader(open(p)))
def sub(p): return {int(r['id']) - 1: r['parsed_value'] for r in rd(p)}
H = {int(r['id0']): r for r in rd(H1)}
if STEP == '03b':
    z = sub(sys.argv[3])
    rows = [[int(r['id0']), z[int(r['id0'])], 'aud_geo_rows->own z1s2c (GvP1 per-question geo)'] for r in rd(sys.argv[4])]
elif STEP == '33':
    rows = []
    for r in rd(sys.argv[3]):
        i = int(r['id0']); m = r['mechanism']
        if m in ('XG_JOINT_ONLY', 'XG_PERQ_ONLY', 'TRACK_JOINT_OBJ') and H[i]['status'] != 'X_LAB_VAL' and H[i]['reason'] == 'S':
            rows.append((i, H[i]['p17'], 'lenient_jointtrack:' + m))
elif STEP == '35':
    AG, CUR, GATE = sys.argv[3:6]
    cur = sub(CUR); rows = []
    for r in rd(GATE):
        i = int(r['id0']); rows.append((i, H[i]['p17'], 'border_gate_only'))
    for i, h in H.items():
        if h['status'] in ('X_TXT', 'GRAY') and cur[i] != h['p17']: rows.append((i, h['p17'], 'border_' + h['status']))
    for r in rd(AG):
        i = int(r['id0'])
        if r['mechanism'] == 'FOCALPOOL' and H[i]['status'] != 'X_LAB_VAL' and H[i]['reason'] == 'S' and cur[i] != H[i]['p17']:
            rows.append((i, H[i]['p17'], 'rul009'))
    seen = set(); rows = [x for x in rows if not (x[0] in seen or seen.add(x[0]))]
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
