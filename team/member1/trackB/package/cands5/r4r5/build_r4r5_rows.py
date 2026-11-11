import csv, os, re, sys, collections
ROWS, HELD, OUT = sys.argv[1:4]
I = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs') + '/'
def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
blk4 = {r['id0']: r['block'] for r in rd(ROWS + '/pred_t045_rows.csv')}
held = {r['id0'] for r in rd(HELD)}
mem5 = [('held', held)] + [(b, {r['id0'] for r in rd(ROWS + '/' + f + '_rows.csv')}) for b, f in
                           (('pred_t009', 'pred_t009'), ('pred_t010', 'pred_t010'), ('pred_t154', 'pred_t011'))]
rows = []
for r in rd(I + 'r4/collect_rows_latest.csv'):
    if r['sel'] != 'True': continue
    rows.append(dict(r, round='r4', block=blk4[r['id0']], ev_out=r['ev']))
for r in rd(I + 'r5/collect_r5_rows.csv'):
    b = next((b for b, s in mem5 if r['id0'] in s), None)
    if b is None: continue
    rows.append(dict(r, round='r5', block=b, ev_out=r['ev_cal']))
XW = re.compile(r'\bval\b|same question|cop(y|ies)|sibling|other render|renders|segmented|plain|family|\brows?\b \d')
PW = re.compile(r'c26c|pred_t007|CaW|\bcur\b')
OVERRIDE = {('1523', 'rul392'): 'X',
            ('3285', 'r5_left'): 'X'}
def cls(r):
    k = (r['id0'], r['key'])
    if k in OVERRIDE: return OVERRIDE[k]
    t = r['route1']; own = re.search(r'_(\d{4})', r['video']).group(1)
    if set(re.findall(r'(?<![\d.])(\d{4})(?![\d.])', t)) - {own}: return 'X'
    if XW.search(t): return 'X'
    if PW.search(t): return 'P'
    return 'S'
vids = collections.defaultdict(set)
for r in rows: vids[float(r['new'])].add(r['video'])
cols = ['id0','cat','video','round','block','key','auto','auto2','shared_new_videos','ev','old','new','route1','route2','script']
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(cols)
    for r in rows:
        a = cls(r); n = len(vids[float(r['new'])])
        a2 = 'X_copy' if a != 'X' and n > 1 else a
        w.writerow([r['id0'], r['cat'], r['video'], r['round'], r['block'], r['key'], a, a2, n, r['ev_out'], r['old'], r['new'],
                    r['route1'][:160], r['route2'][:100], r['script']])
