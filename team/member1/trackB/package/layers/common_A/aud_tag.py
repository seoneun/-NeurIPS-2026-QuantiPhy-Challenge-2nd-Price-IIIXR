import csv, os, sys, collections
H1, RW, OUT = sys.argv[1:4]
D = os.path.dirname(os.path.abspath(__file__))
h = {int(r['id0']): r for r in csv.DictReader(open(H1 + '/h1_rows.csv'))}
st = {int(r['id0']): r for r in csv.DictReader(open(H1 + '/rows_status.csv'))}
A = {r['tag']: r for r in csv.DictReader(open(D + '/s_audit.csv'))}
vd = {r['tag']: r['class'] for r in csv.DictReader(open(D + '/r7_verdict.csv'))}
rows = collections.defaultdict(set)
for i, r in h.items():
    if r['reason'] != 'S': continue
    for x in set(x.strip() for x in st[i]['tags'].split(',') if x.strip()):
        rows[x].add(i)
STRONG = {'other_video', 'other_row_answer', 'other_row_prior', 'other_row_question', 'other_row_geo', 'other_row_depth',
          'other_row_cs', 'per_video_pool', 'per_scene_pool', 'per_video_gate', 'other_phrase_tracks', 'other_patch_other_row'}
SUB = {'rul413': {1999, 2000, 2001}, 'wing': {1649, 1815}, 'rul059': {1795, 1800, 1806}, 'rul156': {2812, 2820},
       'rul016': {311, 315}, 'rul093': {1854}, 'rul080': {2011}, 'disc': {1942}, 'rul451': {2849, 2859, 2860},
       'rul424_OWN': {300, 301, 304, 305, 338, 339, 341, 371, 373, 530, 531, 533, 534, 602, 628, 684, 1023, 2681, 2682, 2683, 2684, 2685, 2944}}
NOT_X = {'lane', 'rul441', 'rul439', 'twin', 'rul348', 'rul345', 'rul335'}
BLOCK_X = {'d3labF', 'd3labF2E1', 'd3labF2E2', 'd3labF2E3', 's3dist31', 's3dist32'}


def is_x(t):
    if t in NOT_X:
        return False
    if t in A:
        kinds = set(k.split(':')[0].split('(')[0] for k in A[t]['cross_kinds'].split(';') if k)
        return A[t]['verdict'] == 'CROSS' and bool(kinds & STRONG)
    return t in vd and vd[t] not in ('S', 'GRAY')


X = collections.defaultdict(set)
for t, s in rows.items():
    if not is_x(t): continue
    rs = (s & SUB[t]) if t in SUB else s
    if t == 'rul424':
        rs = s - SUB['rul424_OWN']
    for i in rs: X[i].add(t)
for f in ['pred_t012', 'pred_t006', 'pred_t007', 'pred_t008', 'pred_t009', 'pred_t010', 'pred_t011', 'pred_t003']:
    p = 'analysis/v3/calib/pred_t012_rows.csv' if f == 'pred_t012' else RW + '/' + f + '_rows.csv'
    for r in csv.DictReader(open(p)):
        i = int(r['id0'])
        if r['block'] in BLOCK_X and h[i]['reason'] == 'S' and float(h[i]['h1']) == float(h[i]['p17']):
            X[i].add('block:' + r['block'])
for i in (211, 344, 3194, 3195):
    if h[i]['reason'] == 'S': X[i].add('claimC_skip')
before = {}
for r in csv.DictReader(open(RW + '/pred_t003_rows.csv')):
    before.setdefault(int(r['id0']), float(r['old']))
ownF = {}
for r in csv.DictReader(open('analysis/v3/ag/rul402/rows.csv')):
    ownF[int(r['id0'])] = float(r['F'])
for r in csv.DictReader(open('analysis/v3/ag/rul403/cand_all.csv')):
    try: ownF.setdefault(int(r['id0']), float(r['F']))
    except ValueError: pass
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule'])
    for i in sorted(X):
        tags = '|'.join(sorted(X[i])); blk_only = all(t.startswith('block:') for t in X[i])
        if blk_only and i in ownF and i in before and 'block:s3dist' not in tags:
            v, src = before[i] * ownF[i], 'ownF_b27'
        elif blk_only and i in before:
            v, src = before[i], 'before_block'
        else:
            v, src = float(h[i]['z1s2c']), 'z1s2c'
        w.writerow([i, v, 'aud_tag:' + tags + ':' + src])
