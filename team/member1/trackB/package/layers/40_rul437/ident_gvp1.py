import json, os, csv, sys
import numpy as np
R = '.'
PARSE, AG, Z1, OUT = sys.argv[1:5]
def key(e):
    if not e or not e.get('phrase'): return None
    rel = e.get('rel'); return e['phrase'] + '|' + ','.join(e.get('where') or []) + ('|' + rel[0] + ':' + rel[1] if rel else '')
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0
TC = {}
def tr(sp, vid):
    p = f'{R}/geo/out/tracks_{sp}/{vid}.json'
    if p not in TC: TC[p] = json.load(open(p)) if os.path.exists(p) else None
    return TC[p]
ident = {}
for line in open(PARSE):
    r = json.loads(line); tk = key(r.get('target')); pk = key(r.get('prior_obj'))
    if not tk or not pk or tk == pk: continue
    if (r.get('target') or {}).get('phrase') == (r.get('prior_obj') or {}).get('phrase'): continue
    d = tr('test', r['video_id'])
    if not d: continue
    Q = {q.get('key'): q.get('frames') or [] for q in d.get('queries') or []}
    if tk not in Q or pk not in Q: continue
    v = [iou(a['st']['box'], b['st']['box']) for a, b in zip(Q[tk], Q[pk]) if a and b and a.get('st') and b.get('st')]
    if len(v) >= 5 and np.mean(v) > 0.9:
        ident[int(r['id'])] = dict(vid=r['video_id'], cat=r['cat'], tk=tk, pk=pk, iou=round(float(np.mean(v)), 3), n=len(v))
Z = {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(Z1, newline='', encoding='utf-8'))}
rows = []
for r in csv.DictReader(open(AG)):
    i = int(r['id0'])
    if r['changed'] != 'True' or not r['mechanism'].startswith('XG_JOINT') or r['cat'] != 'S2' or i not in ident: continue
    rule = ('ident_mchain->gvp1 (aud_geo XG_JOINT m-chain target track == prior track IoU %s; own z1s2c/GvP1)' % str(ident[i]['iou']).lstrip('0')
            if not rows else 'ident_mchain->gvp1 (same)')
    rows.append([i, Z[i], rule])
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
print(len(ident), 'identity rows;', len(rows), 'layer rows ->', OUT)
