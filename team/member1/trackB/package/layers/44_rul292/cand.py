import csv, json, os, re, sys
import numpy as np

parse, h1rows, srcf, measf, trk_own, trk_root, outf = sys.argv[1:8]
P = {r['id']: r for r in map(json.loads, open(parse))}
HR = {int(r['id0']): r for r in csv.DictReader(open(h1rows))}
SRC = {int(r['id0']): r for r in csv.DictReader(open(srcf))}
V = {i: float(r['value']) for i, r in SRC.items()}


def trk(v):
    for t in (trk_own, trk_root):
        p = f'{t}/{v}.json'
        if os.path.exists(p): return json.load(open(p))


def arr(d, ob):
    fr = d['objs'][str(ob)]
    c = np.array([e['st']['c'] if e and e.get('st') else [np.nan] * 2 for e in fr], float)
    box = np.array([e['st']['box'] if e and e.get('st') else [np.nan] * 4 for e in fr], float)
    rr = np.array([e['st']['rr'][:2] if e and e.get('st') else [np.nan] * 2 for e in fr], float)
    return c, box, rr


def tobj(d, phrase):
    for q in d['queries']:
        if q['obj'] is not None and str(q['obj']) in d['objs'] and phrase and (q['phrase'] == phrase or q['phrase'] in phrase or phrase in q['phrase']): return q['obj']


OUT = {}


def put(i, v, rule): OUT[i] = (float(v), rule)


for i, p in P.items():
    ql = p['question'].lower(); m = re.search(r'\b(length|width|height)\s*\(\s*(vertical|horizontal)\s*\)', ql)
    if not m or p['q']['kind'] != 'size' or i not in V: continue
    d = trk(p['video_id']); ob = tobj(d, p['q']['target']) if d else None
    if ob is None: continue
    _, box, _ = arr(d, ob); rho = np.nanmedian(box[:, 3] - box[:, 1]) / np.nanmedian(box[:, 2] - box[:, 0])
    word, axis = m.groups(); base_long = (word == 'length')
    ax_long = (axis == 'horizontal') == (rho < 1)
    if base_long == ax_long: continue
    k = min(rho, 1 / rho)
    put(i, V[i] * k if base_long else V[i] / k, 'R1_axis')
for i, p in P.items():
    pr = [x for x in p['priors'] if x['kind'] == 'size' and x.get('obj') and 'card' in x['obj'].lower()]
    if not pr or i not in V: continue
    x = pr[0]; mm = x['value_si'] * 1000; iso_long = abs(np.log(mm / 85.60)) < abs(np.log(mm / 53.98)); word_long = x['dim'] in ('length', 'long')
    tgt = (p['q'].get('target') or '').lower()
    if iso_long == word_long: continue
    if 'card' in tgt and p['q']['kind'] == 'size': continue
    d = trk(p['video_id']); ob = tobj(d, 'credit card') if d else None
    if ob is None: continue
    _, _, rr = arr(d, ob); k = np.nanmedian(np.nanmin(rr, 1)) / np.nanmedian(np.nanmax(rr, 1))
    put(i, V[i] * k if iso_long else V[i] / k, 'R2_isocard')
HOP = r'\b(rabbit|bunny|hare|kangaroo|frog)\b'
for i, p in P.items():
    if p['video_source'] != 'simulation' or p['q']['kind'] != 'speed' or p['q'].get('t') is None or i not in V: continue
    if not re.search(HOP, p['question'].lower()): continue
    d = trk(p['video_id']); ob = tobj(d, 'rabbit') if d else None
    if ob is None: continue
    c, _, _ = arr(d, ob); dd = np.diff(c, axis=0); st = np.linalg.norm(dd, axis=1); k = round(p['q']['t'] * d['fps'])
    r = np.nanmean(st) / st[k - 1]
    jump_base = abs(V[i] - float(HR[i]['h1'])) / V[i] < 0.01 and SRC[i]['src'].split('|')[-1].find('d2rabbit') >= 0
    if not jump_base: continue
    put(i, V[i] * r, 'R3_hopavg')
for i, p in P.items():
    ql = p['question'].lower()
    if not re.search(r'(velocity|speed) .*when (it )?hits?', ql) or i not in V: continue
    keys = [(x['t'], x['d']) for x in p['depth'] if x['t'] is not None]
    pr = [x for x in p['priors'] if x['kind'] == 'size' and x['dim'] == 'diameter']
    d = trk(p['video_id']); ob = tobj(d, p['q']['target']) if d else None
    if len(keys) < 2 or not pr or ob is None: continue
    c, box, _ = arr(d, ob); rad = (box[:, 2] - box[:, 0] + box[:, 3] - box[:, 1]) / 4; fps = d['fps']; R_ = pr[0]['value_si'] / 2
    f = float(np.mean([dk * rad[round(t * fps)] / R_ for t, dk in keys])); W, H = d['W'], d['H']
    (t0, d0), (t1, d1) = keys[:2]; vy = np.diff(c[:, 1]); vx = np.diff(c[:, 0])
    mov = np.where(np.hypot(vx, vy) > 10)[0][0]
    rev = [j for j in range(mov + 1, len(vy)) if (np.sign(vy[j]) != np.sign(vy[j - 1]) and abs(vy[j - 1]) > 0.3) or (np.sign(vx[j]) != np.sign(vx[j - 1]) and abs(vx[j - 1]) > 0.3)]
    i0 = rev[0]

    def pos(j, mode):
        D = d0 + (d1 - d0) * (j / fps - t0) / (t1 - t0) if mode == 'lin' else f * R_ / rad[j]
        return np.array([1.3 * (c[j, 0] - W / 2) * D / f, 1.3 * (c[j, 1] - H / 2) * D / f, D])
    v5 = {m: np.linalg.norm(pos(i0 + 5, m) - pos(i0, m)) * fps / 5 for m in ('lin', 'sz')}
    u = 100.0 if 'cm' in p['q']['unit'] else 1.0
    put(i, float(np.sqrt(v5['lin'] * v5['sz'])) * u, 'R4_hitFD5')
GM = {}
for r in csv.DictReader(open(measf)):
    if r['ok'] == 'True' and r['value']: GM[int(r['id'])] = float(r['value'])
for i, p in P.items():
    if p['q']['kind'] != 'speed' or p['q'].get('t') != 0 or i not in V: continue
    if not any(x.get('gravity') for x in p['priors']): continue
    d = trk(p['video_id']); ob = tobj(d, p['q']['target']) if d else None
    if ob is None: continue
    c, _, _ = arr(d, ob); st = np.linalg.norm(np.diff(c, axis=0), axis=1)
    loc = st[0:2]
    if len(loc) == 0 or np.isnan(loc).any() or loc.max() >= 0.5 or np.nanmax(st) < 3: continue
    if i not in GM: continue
    if abs(GM[i] / V[i] - 1) < 0.05: continue
    put(i, GM[i], 'R5_rest')
with open(outf, 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['id0', 'value', 'rule'])
    for i, (v, r) in sorted(OUT.items()): w.writerow([i, repr(v), r])
