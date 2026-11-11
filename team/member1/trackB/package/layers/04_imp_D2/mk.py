import sys, re, math, csv

id1, detail, h1rows, out = sys.argv[1:5]
TQ = re.compile(r'at (?:time )?([\d.]+)\s*s\b', re.I)
AC = re.compile(r'^\s*(?:acceleration|gravity)[^=]*=\s*([\d.]+)\s*m/s', re.I)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def ffbdq(r):
    if not re.search(r'\b(speed|velocity)\b', r['question'], re.I):
        return None
    m = TQ.search(r['question']); a = AC.search(r['prior'])
    if not (m and a) or ';' in r['prior'] or ',' in r['prior']:
        return None
    t = float(m.group(1)); a = float(a.group(1)); fps = r['fps']
    if t <= 0 or a <= 0 or not fps:
        return None
    if abs(r['cur'] / (a * t) - 1) > 0.01:
        return None
    k = round(t * fps)
    if k < 1:
        return None
    return a * (k - 0.5) / fps, 'ffbdq_d2'


def cawpull(r):
    g, cs, c = r['g'], r['cs'], r['cur']
    if not (g and cs and g > 0 and cs > 0 and c > 0):
        return None
    if abs(math.log(c / g)) > 0.02:
        return None
    u = abs(math.log(g / cs))
    if not (0.1 < u <= 0.5):
        return None
    return math.exp(0.9 * math.log(c) + 0.1 * math.log(cs)), 'cawpull90'


det = {}
with open(detail, newline='', encoding='utf-8') as f:
    rd = csv.reader(f); hdr = next(rd)
    for r in rd:
        det[int(r[0])] = dict(zip(hdr, r))
h1 = {int(x['id0']): x for x in csv.DictReader(open(h1rows, newline='', encoding='utf-8'))}
res = {}
for q in csv.DictReader(open(id1, newline='', encoding='utf-8')):
    i = int(q['id']) - 1
    x = h1.get(i)
    if q['inference_type'][0] + q['video_type'][1] != 'D2' or x is None or x['reason'] == 'S':
        continue
    r = dict(question=q['question'], prior=q['ground_truth_prior'], fps=float(q['fps']),
             g=num(det[i]['g']), cs=num(det[i]['cs']), cur=float(x['h1']))
    v = ffbdq(r)
    if v is None:
        v = cawpull(r)
    if v is None:
        continue
    v, rule = v
    if v > 0 and abs(math.log(v / r['cur'])) > 1e-12:
        res[i] = (v, rule)
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id0', 'value', 'rule'])
    for i, (v, rule) in sorted(res.items()):
        w.writerow([i, repr(v), 'imp:' + rule])
