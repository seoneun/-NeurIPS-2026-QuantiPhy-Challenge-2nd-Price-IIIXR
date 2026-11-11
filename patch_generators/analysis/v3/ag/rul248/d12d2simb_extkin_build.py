"""rul248: constant-speed prior kinematics for 'extension/rise at t' questions (label-free; labels only in the eval block).
Gates: question matches extension|extended|extend|rise|risen|raised|lifted and q.t (single, >=0.1, no t1/t2), q.kind in {other, disp},
unit m|cm|mm; exactly one prior, kind speed, value_si, no t/t1/t2, not gravity; prior object == question object (nobj2 head noun
or containment); answer_source not patch* and in {caw_nogeo, caw_c1, dis_caw_routed}; motion-start witness on the prior track
(tracks_{split}_v2sc): dominant-axis centre series leaves its start within the first 10% of tracked frames and moves monotonically;
|answer/(v t) - 1| > 0.02. Output v*t in the question unit; _segmented rows with the same question get the same value."""
import sys, os, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul248'
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz40'
UNIT = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
SRC_OK = {'caw_nogeo', 'caw_c1', 'dis_caw_routed'}
RX = re.compile(r'\b(extension|extended|extend|rise|risen|raised|lifted)\b', re.I)
STOP = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'its', 'this', 'that'}
SYN = {'bike': 'bicycle'}
KEEP = {'ring', 'string', 'swing', 'ceiling', 'building', 'king', 'wing', 'spring', 'railing', 'sling', 'awning', 'painting'}


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def nobj2(x):
    w = [SYN.get(z, z) for z in nobj(x).split()]
    while len(w) > 1 and w[0].endswith('ing') and w[0] not in KEEP:
        w = w[1:]
    return " ".join(w)


def head(x):
    s = nobj2(x).split()
    return s[-1] if s else ''


assert nobj2('charging post') == 'post' and head('the charging post') == 'post'


def prior_obj_id(T, phrase):
    for qn in ('queries', 'queries_d'):
        for e in T.get(qn, []) or []:
            if e.get('obj') is None:
                continue
            if 'prior' in (e.get('roles') or []) or e['key'].split('|')[0] == phrase:
                return e['obj']
    return None


def onset_witness(T, oid):
    S = T['objs'].get(str(oid))
    if S is None:
        return None
    c = np.array([s['st']['c'] if s is not None else [np.nan, np.nan] for s in S], float)
    ok = ~np.isnan(c[:, 0])
    if ok.sum() < 8 or not ok[0]:
        return None
    c = c[ok]
    rng = c.max(0) - c.min(0)
    ax = int(np.argmax(rng)); s = c[:, ax]; n = len(s)
    if rng[ax] < 20:
        return None
    dirn = np.sign(s[np.argmax(np.abs(s - s[0]))] - s[0])
    moved = np.where(np.abs(s - s[0]) > max(1.5, 0.02 * rng[ax]))[0]
    on = int(moved[0])                         # first frame clearly displaced
    d = np.diff(s) * dirn
    mono = bool((d >= -1.5).all())             # no reversal beyond 1.5 px
    return dict(onset_idx=on, n=n, frac=on / n, mono=mono, axis='xy'[ax], range_px=round(float(rng[ax]), 1))


def run(split):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j['id'])] = j
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    out = {}
    for rid, j in P.items():
        if rid not in det.index:
            continue
        q = j.get('q') or {}; pr = j.get('priors') or []; qs = j['question']
        if not RX.search(qs):
            continue
        print('  regex hit', split, rid, j['video_id'], qs[:70], q.get('kind'), q.get('t'), q.get('unit'))
        if q.get('kind') not in ('other', 'disp') or q.get('unit') not in UNIT:
            continue
        t = q.get('t')
        if t is None or q.get('t1') is not None or q.get('t2') is not None or float(t) < 0.1:
            continue
        if len(pr) != 1 or pr[0].get('kind') != 'speed' or pr[0].get('value_si') is None or pr[0].get('gravity') \
                or any(pr[0].get(k) is not None for k in ('t', 't1', 't2')):
            print('  gate2 fail', split, rid, qs[:60]); continue
        pobj = pr[0].get('obj') or (j.get('prior_obj') or {}).get('phrase') or ''
        tgt = q.get('target') or (j.get('target') or {}).get('phrase') or ''
        if not (head(pobj) and head(pobj) == head(tgt)) and nobj2(tgt) not in nobj2(pobj):
            print('  gate3 fail', split, rid, pobj, '|', tgt); continue
        src = str(det.loc[rid, 'answer_source'])
        if src.startswith('patch') or src not in SRC_OK:
            print('  gate4 fail', split, rid, src); continue
        tp = f"{R}/geo/out/tracks_{split}_v2sc/{j['video_id']}.json"
        if not os.path.exists(tp):
            print('  no track', split, rid); continue
        T = json.load(open(tp)); oid = prior_obj_id(T, (j.get('prior_obj') or {}).get('phrase'))
        w = onset_witness(T, oid) if oid is not None else None
        if w is None or w['frac'] > 0.10 or not w['mono']:
            print('  gate5 fail', split, rid, w); continue
        v = float(pr[0]['value_si']); val = v * float(t) * UNIT[q['unit']]
        a = float(det.loc[rid, 'answer'])
        print('  cand', split, rid, j['video_id'], qs[:70], '| v=%g t=%g ans=%.4g -> %.4g' % (v, t, a, val), w)
        if abs(a / val - 1) <= 0.02:
            continue
        out[str(rid)] = {'value': val, 'why': f'{TAG}: constant-speed prior v x t (v={v:g} m/s, t={t:g}s, onset frame {w["onset_idx"]}/{w["n"]}); was {a:.4g}'}
    # _segmented / plain copies with the identical question get the same value
    base = lambda v: re.sub(r'_segmented$', '', str(v))
    for k, v in list(out.items()):
        jk = P[int(k)]
        for rid, j in P.items():
            if str(rid) in out or rid not in det.index:
                continue
            if base(j['video_id']) == base(jk['video_id']) and j['question'].strip().lower() == jk['question'].strip().lower():
                if not str(det.loc[rid, 'answer_source']).startswith('patch'):
                    out[str(rid)] = {'value': v['value'], 'why': v['why'] + ' (copy)'}
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", 'w'), indent=1)
    print(split, TAG, len(out))
    for k, v in out.items():
        print('  ', k, P[int(k)]['video_id'], P[int(k)]['cat'], P[int(k)]['question'].strip()[:70], '| %.4g -> %.4g |' % (det.loc[int(k), 'answer'], v['value']), str(det.loc[int(k), 'answer_source']), v['why'])
    return out


res = {s: run(s) for s in ('test', 'val')}
sys.path.insert(0, R + '/cache/tmp')
from lab_rows import labelled
import analyze_caw as A
L = labelled(SUB)
nl = 0
for s in ('test', 'val'):
    for k, v in res[s].items():
        m = L[(L.split == s) & (L.index == int(k))]
        if len(m):
            nl += 1
            print('LABELLED', s, k, 'y', m.y.iloc[0], 'old m', m.m.iloc[0], 'new m', A.row_mra(np.array([v['value']]), np.array([m.y.iloc[0]])))
print('labelled rows changed:', nl)
