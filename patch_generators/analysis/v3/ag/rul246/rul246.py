"""rul246: simulation 2D speed/avgspeed rows (2d_ratio) whose own base is silent about X's speed take the
template-stated time-less speed fact v of X (rul034 linkage: sim bases sharing >= 2 lower-cased questions, transitive closure;
nobj2 object normalisation, rul164), but ONLY when a same-video ruler witness confirms it: size rows about X in a 2d_ratio
mode with g > 0 whose answer equals a stated time-less size fact s of X (same dim) within 2 %; k = geomean(g_s/s), |ln k| >= ln 1.15,
all on one side. Fire if g/(k v) in [1/1.10, 1.10], the ruler moves g toward v, and the answer is > 2 % from v. Output v (row unit).
No ids, no labels in the logic (labels joined at the end for evaluation only).
usage: tools/run_limited.sh --io python analysis/v3/ag/rul246/rul246.py [SUB] [write]"""
import sys, re, json, os, collections
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 3000)
TAG = 'rul246'
args = [a for a in sys.argv[1:] if a != 'write']
SUB = args[0] if args else 'sub_xt_Nzz39'
RAWN = 'geo_v4ntrj5sQ2srngsr_d_vw'
WRITE = 'write' in sys.argv[1:]
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
KEEP = {'ring', 'king', 'wing', 'string', 'spring', 'swing', 'building', 'ceiling', 'painting', 'railing', 'ping', 'thing'}
SYN = {'bike': 'bicycle', 'bikes': 'bicycle', 'motorbike': 'motorcycle', 'aeroplane': 'plane', 'airplane': 'plane'}
ULEN = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'km': 1000.0}
USPD = {'m/s': 1.0, 'cm/s': 0.01, 'mm/s': 0.001, 'km/h': 1 / 3.6}
AG, KMIN, FIREW = 1.02, 1.15, 1.10


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def nobj2(x):
    w = [SYN.get(z, z) for z in nobj(x).split()]
    while len(w) > 1 and w[0].endswith('ing') and w[0] not in KEEP:
        w = w[1:]
    return " ".join(w)


def gv(x):
    return x is not None and x == x and x > 0


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'],
                         vt=str(p.get('video_type')), cat=det['cat'][i], ans=float(det.answer[i]),
                         src=str(det.answer_source[i]).replace('mo_guard:', ''), g=float(det.g[i]), cs=det.cs[i],
                         mode=str(raw.meas_mode[i])))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower(); d['base'] = d.vid.str.replace("_segmented", "", regex=False)
d = d[d.base.str.startswith('simulation_')].reset_index(drop=True)
qs = d.groupby('base').ql.apply(set); sb = list(qs.index)
lk = {b: [c for c in sb if c != b and len(qs[b] & qs[c]) >= 2] for b in sb}
COMP = {}
for b0 in sb:
    if b0 in COMP:
        continue
    st, seen = [b0], {b0}
    while st:
        x = st.pop()
        for y in lk[x]:
            if y not in seen:
                seen.add(y); st.append(y)
    for x in seen:
        COMP[x] = frozenset(seen)
TIMED = r"\d\s*s\b|t\s*=|from|between|before|after|\bat \d"

# fact table: (base, obj2, key, v); key = 'speed' | 'speed_t' | 'accel' | ('size', dim); non-gravity only
F = []
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None:
            continue
        lhs = x['raw'].lower().split('=')[0]
        timed = bool(re.search(TIMED, lhs)) or any(x.get(k) is not None for k in ('t', 't1', 't2'))
        o = nobj2(x.get('obj') or '')
        if x['kind'] == 'accel':
            if x.get('gravity') or not o:
                continue
            key = 'accel'          # any acceleration statement of X is a veto
        elif x['kind'] == 'speed':
            key = 'speed' if not timed else 'speed_t'
        elif x['kind'] == 'size':
            if timed:
                continue
            key = ('size', x.get('dim'))
        else:
            continue
        F.append((r.base, o, key, float(abs(x['value_si'])), x['raw'].strip()))
F = pd.DataFrame(F, columns=['base', 'obj', 'key', 'v', 'raw'])
F = F[~F.duplicated(['base', 'raw'])]
FB = collections.defaultdict(list)
for f in F.itertuples():
    FB[(f.base, f.obj, f.key)].append((f.v, f.raw))
OWNK = collections.defaultdict(set)
for f in F.itertuples():
    OWNK[(f.base, f.obj)].add(f.key if isinstance(f.key, str) else 'size')


def agree(vals):
    return len(vals) > 0 and max(vals) / min(vals) <= AG


def resolve(base, o, key, own_first):
    """returns (v, how, raws) or (None, reason)"""
    if own_first and FB.get((base, o, key)):
        vals = [z[0] for z in FB[(base, o, key)]]
        return (float(np.median(vals)), 'own', [z[1] for z in FB[(base, o, key)]]) if agree(vals) else (None, 'own_disagree')
    mem = sorted(COMP[base] - {base})
    hits = [(c, len(qs[base] & qs[c]), z) for c in mem for z in FB.get((c, o, key), [])]
    if not hits:
        return None, 'nofact'
    vals = [h[2][0] for h in hits]
    if agree(vals) and (key != 'speed' or not any(FB.get((c, o, 'accel')) for c in mem)):
        return float(np.median(vals)), f'comp{len(set(h[0] for h in hits))}', sorted(set(h[0][-4:] + ':' + h[2][1] for h in hits))
    top = max(h[1] for h in hits)
    th = [h for h in hits if h[1] == top]
    vals = [h[2][0] for h in th]
    if not agree(vals):
        return None, 'top_disagree'
    if key == 'speed' and any(FB.get((c, o, 'accel')) for c in mem if len(qs[base] & qs[c]) >= top):
        return None, 'top_accel'
    return float(np.median(vals)), f'top{top}', sorted(set(h[0][-4:] + ':' + h[2][1] for h in th))


d['o2'] = [nobj2(q.get('target') or '') for q in d.q]
d['is2d'] = d.vt.str[1] == '2'
out = []
for r in d.itertuples():
    q = r.q
    if not r.is2d or q.get('kind') not in ('speed', 'avgspeed') or q.get('unit') not in USPD or 'initial' in r.ql:
        continue
    t, t1, t2 = q.get('t'), q.get('t1'), q.get('t2')
    if t is not None:
        if t < 0.1 or t1 is not None or t2 is not None:
            continue
        form = f'at {t:g}s'
    elif t1 is None and t2 is None:
        form = 'whole'
    else:
        continue
    if not r.mode.endswith('2d_ratio') or not gv(r.g) or r.src.startswith('patch'):
        continue
    o = r.o2
    rec = dict(sp=r.sp, rid=r.rid, vid=r.vid, cat=r.cat, form=form, o=o, ans=r.ans, src=r.src, g=r.g, cs=r.cs, q=r.question[:60])
    if not o or {'speed', 'speed_t', 'accel'} & OWNK[(r.base, o)]:
        rec['res'] = 'own_states'; out.append(rec); continue
    sv = resolve(r.base, o, 'speed', own_first=False)
    if sv[0] is None:
        rec['res'] = 'v:' + sv[1]; out.append(rec); continue
    v = sv[0] / USPD[q['unit']]; rec.update(v=v, vhow=sv[1], vraw=sv[2])
    # ruler witnesses: same video, size rows about X, 2d_ratio mode, g>0, answer == stated size fact (same dim) within 2 %
    W = d[(d.sp == r.sp) & (d.vid == r.vid) & (d.o2 == o)]
    ks, wit = [], []
    for w in W.itertuples():
        qw = w.q
        if qw.get('kind') != 'size' or qw.get('unit') not in ULEN or not w.mode.endswith('2d_ratio') or not gv(w.g):
            continue
        sf = resolve(w.base, o, ('size', qw.get('dim')), own_first=True)
        if sf[0] is None:
            continue
        s = sf[0] / ULEN[qw['unit']]
        if abs(w.ans / s - 1) > 0.02:
            continue
        ks.append(w.g / s); wit.append(f"{w.rid}:{qw.get('dim')} g{w.g:.4g}/s{s:.4g}({sf[1]})")
    rec['wit'] = wit
    if not ks:
        rec['res'] = 'nowit'; out.append(rec); continue
    lks = np.log(ks)
    k = float(np.exp(lks.mean())); rec['k'] = k
    if not (np.all(lks >= np.log(KMIN)) or np.all(lks <= -np.log(KMIN))):
        rec['res'] = 'k_small_or_mixed'; out.append(rec); continue
    rr = r.g / (k * v); rec['gkv'] = rr
    if not (1 / FIREW <= rr <= FIREW):
        rec['res'] = 'gkv_out'; out.append(rec); continue
    if not abs(np.log(rr)) < abs(np.log(r.g / v)):
        rec['res'] = 'ruler_away'; out.append(rec); continue
    if abs(r.ans / v - 1) <= 0.02:
        rec['res'] = 'have'; out.append(rec); continue
    rec['res'] = 'FIRE'; out.append(rec)
X = pd.DataFrame(out)
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, np.nan)) for s, i in zip(X.sp, X.rid)]
for c in ('v', 'vhow', 'k', 'gkv', 'wit'):
    if c not in X:
        X[c] = np.nan
print('receivers', len(X)); print(X.groupby(['sp', 'res']).size().to_string())
cols = ['sp', 'rid', 'vid', 'cat', 'form', 'o', 'ans', 'src', 'g', 'v', 'vhow', 'k', 'gkv', 'wit', 'res', 'y']
print('== receivers with resolved v:'); print(X[X.v.notna()][cols].round(3).to_string())
print('== labelled receivers (any state):'); print(X[X.y == X.y][cols].round(3).to_string())
for sp in ('test', 'val'):
    P = {}
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if sp == 'test' else 'valdetail'}.csv", index_col=0)
    for r in X[(X.res == 'FIRE') & (X.sp == sp)].itertuples():
        why = (f"{TAG}: template fact {r.vraw} ({r.vhow}) v={r.v:.4g}; own base silent on '{r.o}'; ruler k={r.k:.3f} from "
               f"{r.wit}; g/(k v)={r.gkv:.3f}; was {r.ans:.4g} ({r.src})")
        P[str(int(r.rid))] = {'value': float(r.v), 'why': why}
    print(sp, 'rows', len(P))
    for k_, e in P.items():
        i = int(k_); y = (Lv.y.get(i, np.nan) if sp == 'val' else Lt.y.get(i, np.nan))
        ev = '' if y != y else '| MRA %.2f -> %.2f' % (A.row_mra(np.array([det.answer[i]]), np.array([y]))[0], A.row_mra(np.array([e['value']]), np.array([y]))[0])
        print('   ', k_, det.video_id[i], det['cat'][i], str(det.question[i])[:60], '| %.4g -> %.4g' % (det.answer[i], e['value']), '| y', y, ev, '|', e['why'])
    if WRITE:
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", 'w'), indent=1)
        print('wrote', f"cache/tmp/ag_{TAG}_patch_{sp}.json")
