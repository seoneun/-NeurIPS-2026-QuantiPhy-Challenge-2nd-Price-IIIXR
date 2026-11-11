"""rul235: template-stated speed fact for a motion row whose geo is void because the parse collapsed target and
prior object onto ONE tracked query (the rul199 condition: same phrase+where key, different raw strings, the tracks
json has a single query for that key carrying both roles 'target' and 'prior').  Geo then measures the prior object scaled
by the prior itself (it echoes the prior), so it can neither witness nor veto a fact; rul199 falls back to CaW, and the
rul034 / rul164 fact link is blocked by its own-geo witness gate (geo within 1.25x of the fact).

Rule (inputs only: parse q/target/prior_obj/priors, raw ground_truth_prior + meas_tracks, tracks json, detail answer):
  receiver: simulation row, q.kind speed, unit m/s, at-t / interval (t >= 0.1 s, not 'initial'), collapsed key as above;
  fact: template component = simulation bases sharing >= 2 lower-cased questions (rul033 / c99 link graph); fact key
        (speed, nobj2(q.target)) stated time-less (no t/t1/t2, not gravity) as a prior by OTHER bases of the component,
        all stated values within 2 %; own base states no speed fact of that key; no component member states an
        acceleration of that object (rul164 gates);
  fire if |answer / fact - 1| > 2 %  ->  answer = fact.  _segmented rows with the identical question get the same value.
Labels are joined at the end for evaluation only. No ids in the logic.
usage: tools/run_limited.sh --io python analysis/v3/ag/d10d2sim/rul235.py [SUB] [write]"""
import sys, re, json, os, collections
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 3000)
TAG = 'rul235'
ARGS = [a for a in sys.argv[1:] if a != 'write']
SUB = ARGS[0] if ARGS else 'sub_xt_Nzz38'
WRITE = 'write' in sys.argv[1:]
RAWN = 'geo_v4ntrj5sQ2srngsr_d_vw'
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
KEEP = {'ring', 'king', 'wing', 'string', 'spring', 'swing', 'building', 'ceiling', 'painting', 'railing', 'ping', 'thing'}
SYN = {'bike': 'bicycle', 'bikes': 'bicycle', 'motorbike': 'motorcycle', 'aeroplane': 'plane', 'airplane': 'plane'}


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def nobj2(x):
    w = [SYN.get(z, z) for z in nobj(x).split()]
    while len(w) > 1 and w[0].endswith('ing') and w[0] not in KEEP:
        w = w[1:]
    return " ".join(w)


def okey(o):
    return (str(o.get("phrase") or "").strip().lower(), tuple(o.get("where") or []))


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), p=p, q=p.get('q') or {},
                         priors=p.get('priors') or [], cat=det['cat'][i], ans=float(det.answer[i]), src=str(det.answer_source[i]),
                         g=det.g[i], cs=det.cs[i], trk=str(raw.meas_tracks[i])))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
d['base'] = d.vid.str.replace("_segmented", "", regex=False)
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
F = collections.defaultdict(lambda: collections.defaultdict(list))
ACC = collections.defaultdict(set)   # objects with ANY stated acceleration (timed or not) per base
for r in d.drop_duplicates(['sp', 'vid']).itertuples():
    for x in r.priors:
        if x.get('kind') == 'accel' and x.get('obj') and not x.get('gravity'):
            ACC[r.base].add(nobj2(x['obj']))
        if x.get('value_si') is None or x.get('gravity') or not x.get('obj'):
            continue
        if x.get('t') is not None or x.get('t1') is not None or x.get('t2') is not None:
            continue
        F[r.base][(x['kind'], x.get('dim') or '', nobj2(x['obj']))].append(float(abs(x['value_si'])))


def fact(base, obj):
    key = ('speed', '', obj)
    if key in F.get(base, {}):
        return None, 'own base states it'
    if any(obj in ACC[c] for c in COMP[base]):
        return None, 'accelerating in template'
    vals = [v for c in COMP[base] - {base} for v in F[c].get(key, [])]
    if not vals:
        return None, 'no fact'
    if max(vals) / min(vals) > 1.02:
        return None, 'facts disagree'
    return float(np.median(vals)), f"{len(vals)} stated"


TJ = {}


def collapsed(r):
    p = r.p
    t, po = p.get('target') or {}, p.get('prior_obj') or {}
    if not t or not po or okey(t) != okey(po) or not okey(t)[0]:
        return False
    rt, rp = str(t.get('raw') or '').strip().lower(), str(po.get('raw') or '').strip().lower()
    if not rt or not rp or rt == rp:
        return False
    fn = None
    for td in ([f"tracks_{r.sp}_{r.trk}"] if r.trk not in ('nan', '') else []) + [f"tracks_{r.sp}_v2sc"]:
        f = f"{R}/geo/out/{td}/{r.vid}.json"
        if os.path.exists(f):
            fn = f; break
    if fn is None:
        return False
    if fn not in TJ:
        TJ[fn] = json.load(open(fn))
    qk = [q for q in TJ[fn].get('queries', []) if (str(q.get('phrase', '')).strip().lower(), tuple(q.get('where') or [])) == okey(t)]
    return len(qk) == 1 and {'target', 'prior'} <= set(qk[0].get('roles') or [])


log, P = [], {'test': {}, 'val': {}}
for r in d.itertuples():
    q = r.q
    if q.get('kind') != 'speed' or q.get('unit') != 'm/s' or 'initial' in r.ql:
        continue
    tt = q.get('t') if q.get('t') is not None else q.get('t1')
    if tt is None and q.get('t2') is None:
        continue                      # whole-clip / time-less forms are not this rule's
    if tt is not None and tt < 0.1:
        continue
    if not collapsed(r):
        continue
    obj = nobj2(q.get('target') or '')
    fv, why = fact(r.base, obj)
    log.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, cat=r.cat, q=r.question[:70], src=r.src, ans=r.ans, g=r.g, cs=r.cs, obj=obj, fact=fv, why=why))
    if fv is None or abs(r.ans / fv - 1) <= 0.02:
        continue
    msg = (f"{TAG}: target/prior collapsed to one tracked query (geo echoes the prior, no witness); template states "
           f"speed of {obj} = {fv:g} m/s ({why}); was {r.ans:.4g} ({r.src})")
    P[r.sp][str(r.rid)] = {'value': fv, 'why': msg}
    if not r.vid.endswith('_segmented'):
        tw = d[(d.sp == r.sp) & (d.vid == r.vid + '_segmented') & (d.ql == r.ql)]
        for j in tw.itertuples():
            if str(j.rid) not in P[r.sp]:
                P[r.sp][str(j.rid)] = {'value': fv, 'why': msg + ' [_segmented copy]'}
print(pd.DataFrame(log).round(3).to_string())
from lab_rows import labelled
import analyze_caw as A
L = labelled(SUB)
for sp in ('test', 'val'):
    Ls = L[L.split == sp]
    Ls = Ls[~Ls.index.duplicated()]
    print(sp, 'patch rows', len(P[sp]))
    for k, e in P[sp].items():
        i = int(k); y = Ls.y.get(i, np.nan)
        ev = '' if y != y else '| LAB y %.4g MRA %.2f -> %.2f' % (y, A.row_mra(np.array([Ls.answer[i]]), np.array([y]))[0], A.row_mra(np.array([e['value']]), np.array([y]))[0])
        print('  ', k, e['value'], e['why'], ev)
    if WRITE:
        json.dump(P[sp], open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", 'w'), indent=1)
        print('wrote', f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json")
