"""rul034: instantaneous / interval speed of X <- template-stated time-less 'speed of X = v' (uniform motion).

Simulation scenes of one template (connected component of bases = video_id minus _segmented whose lower-cased question sets
share >= 2 questions, pooled over test + val INPUTS, as rul018) reuse one animation.  A time-less prior 'speed of X = v m/s'
stated in another member then also answers 'speed of X at t' (t >= 0.1 s) and 'average speed of X from t1 to t2' in a member
whose own base is silent about X, provided X moves uniformly.  Gates (all code, no per-row choices):
 (1) every linked fact for X agrees within 2 % (else retry with the >= 3-question closure); answer = median
 (2) the row's own base states no speed and no acceleration of X        (3) no template member states an acceleration of X
 (4) own geo within [v/1.25, 1.25 v]
 (5) every at-t speed receiver of X in the same recording with a geo value is within 1.25x of v (blocks 0159: 1.5 s geo 2.67x)
 (6) skip rows already within 2 % of v, rows answered by scene_prior / xrow_prior, and rows any existing patch tag sets
     (C24TAT_PATCHES, default = the Nzo list in cache/tmp/sr74.sh; C24TAT_OVERRIDE = tags this rule may override, default none)
Output: v.  Labels are joined only at the end, for evaluation.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul034/rul034.py [write]"""
import sys, re, json, os, collections
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 2000)
TAG = 'rul034'
SUB = os.environ.get('C24TAT_SUB', 'sub_xt_Nzo')
RAWN = os.environ.get('C24TAT_RAW', 'geo_v4ntrj5sQ2srngsr_d_vw')
MINQ = int(os.environ.get('C24TAT_MINQ', '2'))
FALLBACK = os.environ.get('C24TAT_FALLBACK', '1') == '1'
WRITE = 'write' in sys.argv[1:] or os.environ.get('C24TAT_WRITE', '0') == '1'
OVERRIDE = set(x for x in os.environ.get('C24TAT_OVERRIDE', '').split(',') if x)
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
COVER = ('scene_prior', 'xrow_prior')


def patch_tags():
    if os.environ.get('C24TAT_PATCHES'):
        return [t for t in os.environ['C24TAT_PATCHES'].split(',') if t]
    m = re.search(r'"Nzo\|\|--patch_json ([^"\s]+)"', open(f"{R}/cache/tmp/sr74.sh").read())
    return m.group(1).split(',')


TAGS = patch_tags()
PATCHED = {}                                   # (split, row id) -> last tag that sets it
for t_ in TAGS:
    for sp in ('test', 'val'):
        p = f"{R}/cache/tmp/ag_{t_}_patch_{sp}.json"
        if os.path.exists(p):
            for k in json.load(open(p)):
                PATCHED[(sp, int(k))] = t_
print(f'existing patch tags {len(TAGS)}, rows they set: test {sum(1 for s, _ in PATCHED if s == "test")}, '
      f'val {sum(1 for s, _ in PATCHED if s == "val")}; may override: {sorted(OVERRIDE) or "none"}')


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'],
                         cat=det.cat[i], ans=float(det.answer[i]), src=det.answer_source[i], g=det.g[i], cs=det.cs[i]))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower()
d['base'] = d.vid.str.replace("_segmented", "", regex=False)
d = d[d.base.str.startswith('simulation_')].reset_index(drop=True)
qs = d.groupby('base').ql.apply(set); sb = list(qs.index)


def components(minq):
    lk = {b: [c for c in sb if c != b and len(qs[b] & qs[c]) >= minq] for b in sb}
    comp = {}
    for b0 in sb:
        if b0 in comp:
            continue
        st, seen = [b0], {b0}
        while st:
            x = st.pop()
            for y in lk[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        for x in seen:
            comp[x] = frozenset(seen)
    return comp


LEVELS = [MINQ] + ([MINQ + 1] if FALLBACK else [])
COMPS = {m: components(m) for m in LEVELS}
TIMED = r"\d\s*s\b|t\s*=|from|between|before|after|\bat \d"


def pkey(x):
    lhs = x['raw'].lower().split('=')[0]
    if x['kind'] == 'speed':
        return ('speed', None if not re.search(TIMED, lhs) else 'timed')
    if x['kind'] == 'accel':
        return ('grav', None) if (x.get('gravity') or not x['obj']) else ('accel', None)
    return (x['kind'], 'other')


F = []
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None:
            continue
        F.append((r.base, nobj(x['obj'] or ''), *pkey(x), float(abs(x['value_si'])), x['raw'].strip(), r.sp))
F = pd.DataFrame(F, columns=['base', 'obj', 'kind', 'tk', 'v', 'raw', 'fsp']).drop_duplicates(['base', 'raw', 'fsp'])
ownk = collections.defaultdict(set)
for f in F.itertuples():
    ownk[(f.base, f.obj)].add(f.kind)
compacc = set((cm[f.base], f.obj) for cm in COMPS.values() for f in F.itertuples() if f.kind == 'accel')
SPF = {b: g for b, g in F[(F.kind == 'speed') & F.tk.isna()].groupby('base')}


def gv(x):
    return x == x and x is not None and x > 0


out = []
for r in d.itertuples():
    q = r.q; o = nobj(q.get('target') or '')
    if not o or q.get('kind') not in ('speed', 'avgspeed') or q.get('unit') != 'm/s' or 'initial' in r.ql:
        continue
    t, t1, t2 = q.get('t'), q.get('t1'), q.get('t2')
    if t is not None:
        if t < 0.1:
            continue
        form = f'at {t:g}s'
    elif t1 is not None or t2 is not None:
        form = f'{t1}-{t2}s'
    else:
        continue                                   # time-less forms are rul018's
    if {'speed', 'accel'} & ownk[(r.base, o)]:
        continue                                   # gate 2: own base states a speed / acceleration of X
    src = None
    for lev in LEVELS:
        s_ = [(c, f.v, f.raw, f.fsp) for c in sorted(COMPS[lev][r.base] - {r.base}) if c in SPF for f in SPF[c].itertuples() if f.obj == o]
        if not s_:
            break
        src, used = s_, lev
        vals = [z[1] for z in src]
        agree = max(vals) / min(vals) <= 1.02
        if agree:
            break
    if not src:
        continue
    v = float(np.median(vals))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, base=r.base, cat=r.cat, form=form, q=r.question[:66], obj=o, v=v, lev=used,
                    agree=agree, tacc=(COMPS[used][r.base], o) in compacc, srcb=','.join(sorted(set(z[0][-4:] for z in src))),
                    fsp=','.join(sorted(set(z[3] for z in src))), raw0=src[0][2], ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    gw=gv(r.g) and v / 1.25 <= r.g <= 1.25 * v, have2=abs(r.ans - v) <= 0.02 * v,
                    covd=r.src in COVER, ptag=PATCHED.get((r.sp, r.rid), '')))
X = pd.DataFrame(out)
bad = set((s, v_, o) for s, v_, o, g_, w_ in zip(X.sp, X.vid, X.obj, X.g, X.gw) if gv(g_) and not w_)
X['sibbad'] = [(s, v_, o) in bad for s, v_, o in zip(X.sp, X.vid, X.obj)]
X['patched'] = (X.ptag != '') & ~X.ptag.isin(OVERRIDE)
X['fire'] = X.agree & ~X.tacc & X.gw & ~X.sibbad & ~X.have2 & ~X.covd & ~X.patched
# ---------------- evaluation (labels only here)
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, np.nan)) for s, i in zip(X.sp, X.rid)]
lab = X.y == X.y
X['m0'] = np.where(lab, A.row_mra(X.ans.values, X.y.fillna(1).values), np.nan)
X['m1'] = np.where(lab, A.row_mra(X.v.values, X.y.fillna(1).values), np.nan)
X['gr'] = (X.g / X.v).round(2); X['ar'] = (X.ans / X.v).round(2)
print(f'receivers {len(X)} (test {sum(X.sp == "test")}, val {sum(X.sp == "val")}); labelled receivers {int(lab.sum())}')
print('gate funnel (test/val):')
m = pd.Series(True, index=X.index)
for nm, c in (('linked facts agree', X.agree), ('no template accel', ~X.tacc), ('own geo within 1.25x', X.gw),
              ('siblings within 1.25x', ~X.sibbad), ('not already within 2%', ~X.have2), ('not scene/xrow prior', ~X.covd),
              ('not set by an existing patch', ~X.patched)):
    m = m & c
    print(f'   {nm:32s} test {int((m & (X.sp == "test")).sum()):3d}  val {int((m & (X.sp == "val")).sum()):3d}')
FI = X[X.fire].sort_values(['sp', 'vid', 'rid'])
cols = ['sp', 'rid', 'vid', 'cat', 'obj', 'form', 'v', 'ans', 'src', 'g', 'cs', 'gr', 'ar', 'lev', 'srcb', 'fsp', 'y', 'm0', 'm1']
print('== fired rows'); print(FI[cols].round(3).to_string())
print('== blocked only by an existing patch')
print(X[X.agree & ~X.tacc & X.gw & ~X.sibbad & ~X.have2 & ~X.covd & X.patched][cols + ['ptag']].round(3).to_string())
print('fired test by cat', FI[FI.sp == 'test'].groupby('cat').size().to_dict(), ' val', int((FI.sp == 'val').sum()))
Lb = FI[FI.y == FI.y]
print('labelled fired n=%d  MRA before %.3f after %.3f' % (len(Lb), Lb.m0.sum(), Lb.m1.sum()) if len(Lb) else 'labelled fired: none')
# related label evidence: labelled time-less average-speed rows about X whose template states X's time-less speed
ev = []
for r in d.itertuples():
    q = r.q; o = nobj(q.get('target') or '')
    if q.get('kind') not in ('speed', 'avgspeed') or not o:
        continue
    y = Lv.y.get(r.rid, np.nan) if r.sp == 'val' else Lt.y.get(r.rid, np.nan)
    if y != y:
        continue
    s_ = [f.v for c in sorted(COMPS[MINQ][r.base] - {r.base}) if c in SPF for f in SPF[c].itertuples() if f.obj == o]
    if s_ and max(s_) / min(s_) <= 1.02:
        ev.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, obj=o, kind=q['kind'], t=q.get('t'), v=float(np.median(s_)), y=y))
E = pd.DataFrame(ev).drop_duplicates(['vid', 'obj', 'kind', 't'])
print('== related labelled evidence (template speed agrees, any speed form):')
print(E.assign(err=(E.v / E.y - 1).round(3)).to_string() if len(E) else ' none')
X.to_pickle(f"{R}/cache/tmp/ag_{TAG}_X.pkl")
if WRITE:
    for sp in ('test', 'val'):
        Z = FI[FI.sp == sp]
        P = {str(int(r.rid)): {'value': float(r.v), 'why': f"{TAG}: template ({r.srcb}) states {r.raw0!r} (time-less, uniform motion); "
                               f"asked {r.form}; own base silent; own geo {r.g:.3g} within 1.25x; was {r.ans:.4g} ({r.src})"}
             for r in Z.itertuples()}
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print('wrote', f"cache/tmp/ag_{TAG}_patch_{sp}.json", len(P))
