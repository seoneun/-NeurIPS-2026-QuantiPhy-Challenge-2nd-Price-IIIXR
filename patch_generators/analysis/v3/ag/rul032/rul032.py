"""rul032: template-stated time-less acceleration copied to 'acceleration of X (at t)' rows of other template members.

Inputs only (labels are joined at the end, for evaluation):
  template = connected component of simulation bases (video_id minus _segmented) whose lower-cased question sets share >= 2
             questions, pooled over test + val inputs (rul018 linkage); when the linked facts disagree, the stricter >= 3 component.
  fact     = a prior line of a member (parse_{split}_v2 priors): kind accel, not gravity, object present, t / t1 / t2 None and no time
             words on the left of '=' (before / after / at / t= / '<n> s'); value |value_si| (m/s^2).
  receiver = a simulation row asking 'acceleration of X at t' or 'acceleration of X' (no interval), unit m/s^2, same normalised X;
             its own base states no acceleration of X.
  gates    : every linked fact for X agrees within 2 %; own geo within 1.25x of a (C24ACC_WIT=geo, default) -- the scan's
             'geo OR CaW' (C24ACC_WIT=either) is kept as an option only; the answer is not already within 2 % of a; the row is not
             answered by scene_prior / xrow_prior / patch:rul018 / patch:rul020 and is not in any of the 85 sr74 patches.
  output   : a (constant acceleration: the acceleration at any t equals the template's time-less acceleration).
Why geo only: on the 21 labelled acceleration rows CaW is within 1.25x of the label on 24 % (geo 48 %), so CaW is a weak witness;
the one row the CaW-only path adds (test 1496, simulation_0311 car <- 0300 'acceleration of the car = 5') is the pair rul420's
motion gate already rejected: 0311's car is already moving at t = 0.25 s (189 px/s) and its pixel speed falls from 1.0 to 2.5 s
(slope -33 px/s^2), whereas 0300's car starts from rest (32 px/s at 0.25 s) and speeds up -- not the same scripted motion."""
import sys, re, json, os, collections
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 2000)
SUB = 'sub_xt_Nzo'
WIT = os.environ.get('C24ACC_WIT', 'geo')
WRITE = 'write' in sys.argv[1:]
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
COVER = ('scene_prior', 'xrow_prior', 'patch:rul018', 'patch:rul020')
TIMED = re.compile(r"\d\s*s\b|t\s*=|\bbefore\b|\bafter\b|\bat\b|\bfrom\b|\bbetween\b|\buntil\b|\bduring\b")

# the 85 patches of the current build (cache/tmp/sr74.sh --patch_json, Nzo line)
_sr = open(f"{R}/cache/tmp/sr74.sh").read()
TAGS = re.search(r'"Nzo\|\|--patch_json ([^"]+)"', _sr).group(1).split(',')
PATCHED = {sp: set() for sp in ('test', 'val')}
for tg in TAGS:
    for sp in ('test', 'val'):
        f = f"{R}/cache/tmp/ag_{tg}_patch_{sp}.json"
        if os.path.exists(f):
            PATCHED[sp] |= {int(k) for k in json.load(open(f))}
print('patch tags', len(TAGS), 'patched rows', {k: len(v) for k, v in PATCHED.items()})


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'],
                         depth=str(raw.depth_info[i]), cat=det.cat[i], ans=float(det.answer[i]), src=str(det.answer_source[i]),
                         g=det.g[i], cs=det.cs[i]))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower()
d['base'] = d.vid.str.replace("_segmented", "", regex=False)
d = d[d.base.str.startswith('simulation_')].reset_index(drop=True)
qs = d.groupby('base').ql.apply(set)
SB = list(qs.index)


def components(minq):
    lk = {b: [c for c in SB if c != b and len(qs[b] & qs[c]) >= minq] for b in SB}
    comp = {}
    for b0 in SB:
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


COMPS = {2: components(2), 3: components(3)}

# ---- facts: time-less, non-gravity accelerations
F = []
ownacc = collections.defaultdict(set)          # base -> objects with any acceleration statement (timed or not)
for r in d.itertuples():
    for x in r.priors:
        if x.get('kind') != 'accel' or x.get('value_si') is None:
            continue
        o = nobj(x.get('obj') or '')
        if o:
            ownacc[r.base].add(o)
        lhs = x['raw'].lower().split('=')[0]
        if x.get('gravity') or not o or any(x.get(k) is not None for k in ('t', 't1', 't2')) or TIMED.search(lhs):
            continue
        F.append((r.base, o, float(abs(x['value_si'])), x['raw'].strip(), r.sp))
F = pd.DataFrame(F, columns=['base', 'obj', 'v', 'raw', 'fsp']).drop_duplicates(['base', 'raw', 'fsp'])
Fb = {b: g for b, g in F.groupby('base')}
print('time-less acceleration facts', len(F), 'bases', F.base.nunique())


def gv(x):
    return x == x and x is not None and x > 0


def within(x, v, f=1.25):
    return gv(x) and v / f <= x <= v * f


out = []
for r in d.itertuples():
    q = r.q
    if q.get('kind') != 'accel' or q.get('unit') != 'm/s^2' or q.get('t1') is not None or q.get('t2') is not None:
        continue
    o = nobj(q.get('target') or '')
    if not o or o in ownacc[r.base]:
        continue
    src = None; used = None; agree = False
    for lev in (2, 3):
        s_ = [(c, f.v, f.raw, f.fsp) for c in sorted(COMPS[lev][r.base] - {r.base}) if c in Fb for f in Fb[c].itertuples() if f.obj == o]
        if not s_:
            break
        src, used = s_, lev
        vals = [z[1] for z in s_]
        agree = max(vals) / min(vals) <= 1.02
        if agree:
            break
    if not src:
        continue
    v = float(np.median([z[1] for z in src]))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, base=r.base, cat=r.cat, q=r.question[:62], obj=o, t=q.get('t'), v=v, lev=used,
                    agree=agree, nsrc=len(set(z[0] for z in src)), srcb=','.join(sorted(set(z[0][-4:] for z in src))),
                    fsp=','.join(sorted(set(z[3] for z in src))), raw0=src[0][2], ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    gw=within(r.g, v), cw=within(r.cs, v), have2=abs(r.ans - v) <= 0.02 * v, covd=r.src in COVER,
                    pat=r.rid in PATCHED[r.sp], depth=r.depth[:60]))
X = pd.DataFrame(out)
X['wit'] = X.gw if WIT == 'geo' else (X.gw | X.cw)
X['fire'] = X.agree & X.wit & ~X.have2 & ~X.covd & ~X.pat
# ---- labels: evaluation only
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, np.nan)) for s, i in zip(X.sp, X.rid)]
X['gr'] = (X.g / X.v).round(2); X['cr'] = (X.cs / X.v).round(2)
cols = ['sp', 'rid', 'vid', 'cat', 'q', 'obj', 'v', 'lev', 'nsrc', 'srcb', 'ans', 'src', 'gr', 'cr', 'agree', 'gw', 'cw', 'have2', 'covd', 'pat', 'fire', 'y']
print(f'== receivers (witness = {WIT})')
print(X[cols].sort_values(['sp', 'vid']).round(3).to_string(index=False))
FI = X[X.fire]
print('fired by split/cat', FI.groupby(['sp', 'cat']).size().to_dict())
lab = FI[FI.y == FI.y]
print('labelled fired rows:', len(lab))
if len(lab):
    print(lab[cols].round(3).to_string(index=False))

# ---- evidence 1: CaW vs geo as a witness on labelled acceleration rows (all-237 + val, all categories)
La = L[L.question.astype(str).str.lower().str.contains('acceleration')] if 'question' in L.columns else None
if La is None:
    Q = {}
    for sp in ('test', 'val'):
        raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
        Q[sp] = raw.question.astype(str)
    L['question'] = [Q[s].get(i, '') for s, i in zip(L.split, L.index)]
    La = L[L.question.str.lower().str.contains('acceleration')]
print('== labelled acceleration rows n=%d (val %d)' % (len(La), (La.split == 'val').sum()))
for nm, c in (('geo', 'g'), ('CaW', 'cs')):
    rr = La[c] / La.y
    print(f'  {nm}: within 1.25x {(rr.between(0.8, 1.25)).mean():.2f}  within 10% {(rr.between(0.9, 1.1)).mean():.2f}  '
          f'distinct values {La[c].round(3).nunique()}  most common {La[c].round(3).value_counts().head(3).to_dict()}')
# CaW answers on every simulation acceleration row (inputs): how often the default 2.0
Sa = d[d.q.map(lambda q: q.get('kind') == 'accel')]
print('  CaW on sim accel rows (inputs): n=%d, share == 2.0: %.2f' % (len(Sa), (Sa.cs.round(3) == 2.0).mean()),
      'top', Sa.cs.round(3).value_counts().head(4).to_dict())

# ---- evidence 2: convention 'acceleration at t = template time-less acceleration' inside labelled videos (labels evaluate)
# a video stating 'acceleration of X = a' that also asks speed of X at two times: dv/dt from its labels vs a
print('== within-video convention check (labelled speed@t pairs of an object whose time-less acceleration the video states)')
Ls = L.copy(); Ls['vid'] = Ls.video_id.astype(str)
for sp in ('test', 'val'):
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    Z = Ls[Ls.split == sp]
    for v, g in Z.groupby('vid'):
        recs = [(PP[i]['q'], PP[i]['priors'], y) for i, y in zip(g.index, g.y) if i in PP]
        facts = {nobj(x.get('obj') or ''): abs(x['value_si']) for _, pr, _ in recs for x in pr
                 if x.get('kind') == 'accel' and x.get('value_si') is not None and not x.get('gravity') and x.get('t') is None and x.get('t1') is None}
        sp_t = collections.defaultdict(list)
        for q, _, y in recs:
            if q.get('kind') == 'speed' and q.get('t') is not None:
                sp_t[nobj(q.get('target') or '')].append((float(q['t']), y))
        for o, a in facts.items():
            pts = sorted(set(sp_t.get(o, [])))
            if len(pts) >= 2:
                (t1, v1), (t2, v2) = pts[0], pts[-1]
                print(f'  {sp} {v} {o!r}: stated a={a:g}; labels v({t1:g})={v1:g}, v({t2:g})={v2:g} -> dv/dt={(v2 - v1) / (t2 - t1):.3f}')

# ---- evidence 3 (inputs): constant acceleration of the receiver's own track -- pixel speed linear in t
CSV_FPS = {}
for sp in ('test', 'val'):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", usecols=['video_id', 'fps'])
    CSV_FPS.update({(sp, v): float(f) for v, f in zip(raw.video_id, raw.fps)})


def track_c(sp, v, o):
    for dd in ('v2sc', 'v3nt'):
        p = f"{R}/geo/out/tracks_{sp}_{dd}/{v}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        qq = (j.get('queries_d') or []) + (j.get('queries') or [])
        k = next((q for q in qq if nobj(str(q.get('key')).split('|')[0]) == o and q.get('obj') is not None), None)
        if k is None:
            return None
        return {f: np.array(e['st']['c'], float) for f, e in zip(j['fidx'], j['objs'][str(k['obj'])]) if e and e.get('st')}
    return None


print('== receiver tracks: centre pixel speed vs t (+-0.25 s, CSV fps), linear fit')
for r in X[X.agree].itertuples():
    c = track_c(r.sp, r.vid, r.obj)
    f = CSV_FPS.get((r.sp, r.vid))
    if not c or not f:
        print('  ', r.rid, r.vid, 'no track'); continue
    h = max(1, int(round(0.25 * f)))
    ts, ss = [], []
    for k in sorted(c):
        if k - h in c and k + h in c:
            ts.append(k / f); ss.append(float(np.hypot(*(c[k + h] - c[k - h]))) / (2 * h / f))
    ts, ss = np.array(ts), np.array(ss)
    if len(ts) < 6:
        print('  ', r.rid, r.vid, 'short track'); continue
    A1 = np.vstack([ts, np.ones_like(ts)]).T
    co, res, *_ = np.linalg.lstsq(A1, ss, rcond=None)
    r2 = 1 - ((ss - A1 @ co) ** 2).sum() / ((ss - ss.mean()) ** 2).sum()
    print(f'   {r.sp} {r.rid} {r.vid} {r.obj!r}: n={len(ts)} slope {co[0]:.1f} px/s^2, intercept {co[1]:.1f} px/s, R2 {r2:.3f}, '
          f'v(t0)={ss[0]:.0f} v(end)={ss[-1]:.0f} px/s | fire={r.fire}')

X.to_pickle(f"{R}/cache/tmp/ag_c24acc_X_{WIT}.pkl")
if WRITE:
    for sp in ('test', 'val'):
        Z = FI[FI.sp == sp]
        P = {str(int(r.rid)): {'value': float(r.v), 'why': f"rul032: template ({r.srcb}) states {r.raw0!r} (time-less, constant acceleration); "
                               f"own base silent on {r.obj}; geo {r.g:.3g} ({r.gr}x) witness; was {r.ans:.4g} ({r.src})"} for r in Z.itertuples()}
        json.dump(P, open(f"{R}/cache/tmp/ag_c24acc_patch_{sp}.json", "w"), indent=1)
        print('wrote', sp, len(P), sorted(P))
