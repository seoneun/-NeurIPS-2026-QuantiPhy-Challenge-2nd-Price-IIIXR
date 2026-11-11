"""rul033: a template-stated size ('<dim> of X = s', a prior of another member of the same simulation template) answers a
size question '<dim> of X' (same dim, same normalised object) in a template member whose own base states no such size.

Inputs only (labels joined at the end for evaluation):
 template = connected component of simulation bases (video_id without _segmented) whose lower-cased question sets share
            >= 2 questions, pooled over test + val inputs (rul018 linkage, transitive closure); when the linked values for
            (X, dim) disagree by more than 2 %, the stricter >= 3-question component is used instead.
 fact     = a prior line of any member (parse_{split}_v2 priors, value_si) of kind size, with its dim and normalised object.
 receiver = a simulation size row (unit m) asking (X, dim), whose own base states no (X, dim) size.
Gates: every linked value agrees within 2 %; witness = own geo or CaW within 1.25x of s, decided on the plain (non-_segmented)
 rows of the same (base, X, dim) when there are any (all of them must pass), so a _segmented copy follows its plain video;
 skip when the answer is already within 2 % of s, or the answer source is scene_prior / xrow_prior / patch:rul018 /
 patch:rul020, or (C24SZ_OVERRIDE=0, default) any existing patch (patch:* / patch_copy) set the row.  Output s.
Run:  tools/run_limited.sh --io python analysis/v3/ag/rul033/rul033.py [write]
"""
import sys, re, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 2000)
SUB = os.environ.get('C24_SUB', 'sub_xt_Nzo')
OVERRIDE = os.environ.get('C24SZ_OVERRIDE', '0') == '1'      # 1: also replace rows an existing patch set
SCENE = os.environ.get('C24SZ_SCENE', '1') == '1'            # 1: scene gate (see below)
WRITE = 'write' in sys.argv[1:]
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
COVER = ('scene_prior', 'xrow_prior', 'patch:rul018', 'patch:rul020')


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def src0(s):
    return str(s).replace('mo_guard:', '')


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'],
                         cat=det.cat[i], ans=float(det.answer[i]), src=src0(det.answer_source[i]), g=det.g[i], cs=det.cs[i]))
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


LEVELS = [2, 3]
COMPS = {m: components(m) for m in LEVELS}

F = []                                    # size facts of every base (input priors of test and val rows)
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None or x.get('kind') != 'size':
            continue
        F.append((r.base, nobj(x['obj'] or ''), x.get('dim'), float(abs(x['value_si'])), x['raw'].strip(), r.sp))
F = pd.DataFrame(F, columns=['base', 'obj', 'dim', 'v', 'raw', 'fsp']).drop_duplicates(['base', 'raw', 'fsp'])
own = set(zip(F.base, F.obj, F.dim))
Fb = {b: g for b, g in F.groupby('base')}


def gv(x):
    return x == x and x is not None and x > 0


def within(x, v, f=1.25):
    return gv(x) and v / f <= x <= v * f


out = []
for r in d.itertuples():
    q = r.q
    if q.get('kind') != 'size' or q.get('unit') != 'm':
        continue
    o, dm = nobj(q.get('target') or ''), q.get('dim')
    if not o or (r.base, o, dm) in own:
        continue                                   # own base states it (the pipeline uses it)
    src = None
    for lev in LEVELS:
        cm = COMPS[lev]
        s_ = [(c, f.v, f.raw, f.fsp) for c in sorted(cm[r.base] - {r.base}) if c in Fb for f in Fb[c].itertuples()
              if f.obj == o and f.dim == dm]
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
    # scene gate: some base that states the fact shares with the receiver's base a question about ANOTHER object (the shared
    # rock / pier / shark / plane), not only generic questions about X itself ('what is the length of the car' is asked of
    # every car scene; a red sports car on a bridge and a white pickup next to a plane both ask it)
    head = o.split()[-1]
    scq = sorted(set(qq for c in set(z[0] for z in src) for qq in (qs[r.base] & qs[c]) if head not in nobj(qq).split()))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, base=r.base, cat=r.cat, dim=dm, q=r.question[:70], obj=o, v=v, agree=agree, lev=used,
                    nsrc=len(set(z[0] for z in src)), srcb=','.join(sorted(set(z[0][-4:] for z in src))),
                    fsp=','.join(sorted(set(z[3] for z in src))), raw0=src[0][2], ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    gw=within(r.g, v), cw=within(r.cs, v), have2=abs(r.ans - v) <= 0.02 * v, covd=r.src in COVER,
                    patched=r.src.startswith('patch:') or r.src == 'patch_copy', scene=len(scq) > 0, scq=' | '.join(scq)[:150]))
X = pd.DataFrame(out)
X['wit0'] = X.gw | X.cw
pl = X[~X.vid.str.endswith('_segmented')].set_index(['sp', 'base', 'obj', 'dim'])
pgate = pl.wit0.groupby(level=[0, 1, 2, 3]).min()          # every plain-recording row of (split, base, object, dim) must witness
X['wit'] = [bool(pgate.get((s, b, o, dm), w)) for s, b, o, dm, w in zip(X.sp, X.base, X.obj, X.dim, X.wit0)]
X['fire'] = X.agree & X.wit & ~X.have2 & ~X.covd & (OVERRIDE | ~X.patched) & (X.scene | (not SCENE))
# ---- labels: evaluation only
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, np.nan)) for s, i in zip(X.sp, X.rid)]
lab = X.y == X.y
X['f2'] = np.where(lab, (X.v - X.y).abs() <= 0.02 * X.y, np.nan)
X['m0'] = np.where(lab, A.row_mra(X.ans.values, X.y.fillna(1).values), np.nan)
X['m1'] = np.where(lab, A.row_mra(X.v.values, X.y.fillna(1).values), np.nan)
X['gr'] = (X.g / X.v).round(2); X['cr'] = (X.cs / X.v).round(2)
print(f'OVERRIDE={OVERRIDE} SCENE={SCENE} templates>1 (>=2q): {len(set(c for c in COMPS[2].values() if len(c) > 1))}; size receivers {len(X)}')
print(X.groupby('sp').agg(recv=('rid', 'size'), agree=('agree', 'sum'), have2=('have2', 'sum'), covd=('covd', 'sum'),
                          patched=('patched', 'sum'), wit=('wit', 'sum'), scene=('scene', 'sum'), fire=('fire', 'sum'), lab=('y', lambda s: s.notna().sum())))
cols = ['lev', 'sp', 'rid', 'vid', 'cat', 'obj', 'dim', 'v', 'y', 'ans', 'src', 'g', 'cs', 'agree', 'wit', 'scene', 'have2', 'covd', 'patched',
        'fire', 'm0', 'm1', 'srcb']
print('== labelled receivers (any gate state)')
print(X[lab][cols].sort_values(['vid']).round(3).to_string())
FI = X[X.fire]
print('== firing rows')
print(FI[cols + ['fsp', 'q']].sort_values(['sp', 'vid']).round(3).to_string())
N = X[X.agree & ~X.covd & ~X.have2]
print('== gate cross-table (agreeing, not covered, not already within 2%), by split: witness x scene')
print(pd.crosstab([N.sp, N.wit], N.scene))
lb = X[X.agree & (X.y == X.y)]
print('== labelled agreeing receivers: fact within 2% of label by (witness, scene)')
print(lb.groupby(['wit', 'scene']).agg(n=('rid', 'size'), f2=('f2', 'sum')))
print('== scene evidence of passing-witness rows')
print(N[N.wit][['sp', 'rid', 'vid', 'obj', 'dim', 'v', 'ans', 'scene', 'scq']].sort_values(['sp', 'vid']).to_string())
print('fired test by cat', FI[FI.sp == 'test'].groupby('cat').size().to_dict())
Lb = FI[FI.y == FI.y]
print('labelled fired n=%d  sum MRA before %.3f after %.3f' % (len(Lb), Lb.m0.sum(), Lb.m1.sum()))
X.to_pickle(f"{R}/cache/tmp/ag_c24sz_X{'_ovr' if OVERRIDE else ''}.pkl")
if WRITE:
    for sp in ('test', 'val'):
        Z = FI[FI.sp == sp]
        P = {str(int(r.rid)): {'value': float(r.v), 'why': f"rul033: template ({r.srcb}) states {r.raw0!r}; own base silent; "
                               f"geo {r.g:.3g} / CaW {r.cs:.3g} witness; was {r.ans:.4g} ({r.src})"} for r in Z.itertuples()}
        json.dump(P, open(f"{R}/cache/tmp/ag_c24sz_patch_{sp}.json", "w"), indent=1)
        print('wrote', sp, len(P))
