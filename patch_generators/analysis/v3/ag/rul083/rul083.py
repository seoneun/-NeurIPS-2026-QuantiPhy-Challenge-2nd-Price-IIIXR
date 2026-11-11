"""rul083: prior-text twin propagation of template-stated SIZES (coverage gap of scene_prior / rul033 / rul031 /
rul054 / rul067 / rul075).

Template mechanisms answer a simulation size row with a size that another video of the scene template states as its
prior.  Each has a per-row witness gate (own geo or CaW near the fact), so two cameras of the SAME scene variant -
identical prior text, identical question - can land on different sides of the gate by a CaW coin flip (dragon
0302 CaW 0.87 -> fires, 0309 CaW 1.53 -> does not, while both geo read 2.36 / 2.32).

Rule (inputs only: detail answer/answer_source, raw question + ground_truth_prior, parse q.kind/q.unit/priors.gravity):
  receiver r: simulation, kind size, unit m, no gravity prior; a plain video, or a segmented row whose base has no plain
    row with the same question; answer_source not in TPL and not a c48 patch;
  donors t: another simulation base, identical whitespace/case-normalised prior text, identical normalised question,
    answer_source in TPL; all donor answers agree within 2 % -> F (median);
  gate: F equals (2 %) a size stated in some simulation prior '<dim> of the <obj> = v m|cm|mm' whose obj head noun
    (last word of obj before any ' of ') occurs as a whole word in r's question; |r/F - 1| > 2 %;
  =>  r = F; the segmented copy of r with the same question also gets F.
Motion rows are excluded on purpose (labelled astronaut motion twins differ 1.2-3.7x).  Labels joined only to evaluate.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul083/rul083.py [BUILD] [write]"""
import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
ARGS = [a for a in sys.argv[1:] if a != 'write']
BUILD = ARGS[0] if ARGS else 'sub_xt_Nzz4'
WRITE = 'write' in sys.argv[1:]
TAG = 'rul083'
TPL = ('scene_prior', 'patch:rul033', 'patch:rul031', 'patch:rul054', 'patch:rul067', 'patch:rul075')
UNIT = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
DIMW = r'(?:length|height|width|diameter|wingspan|wing span|size|radius|thickness|depth)'
OBJ = re.compile(DIMW + r'\s+of\s+(?:the\s+)?([a-z][a-z ]*?)\s*=\s*([0-9]*\.?[0-9]+)\s*(mm|cm|m)(?![a-z/^0-9])')


def pnorm(p):
    return re.sub(r'\s+', '', str(p).lower()).rstrip('.')


def qnorm(q):
    return re.sub(r'\s+', ' ', str(q).lower().strip())


def src0(s):
    return str(s).replace('mo_guard:', '')


fr = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    det = pd.read_csv(f'{R}/submissions/{BUILD}_{suf}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    assert (raw.video_id.astype(str).reindex(det.index).values == det.video_id.astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f'{R}/geo/out/parse_{sp}_v2.jsonl'))}
    d = pd.DataFrame({'sp': sp, 'rid': det.index.astype(int), 'vid': det.video_id.astype(str).values,
                      'cat': det['cat'].values, 'ans': det.answer.astype(float).values,
                      'src': [src0(s) for s in det.answer_source], 'g': det.g.values, 'cs': det.cs.values,
                      'question': raw.question.reindex(det.index).astype(str).values,
                      'prior': raw.ground_truth_prior.reindex(det.index).astype(str).values})
    d['kind'] = [(PP[int(i)].get('q') or {}).get('kind') for i in det.index]
    d['unit'] = [(PP[int(i)].get('q') or {}).get('unit') for i in det.index]
    d['grav'] = [any(x.get('gravity') for x in (PP[int(i)].get('priors') or [])) for i in det.index]
    fr.append(d)
D = pd.concat(fr, ignore_index=True)
D['base'] = D.vid.str.replace('_segmented', '', regex=False)
D['seg'] = D.vid.str.endswith('_segmented')
D['ql'] = D.question.map(qnorm)
D['pn'] = D.prior.map(pnorm)
S = D[D.base.str.startswith('simulation_')].copy()

# stated sizes in simulation priors: (head noun of obj, value in m)
stated = []
for pr in S.drop_duplicates('base').prior:
    for o, v, u in OBJ.findall(str(pr).lower()):
        head = o.split(' of ')[0].split()
        if head:
            stated.append((head[-1], float(v) / UNIT[u]))
print('stated size facts', len(stated))

plainq = set(zip(S[~S.seg].base, S[~S.seg].ql))
S['recv_ok'] = [(not sg) or ((b, q) not in plainq) for sg, b, q in zip(S.seg, S.base, S.ql)]
rows = []
for i, r in S.iterrows():
    if not (r.kind == 'size' and r.unit == 'm' and r.recv_ok and not r.grav):
        continue
    if r.src in TPL or r.src.startswith('patch:c48'):
        continue
    don = S[(S.base != r.base) & (S.pn == r.pn) & (S.ql == r.ql) & S.src.isin(TPL)]
    if not len(don):
        continue
    F = don.ans.astype(float).values
    if F.max() / F.min() > 1.02:
        continue
    f = float(np.median(F))
    hit = [s for nn, s in stated if re.search(r'\b' + re.escape(nn) + r'\b', r.ql) and abs(f / s - 1) <= 0.02]
    if not hit:
        continue
    if abs(r.ans / f - 1) <= 0.02:
        continue
    dg = [x for x in don.g if x == x and x > 0]
    rows.append(dict(sp=r.sp, rid=int(r.rid), vid=r.vid, cat=r['cat'], q=r.question, prior=r.prior, ans=r.ans, src=r.src,
                     g=r.g, cs=r.cs, F=f, ndon=len(don), gwit=(r.g / np.median(dg) if dg and r.g == r.g else np.nan),
                     donors=','.join(sorted(set(f'{v[11:]}:{s.replace("patch:", "")}' for v, s in zip(don.vid, don.src))))))
X = pd.DataFrame(rows, columns=['sp', 'rid', 'vid', 'cat', 'q', 'prior', 'ans', 'src', 'g', 'cs', 'F', 'ndon', 'gwit', 'donors'])
seg_add = []
for r in X.itertuples():
    cp = D[(D.sp == r.sp) & (D.vid == r.vid + '_segmented') & (D.ql == qnorm(r.q))]
    for c in cp.itertuples():
        if abs(c.ans / r.F - 1) > 0.02:
            seg_add.append(dict(sp=c.sp, rid=int(c.rid), vid=c.vid, cat=c.cat, q=c.question, prior=c.prior, ans=c.ans,
                                src=c.src, g=c.g, cs=c.cs, F=r.F, ndon=r.ndon, gwit=np.nan, donors=f'plain twin {r.rid}'))
ALL = pd.concat([X, pd.DataFrame(seg_add, columns=X.columns)], ignore_index=True)
pd.set_option('display.width', 320); pd.set_option('display.max_colwidth', 64); pd.set_option('display.max_rows', 300)
print('fired', len(X), '+ segmented copies', len(seg_add))
print(ALL.drop(columns=['prior']).round(3).to_string())

# ---- evaluation (labels only evaluate)
import analyze_caw as A
from lab_rows import labelled
L = labelled(BUILD)
lab = {(s, int(i)): y for s, i, y in zip(L.split, L.index, L.y)}
ALL['y'] = [lab.get((s, int(i)), np.nan) for s, i in zip(ALL.sp, ALL.rid)]
Y = ALL[ALL.y.notna()]
print('labelled changed rows:', len(Y))
if len(Y):
    print(Y[['sp', 'rid', 'vid', 'ans', 'F', 'y']].to_string())
    print('before', A.row_mra(Y.ans.values, Y.y.values).mean(), 'after', A.row_mra(Y.F.values, Y.y.values).mean())
print('test rows changed by cat', ALL[ALL.sp == 'test'].groupby('cat').size().to_dict(), '| val', int((ALL.sp == 'val').sum()))
# (a) TPL-source accuracy on labelled simulation size rows (donor reliability)
S['y'] = [lab.get((s, int(i)), np.nan) for s, i in zip(S.sp, S.rid)]
T = S[S.y.notna() & (S.kind == 'size') & S.src.isin(TPL)].drop_duplicates(['vid', 'ql'])
print('\n(a) labelled sim size rows answered by TPL sources: n=%d, MRA %.3f, exact(2%%) %d' %
      (len(T), A.row_mra(T.ans.values, T.y.values).mean() if len(T) else np.nan, int((abs(T.ans / T.y - 1) <= 0.02).sum())))
print(T[['sp', 'rid', 'vid', 'question', 'ans', 'src', 'y']].to_string() if len(T) else '')
# (b) label agreement of identical-question simulation size rows across different bases (same asset -> same size?)
Z = S[S.y.notna() & (S.kind == 'size')].drop_duplicates(['base', 'ql'])
pairs = []
for q, gq in Z.groupby('ql'):
    gq = gq.reset_index(drop=True)
    for a in range(len(gq)):
        for b in range(a + 1, len(gq)):
            if gq.base[a] != gq.base[b]:
                pairs.append(dict(q=q[:60], b1=gq.base[a][11:], b2=gq.base[b][11:], y1=gq.y[a], y2=gq.y[b],
                                  same_prior=gq.pn[a] == gq.pn[b], agree=abs(gq.y[a] / gq.y[b] - 1) <= 0.02))
P = pd.DataFrame(pairs)
if len(P):
    print('\n(b) labelled identical-question size pairs across bases: n=%d, labels agree(2%%) %d, identical prior %d' %
          (len(P), int(P.agree.sum()), int(P.same_prior.sum())))
    print(P.to_string())
out = {'test': {}, 'val': {}}
for r in ALL.itertuples():
    out[r.sp][str(r.rid)] = {'value': float(r.F), 'why': f'{TAG}: identical-prior-text twin(s) [{r.donors}] answer the '
                             f'template-stated size {r.F:g} m (stated in a simulation prior); was {r.ans:.4g} ({r.src})'}
if WRITE:
    for sp in ('test', 'val'):
        json.dump(out[sp], open(f'{R}/cache/tmp/ag_{TAG}_patch_{sp}.json', 'w'), indent=1)
        print('wrote', f'cache/tmp/ag_{TAG}_patch_{sp}.json', len(out[sp]))
