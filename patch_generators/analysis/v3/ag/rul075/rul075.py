"""rul075: scene_prior (template-stated size) extended to prior-text twins inside a template component.

scene_prior answers a simulation size row '<dim> of X' with a size that another video F of the same scene template states as
its prior, but only when the receiver video shares >= 3 question texts with F (direct link).  The rejected 'trans' extension
(connected components) let a 0.75 m boat reach 9 m ships.  This rule takes a much narrower transitive step:

  template component = connected component of simulation bases sharing >= 2 lower-cased question texts (pooled test + val,
                       as rul033 / rul067);
  V (receiver base) has no direct link (>= 3 shared questions) to any fact holder F stating (X, dim);
  a twin V' of V in the same component has the IDENTICAL normalised prior text (same stated quantity, same value: the same
  scene variant), V' is directly linked to F, and V' already answers its own (X, dim) row with F's fact (within 2 %) through
  a template mechanism (scene_prior / rul033 / rul067);
  => V is treated as linked: its (X, dim) row answers F's fact.
Object match: nobj (rul033) after dropping 'body'/'overall'/'full'/'whole' words and leading motion participles (rul067's
strip_ing).  Gates kept: all linked facts agree within 2 %; V's own prior states no (X, dim) size; V has no second asset
with the same stripped name; skip rows already within 2 %, rows answered by scene_prior / xrow_prior / rul018 / rul020, and
rows any existing patch set.  _segmented copies follow their plain row (var_ans_plain).  Output: the fact value (m).
Labels joined at the end for evaluation only.
usage: python rul075.py [write]   (env C45_SUB, default sub_xt_Nzz2)"""
import sys, re, json, os
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 2000)
SUB = os.environ.get('C45_SUB', 'sub_xt_Nzz2')
WRITE = 'write' in sys.argv[1:]
TAG = 'rul075'
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one",
        "body", "overall", "full", "whole"}
COVER = ('scene_prior', 'xrow_prior', 'patch:rul018', 'patch:rul020')
TPL = ('scene_prior', 'patch:rul033', 'patch:rul067', 'patch_copy', 'var_ans_plain')
NOT_PART = {"ring", "king", "wing", "string", "spring", "swing", "building", "ceiling", "painting", "railing", "ping", "thing", "sing"}


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def strip_ing(o):
    w = o.split()
    while len(w) > 1 and w[0].endswith("ing") and w[0] not in NOT_PART:
        w = w[1:]
    return " ".join(w)


def gv(x):
    return x == x and x is not None and x > 0


def src0(s):
    return str(s).replace('mo_guard:', '')


def pnorm(p):
    return re.sub(r"\s+", "", str(p).lower()).rstrip('.')


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'],
                         pn=pnorm(raw.ground_truth_prior[i]), cat=det.cat[i], ans=float(det.answer[i]), src=src0(det.answer_source[i]),
                         g=det.g[i], cs=det.cs[i]))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower()
d['base'] = d.vid.str.replace("_segmented", "", regex=False)
d = d[d.base.str.startswith('simulation_')].reset_index(drop=True)
d['kind'] = d.q.map(lambda x: x.get('kind')); d['dm'] = d.q.map(lambda x: x.get('dim'))
d['o'] = d.q.map(lambda x: nobj(x.get('target') or '')); d['so'] = d.o.map(strip_ing)
qs = d.groupby('base').ql.apply(set); sb = list(qs.index)
pset = d.groupby('base').pn.apply(set)
qobj = d[d.kind == 'size'].groupby('base').o.apply(set)


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


COMP = components(2)


def direct(b, c):
    return len(qs[b] & qs[c]) >= 3


F = []
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None or x.get('kind') != 'size':
            continue
        F.append((r.base, strip_ing(nobj(x['obj'] or '')), x.get('dim'), float(abs(x['value_si'])), x['raw'].strip()))
F = pd.DataFrame(F, columns=['base', 'so', 'dim', 'v', 'raw']).drop_duplicates(['base', 'so', 'dim', 'v'])
own_s = set(zip(F.base, F.so, F.dim))

out = []
for r in d[(d.kind == 'size') & (d.q.map(lambda x: x.get('unit')) == 'm')].itertuples():
    so, dm = r.so, r.dm
    if not so or (r.base, so, dm) in own_s:
        continue
    S = F[(F.so == so) & (F.dim == dm) & F.base.isin(COMP[r.base] - {r.base})]
    if not len(S):
        continue
    vals = S.v.values; agree = vals.max() / vals.min() <= 1.02; v = float(np.median(vals))
    dlink = sorted(c[-4:] for c in set(S.base) if direct(r.base, c))
    # twins: same component, identical normalised prior text, directly linked to a fact holder, own (X, dim) row = the fact
    tw = []
    for c in sorted(COMP[r.base] - {r.base}):
        if not (pset[c] & pset[r.base]) or (c, so, dm) in own_s:
            continue
        if not any(direct(c, f) for f in set(S.base)):
            continue
        St = S[[direct(c, f) for f in S.base]]        # the facts the twin itself receives (its direct links)
        vt = float(np.median(St.v.values))
        if St.v.max() / St.v.min() > 1.02:
            continue
        z = d[(d.base == c) & (d.kind == 'size') & (d.so == so) & (d.dm == dm)]
        zt = z[(np.abs(z.ans - vt) <= 0.02 * vt) & z.src.isin(TPL)]
        if len(zt):
            # twin-measurement witness: the receiver's own geo (or CaW) for (X, dim) within 1.25x of the twin's own geo (CaW)
            tg = [x for x in zt.g if gv(x)]; tc = [x for x in zt.cs if gv(x)]
            wg = bool(tg) and gv(r.g) and 0.8 <= r.g / float(np.median(tg)) <= 1.25
            wc = bool(tc) and gv(r.cs) and 0.8 <= r.cs / float(np.median(tc)) <= 1.25
            tw.append((c[-4:], ','.join(sorted(set(zt.src))), vt, wg or wc,
                       f"{c[-4:]} g {np.median(tg) if tg else np.nan:.3g} c {np.median(tc) if tc else np.nan:.3g}",
                       St.raw.iloc[0], ','.join(sorted(set(b[-4:] for b in St.base)))))
    # looser alternative (reported only): a directly linked member asks the identical question and answers the fact
    al = sorted(set(c[-4:] for c in COMP[r.base] - {r.base} if direct(r.base, c)
                    for z in d[(d.base == c) & (d.ql == r.ql) & (np.abs(d.ans - v) <= 0.02 * v) & d.src.isin(TPL)].itertuples()))
    twoasset = any(x != r.o and strip_ing(x) == so for x in qobj.get(r.base, set()))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, base=r.base, cat=r.cat, dim=dm, q=r.question[:60], so=so, v=v, agree=agree,
                    nf=S.base.nunique(), fb=','.join(sorted(set(b[-4:] for b in S.base))), raw0=(tw[0][5] if tw else S.raw.iloc[0]), tfb=(tw[0][6] if tw else ''), dlink=','.join(dlink),
                    twin=';'.join(f'{t[0]}:{t[1]}' for t in tw), alt=','.join(al),
                    vt=(float(np.median([t[2] for t in tw])) if tw else np.nan),
                    tagree=(max(t[2] for t in tw) / min(t[2] for t in tw) <= 1.02 if tw else False),
                    twit=any(t[3] for t in tw), twg=';'.join(t[4] for t in tw), ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    have2=abs(r.ans - v) <= 0.02 * v, covd=r.src in COVER, patched=r.src.startswith('patch:') or r.src == 'patch_copy',
                    twoasset=twoasset))
X = pd.DataFrame(out)
WIT = os.environ.get('C45_WIT', '1') == '1'     # 1: twin-measurement witness required
X['v'] = np.where(X.twin != '', X.vt, X.v)
X['have2'] = np.abs(X.ans - X.v) <= 0.02 * X.v
X['fire'] = X.tagree & (X.twit | (not WIT)) & (X.dlink == '') & (X.twin != '') & ~X.have2 & ~X.covd & ~X.patched & ~X.twoasset & ~X.vid.str.endswith('_segmented')
X['fire_alt'] = X.agree & (X.dlink == '') & (X.alt != '') & ~X.have2 & ~X.covd & ~X.patched & ~X.twoasset & ~X.vid.str.endswith('_segmented')
pf = set(zip(X.sp[X.fire], X.base[X.fire], X.q[X.fire]))
X['fire_copy'] = X.vid.str.endswith('_segmented') & (X.src == 'var_ans_plain') & pd.Series([(s, b, qq) in pf for s, b, qq in zip(X.sp, X.base, X.q)], index=X.index)
X['fire_all'] = X.fire | X.fire_copy
# ---- labels: evaluation only
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, np.nan)) for s, i in zip(X.sp, X.rid)]
lab = X.y == X.y
X['m0'] = np.where(lab, A.row_mra(X.ans.values, X.y.fillna(1).values), np.nan)
X['m1'] = np.where(lab, A.row_mra(X.v.values, X.y.fillna(1).values), np.nan)
cols = ['sp', 'rid', 'vid', 'cat', 'q', 'so', 'dim', 'v', 'y', 'ans', 'src', 'g', 'cs', 'agree', 'fb', 'dlink', 'twin', 'twit', 'twg', 'alt', 'have2', 'patched', 'twoasset',
        'fire_all', 'fire_alt', 'm0', 'm1']
print(f'size rows whose component holds a fact for (X, dim): {len(X)}; without a direct link: {int((X.dlink == "").sum())}')
print(X[X.dlink == ''][cols].sort_values(['sp', 'vid']).round(3).to_string())
FI = X[X.fire_all]
print('\nFIRED test by cat', FI[FI.sp == 'test'].groupby('cat').size().to_dict(), '| val', int((FI.sp == 'val').sum()))
print(FI[cols].round(3).to_string())
print('alt variant fires', X[X.fire_alt][['sp', 'rid', 'vid', 'q', 'v', 'ans', 'src', 'y']].round(3).to_string())
Lb = FI[FI.y == FI.y]
print('labelled fired n=%d' % len(Lb), Lb[['sp', 'rid', 'y', 'm0', 'm1']].to_string() if len(Lb) else '')
# ---- consistency check on labelled rows: does a component fact for (X, dim) equal the label?
print('\nlabelled sim size rows with a component fact for (X, dim) (any link):')
XL = X[X.y == X.y]
print(XL[['sp', 'rid', 'vid', 'q', 'v', 'y', 'ans', 'src', 'dlink', 'twin', 'fb']].round(3).to_string())
if WRITE:
    for sp in ('test', 'val'):
        Z = FI[FI.sp == sp]
        P = {str(int(r.rid)): {'value': float(r.v), 'why': (f"{TAG}: template ({r.tfb}) states {r.raw0!r}; own base has no >=3-question link to it, "
                                                            f"but its prior-text twin ({r.twin}) is linked and answers the fact; was {r.ans:.4g} ({r.src}); "
                                                            f"geo {r.g:.3g} CaW {r.cs:.3g}" + (" [segmented copy follows its plain row]" if r.fire_copy else ""))}
             for r in Z.itertuples()}
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print('wrote', f"cache/tmp/ag_{TAG}_patch_{sp}.json", len(P))
