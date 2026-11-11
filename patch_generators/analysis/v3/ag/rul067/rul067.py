"""rul067: coverage extension of rul033 (template-stated size) / scene_prior 3 -- object identity up to a leading motion participle.

rul033 answers a simulation size row '<dim> of X' with a size that another member of the same template states as its prior
('length of the yellow car = 5.44m'), with the same normalised object X. Its object test is exact string identity
(nobj(target) == nobj(fact object)), so a question that names the same asset with a leading motion / state participle
('the length of the TURNING yellow car') is never linked to the fact 'length of the yellow car = 5.44m', although a participle
describes what the object does in that clip, not which asset it is. Coverage gap: 15 rows (13 test + 1 val + 1 test copy of a
val row; 2 of the test rows are _segmented copies that follow their plain row).

Lifted limit (the only change against rul033, code otherwise identical): the receiver object o and the fact object fo match when
strip_ing(o) == strip_ing(fo), strip_ing = drop leading words ending in 'ing' (not ring/king/wing/string/spring/swing/building/
ceiling/painting/railing/ping/thing). Only rows that rul033 cannot link (no exact fact) are considered.
Extra guard: the receiver's own base asks about no other object with the same stripped name but a different full name (then the
participle would distinguish two assets, e.g. a turning and a parked yellow car).
Gates kept from rul033: template = connected component of simulation bases sharing >= 2 questions (>= 3 when facts disagree);
linked facts agree within 2 %; own base states no such size (stripped or not); witness = own geo or CaW within 1.25x of s on
every plain row of (base, object, dim); scene gate (a fact base shares a question about ANOTHER object with the receiver base);
skip rows already within 2 %, scene_prior / xrow_prior / patch:rul018 / patch:rul020, and rows any existing patch set.
Output s in the question unit (m). Labels are joined at the end for evaluation only.
usage: python rul067.py [write]   (env C43_SUB, default sub_xt_Nzz)"""
import sys, re, json, os
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 2000)
SUB = os.environ.get('C43_SUB', 'sub_xt_Nzz')
WRITE = 'write' in sys.argv[1:]
TAG = 'rul067' + ('_strict' if os.environ.get('C43_STRICT', '0') == '1' else '')
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
COVER = ('scene_prior', 'xrow_prior', 'patch:rul018', 'patch:rul020')
NOT_PART = {"ring", "king", "wing", "string", "spring", "swing", "building", "ceiling", "painting", "railing", "ping", "thing", "sing"}


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)


def strip_ing(o):
    w = o.split()
    while len(w) > 1 and w[0].endswith("ing") and w[0] not in NOT_PART:
        w = w[1:]
    return " ".join(w)


def src0(s):
    return str(s).replace('mo_guard:', '')


def ed1(a, b):
    """True when a and b differ by at most one inserted / deleted / substituted character"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] or (len(a) == len(b) and a[i + 1:] == b[i + 1:])


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
# objects named by the size / motion questions of each base (for the two-asset guard)
qobj = d.groupby('base').q.apply(lambda z: set(nobj(x.get('target') or '') for x in z) - {''})


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
F = []
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None or x.get('kind') != 'size':
            continue
        F.append((r.base, nobj(x['obj'] or ''), x.get('dim'), float(abs(x['value_si'])), x['raw'].strip(), r.sp))
F = pd.DataFrame(F, columns=['base', 'obj', 'dim', 'v', 'raw', 'fsp']).drop_duplicates(['base', 'raw', 'fsp'])
F['so'] = F.obj.map(strip_ing)
own = set(zip(F.base, F.obj, F.dim)); own_s = set(zip(F.base, F.so, F.dim))
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
        continue
    # rul033's own (exact) link: rows it can link are its own population, not this extension's
    exact = any(f.obj == o and f.dim == dm for c in COMPS[2][r.base] - {r.base} if c in Fb for f in Fb[c].itertuples())
    so = strip_ing(o)
    if exact or so == o or (r.base, so, dm) in own_s:
        continue
    # receiver guard: a third name form of the same stripped object in the receiver base ('parked yellow car') = two assets
    # (a one-character variant of o, e.g. the template's typo 'turing yellow car', is the same name)
    twoasset = any(x not in (o, so) and strip_ing(x) == so and not ed1(x, o) for x in qobj.get(r.base, set()))
    src = None
    for lev in LEVELS:
        cm = COMPS[lev]
        s_ = [(c, f.v, f.raw, f.fsp) for c in sorted(cm[r.base] - {r.base}) if c in Fb for f in Fb[c].itertuples()
              if f.so == so and f.dim == dm]
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
    head = so.split()[-1]
    fb_ = set(z[0] for z in src)
    scq = sorted(set(qq for c in fb_ for qq in (qs[r.base] & qs[c]) if head not in nobj(qq).split()))
    # identity witness: a fact base itself names the stated object with the participle in its own questions (one asset, two names)
    idw = sorted(set(c[-4:] for c in fb_ if any(x != so and strip_ing(x) == so for x in qobj.get(c, set()))))
    # object-level scene evidence: the receiver base and a fact base ask about the same OTHER object (any question text)
    sco = sorted(set(x for c in fb_ for x in (qobj.get(r.base, set()) & qobj.get(c, set())) if head not in x.split()))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, base=r.base, cat=r.cat, dim=dm, q=r.question[:70], obj=o, so=so, v=v, agree=agree, lev=used,
                    nsrc=len(set(z[0] for z in src)), srcb=','.join(sorted(set(z[0][-4:] for z in src))),
                    fsp=','.join(sorted(set(z[3] for z in src))), raw0=src[0][2], ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    gw=within(r.g, v), cw=within(r.cs, v), have2=abs(r.ans - v) <= 0.02 * v, covd=r.src in COVER,
                    patched=r.src.startswith('patch:') or r.src == 'patch_copy', scene=len(scq) > 0, twoasset=twoasset, scq=' | '.join(scq)[:120],
                    idw=','.join(idw), sceneo=len(sco) > 0, sco=','.join(sco)[:60]))
X = pd.DataFrame(out)
X['wit0'] = X.gw | X.cw
pl = X[~X.vid.str.endswith('_segmented')].set_index(['sp', 'base', 'obj', 'dim'])
pgate = pl.wit0.groupby(level=[0, 1, 2, 3]).min()
X['wit'] = [bool(pgate.get((s, b, o, dm), w)) for s, b, o, dm, w in zip(X.sp, X.base, X.obj, X.dim, X.wit0)]
STRICT = os.environ.get('C43_STRICT', '0') == '1'   # 1: rul033's exact-question scene gate; 0: object-level scene gate + identity witness
X['fire'] = X.agree & X.wit & ~X.have2 & ~X.covd & ~X.patched & ~X.twoasset & ((X.scene) if STRICT else (X.sceneo & (X.idw != '')))
# segmented copies whose source is var_ans_plain follow their plain row through the pipeline's patch_copy step; they are written
# too (the patch then holds every changed row's final answer), only when the plain row fires
seg_follow = X.vid.str.endswith('_segmented') & (X.src == 'var_ans_plain')
pf = set(zip(X.sp[X.fire & ~X.vid.str.endswith('_segmented')], X.base[X.fire & ~X.vid.str.endswith('_segmented')], X.q[X.fire & ~X.vid.str.endswith('_segmented')]))
X['fire_copy'] = seg_follow & pd.Series([(s, b, qq) in pf for s, b, qq in zip(X.sp, X.base, X.q)], index=X.index)
X['fire_all'] = X.fire | X.fire_copy
# ---- labels: evaluation only (val, all-237, fuzzy typo copies of val questions)
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
FZ = pd.read_csv(f"{R}/cache/tmp/ag_a42s0_fuzzycopies.csv"); FZ = FZ[FZ.test != 494]
Fy = dict(zip(FZ.test.astype(int), FZ.y.astype(float)))
X['y'] = [(Lv.y.get(i, np.nan) if s == 'val' else Lt.y.get(i, Fy.get(int(i), np.nan))) for s, i in zip(X.sp, X.rid)]
lab = X.y == X.y
X['m0'] = np.where(lab, A.row_mra(X.ans.values, X.y.fillna(1).values), np.nan)
X['m1'] = np.where(lab, A.row_mra(X.v.values, X.y.fillna(1).values), np.nan)
cols = ['sp', 'rid', 'vid', 'cat', 'obj', 'so', 'dim', 'v', 'y', 'ans', 'src', 'g', 'cs', 'agree', 'wit', 'scene', 'sceneo', 'idw', 'twoasset',
        'have2', 'patched', 'fire_all', 'm0', 'm1', 'srcb', 'sco']
print(f'receivers (participle-stripped link only) {len(X)}; fire {int(X.fire.sum())} + segmented copies {int(X.fire_copy.sum())}')
print(X[cols].sort_values(['sp', 'vid']).round(3).to_string())
FI = X[X.fire_all]
print('fired test by cat', FI[FI.sp == 'test'].groupby('cat').size().to_dict(), '| val', int((FI.sp == 'val').sum()))
Lb = FI[FI.y == FI.y]
print('labelled fired n=%d  sum MRA before %.3f after %.3f  worse %d better %d' % (len(Lb), Lb.m0.sum(), Lb.m1.sum(),
      int((Lb.m1 < Lb.m0 - 1e-9).sum()), int((Lb.m1 > Lb.m0 + 1e-9).sum())))
# macro effect on val and all-237
P_ = {sp: {int(r.rid): float(r.v) for r in FI[FI.sp == sp].itertuples()} for sp in ('test', 'val')}
L['new'] = [P_[s].get(int(i), a) for i, s, a in zip(L.index, L.split, L.answer)]
L['m1'] = A.row_mra(L.new.values, L.y.values)
for nm, z in (('val', L[L.split == 'val']), ('all-237', L)):
    mo, mn = A.macro(z.m, z['cat'])[0], A.macro(z.m1, z['cat'])[0]
    print(f'{nm}: n {len(z)} changed {int((~np.isclose(z.new, z.answer)).sum())} macro {mo:.4f} -> {mn:.4f} ({mn - mo:+.4f}) row-units {float((z.m1 - z.m).sum()):+.3f}')
# same-template consistency: the exact-name rows of the same fact (e.g. 'length of the yellow car') in other members
ex = d[(d.q.map(lambda x: x.get('kind')) == 'size')].copy()
ex['o'] = ex.q.map(lambda x: nobj(x.get('target') or '')); ex['dm'] = ex.q.map(lambda x: x.get('dim'))
for (so, dm), g in FI.groupby(['so', 'dim']):
    E = ex[(ex.o == so) & (ex.dm == dm)]
    print(f"same-template rows naming '{so}' {dm} directly: {len(E)}; answers {E.ans.round(3).tolist()} sources {E.src.value_counts().to_dict()}")
if WRITE:
    for sp in ('test', 'val'):
        Z = FI[FI.sp == sp]
        P = {str(int(r.rid)): {'value': float(r.v), 'why': (f"{TAG}: template ({r.srcb}) states {r.raw0!r}; '{r.obj}' = '{r.so}' up to a leading motion "
                                                            f"participle; own base silent; geo {r.g:.3g} / CaW {r.cs:.3g} witness; was {r.ans:.4g} ({r.src})"
                                                            + (" [segmented copy follows its plain row]" if r.fire_copy else ""))}
             for r in Z.itertuples()}
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print('wrote', f"cache/tmp/ag_{TAG}_patch_{sp}.json", len(P))
