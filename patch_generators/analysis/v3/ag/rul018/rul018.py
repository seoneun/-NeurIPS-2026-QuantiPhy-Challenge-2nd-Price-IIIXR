"""rul018: template-stated time-less speed.
Simulation scenes of one template (connected component of bases sharing >= 2 question texts, pooled over test+val) reuse the
same animation; a time-less prior 'speed of X = v m/s' stated in another base of the template answers
 A  average / whole-clip speed questions about X when the row's own geo is within 1.25x of v;
 A2 the same when own geo fails, if >= 3 avg-form rows about X in other bases of the template have geo whose median ratio to v is
    within 1.15x (>= 2/3 within 1.25x), unless a second own-base row about X confirms the row's own off-template geo (guard);
 B  instantaneous 'at t' speed questions: answer geo when geo is within 1.25x of v and CaW is outside [v/2, 2v].
Output = final answers (A/A2: v, B: g). Labels are only used for evaluation below."""
import sys, re as re_, json, os
import pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 500)
SUB = os.environ.get('C21_SUB', 'sub_xt_Nzn')
DEDUP = int(os.environ.get('C21_DEDUP', '1'))     # count each (video, question) once as an A2 witness
RULES = set(os.environ.get('C21_RULES', 'A,A2,B').split(','))
STOP2_ = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}


def nobj_(x):
    return " ".join(w for w in re_.sub(r"[^a-z ]", " ", x.lower()).split() if w not in STOP2_)


def sprior_(p):
    m_ = re_.match(r"\s*(?:the )?(?:average )?speed of (?:the )?(.+?)\s*=\s*([\d.]+)\s*m/s\s*$", str(p).lower().strip())
    if not m_ or re_.search(r"\d\s*s\b|at t|from|between", str(p).lower().split("=")[0]):
        return None
    return nobj_(m_.group(1)), float(m_.group(2))


def avgform(ql):
    isavg = ('average' in ql) or ('beginning to the end' in ql) or bool(re_.search(r"^what is the speed of (?:the )?[a-z ]+ in m/s", ql))
    return isavg and not re_.search(r"from \d|between \d", ql)


def atform(ql):
    return bool(re_.search(r" at (?:time )?\d", ql))


def rowobj(ql):
    if "m/s" not in ql or "m/s^2" in ql:
        return None
    mo = re_.search(r"(?:speed|velocity) of (?:the )?(.+?)(?: at | from | between | in m/s|\?|,| during| when| while)", ql)
    return nobj_(mo.group(1)) if mo else None


def gv(x):
    return x == x and x > 0


L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]
frames = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    x = raw[['video_id', 'question', 'ground_truth_prior']].copy()
    x['sp'] = sp; x['rid'] = x.index
    for c_ in ('answer', 'answer_source', 'cs', 'g', 'cat'):
        x[c_] = det[c_].reindex(x.index).values
    x['y'] = (det.y if sp == 'val' else Lt.y).reindex(x.index).values
    frames.append(x)
d = pd.concat(frames, ignore_index=True)
d['video_id'] = d.video_id.astype(str)
d['ql'] = d.question.astype(str).str.strip().str.lower()
d['base'] = d.video_id.str.replace("_segmented", "", regex=False)
issim = d.base.str.startswith("simulation_")
# (1) template links over the pooled question sets, connected components
qsets = d[issim].groupby('base').ql.apply(set)
sb = list(qsets.index)
link = {b: [c for c in sb if c != b and len(qsets[b] & qsets[c]) >= 2] for b in sb}
comp = {}
for b0 in sb:
    if b0 in comp:
        continue
    st, seen = [b0], {b0}
    while st:
        x_ = st.pop()
        for y_ in link[x_]:
            if y_ not in seen:
                seen.add(y_); st.append(y_)
    for x_ in seen:
        comp[x_] = seen
# (2) stated time-less speeds per base (every distinct prior text of the base)
spri = {}
for b, g_ in d[issim].groupby('base'):
    spri[b] = {sprior_(p) for p in g_.ground_truth_prior.unique()} - {None}
print('bases', len(sb), 'with stated speed', sum(1 for b in spri if spri[b]),
      'with >1 distinct stated speed', sum(1 for b in spri if len(spri[b]) > 1))
# (3) row objects and question forms
d['obj'] = [rowobj(q) if s else None for q, s in zip(d.ql, issim)]
d['isavg'] = d.ql.map(avgform); d['isat'] = d.ql.map(atform)
d['gok'] = d.g.map(gv)
res = []; blocked = []
for i in d.index[d.obj.notna()]:
    b = d.base[i]; o = d.obj[i]
    if any(oo == o for oo, _ in spri.get(b, ())):
        continue
    vals = [val for c in sorted(comp[b] - {b}) for oo, val in spri.get(c, ()) if oo == o]
    if not vals or max(vals) / min(vals) > 1.02:
        continue
    v = float(np.median(vals)); g = d.g[i]; c = d.cs[i]; a = d.answer[i]
    lo, hi = v / 1.25, 1.25 * v
    new = np.nan; rule = ''; why = ''
    if d.isavg[i] and gv(g) and lo <= g <= hi:
        new = v; rule = 'A'
        why = f'rul018 A: template states speed of {o} = {v:g} m/s in {len(vals)} other base(s); own geo {g:.3g} within 1.25x'
    elif d.isavg[i]:
        W = d[(d.obj == o) & d.isavg & d.base.isin(comp[b] - {b}) & d.gok]
        if DEDUP:
            W = W.drop_duplicates(['video_id', 'ql'])
        if len(W) >= 3:
            r_ = W.g.values / v
            if 1 / 1.15 < np.median(r_) < 1.15 and np.mean((r_ > 1 / 1.25) & (r_ < 1.25)) >= 2 / 3:
                # guard: another own-base row about the same object confirms this row's off-template geo
                G = d[(d.base == b) & (d.obj == o) & d.gok & ~((d.video_id == d.video_id[i]) & (d.ql == d.ql[i]))]
                hit = G[(G.g >= g / 1.25) & (G.g <= g * 1.25) & ((G.g < lo) | (G.g > hi))] if gv(g) else G.iloc[:0]
                if len(hit):
                    blocked.append((d.sp[i], d.rid[i], d.video_id[i], d.cat[i], d.ql[i][:60], v, g, a,
                                    list(zip(hit.sp, hit.rid, hit.video_id, hit.g.round(3)))))
                else:
                    new = v; rule = 'A2'
                    why = (f'rul018 A2: template states speed of {o} = {v:g} m/s; own geo off, {len(W)} sibling avg rows '
                           f'median geo/v {np.median(r_):.2f}')
    elif d.isat[i] and gv(g) and lo <= g <= hi and gv(c) and not (v / 2 <= c <= 2 * v) and abs(a - g) > 0.02 * g:
        new = g; rule = 'B'
        why = f'rul018 B: at-t speed, geo {g:.3g} within 1.25x of template speed {v:g}, CaW {c:.3g} outside 2x'
    if rule not in RULES or new != new or abs(new - a) <= 1e-3 * abs(a):
        continue
    res.append(dict(rule=rule, sp=d.sp[i], rid=int(d.rid[i]), vid=d.video_id[i], cat=d.cat[i], q=str(d.question[i])[:72], v=v,
                    ans=a, g=g, cs=c, src=d.answer_source[i], new=new, y=d.y[i], why=why))
X = pd.DataFrame(res, columns=['rule', 'sp', 'rid', 'vid', 'cat', 'q', 'v', 'ans', 'g', 'cs', 'src', 'new', 'y', 'why'])
print('changed', len(X), X.groupby(['sp', 'rule', 'cat']).size().to_dict())
print('A2 guard blocked:')
for z in blocked:
    print('  ', z)
Lb = X[X.y == X.y]
if len(Lb):
    a0 = A.row_mra(Lb.ans.values, Lb.y.values); a1 = A.row_mra(Lb.new.values, Lb.y.values)
    print('labelled n=%d before %.2f after %.2f  worse %d better %d' % (len(Lb), a0.sum(), a1.sum(), (a1 < a0 - 1e-9).sum(),
                                                                        (a1 > a0 + 1e-9).sum()))
    print(Lb.drop(columns=['why']).assign(m0=a0, m1=a1).round(3).to_string())
T = X[X.sp == 'test']
print(T.drop(columns=['why']).round(3).to_string())
if os.environ.get('C21_WRITE', '0') == '1':
    for sp in ('test', 'val'):
        P = {str(r.rid): {'value': float(r.new), 'why': r.why} for r in X[X.sp == sp].itertuples()}
        TAG = os.environ.get('C21_TAG', 'rul018')
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print('wrote', TAG, sp, len(P))
        for k, z in P.items():
            print('   ', k, round(z['value'], 4), z['why'])
