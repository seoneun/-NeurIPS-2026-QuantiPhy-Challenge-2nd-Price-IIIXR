"""Coverage audit of the 'motion family' mechanism (speed / acceleration label stated in the prior of another render of the
same Blender template). Template family = simulation bases sharing >= 2 identical question texts (definition of rule rul018).
For every scored simulation motion row (speed / avgspeed / accel) whose object (normalised noun) has a time-less stated
speed / acceleration in a sibling's prior, print best21v and the stated value; flag |ratio-1| > 5%."""
import re, itertools
import numpy as np, pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv'); M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv')
b21 = pd.read_csv(f'{R}/submissions/sub_xt_best21v_id1.csv').parsed_value.values
A['base'] = A.video.str.replace('_segmented', '', regex=False); A['sc'] = M.scored.values.astype(bool)
S = A[A.src_type == 'simulation']
qs = S.groupby('base').question.apply(lambda s: set(s.str.strip().str.lower()))
par = {b: b for b in qs.index}
def f(x):
    while par[x] != x: x = par[x]
    return x
for a, b in itertools.combinations(qs.index, 2):
    if len(qs[a] & qs[b]) >= 2: par[f(a)] = f(b)
fam = {b: f(b) for b in qs.index}
STOP = {'the', 'moving', 'walking', 'running', 'flying', 'of', 'a'}
def norm(s): return ' '.join(w for w in re.sub(r'[^a-z ]', ' ', s.lower()).split() if w not in STOP)
PR = re.compile(r'^\s*(speed|velocity|acceleration) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*m/s', re.I)
facts = []
for b, g in S.groupby('base'):
    m = PR.search(str(g.prior.iloc[0]))
    if m and ' at ' not in m.group(2) and 'before' not in m.group(2):
        facts.append(dict(base=b, fam=fam[b], q='acc' if m.group(1).lower().startswith('acc') else 'spd', obj=norm(m.group(2)), v=float(m.group(3))))
F = pd.DataFrame(facts)
QQ = re.compile(r'(speed|velocity|acceleration) of (?:the )?(.+?)(?: at | from | in m| between |\?|$)', re.I)
out = []
for _, r in S[S.sc & S.kind.isin(['speed', 'avgspeed', 'accel'])].iterrows():
    m = QQ.search(r.question)
    if not m: continue
    q = 'acc' if m.group(1).lower().startswith('acc') else 'spd'; o = norm(m.group(2))
    c = F[(F.fam == fam[r.base]) & (F.base != r.base) & (F.q == q)]
    c = c[c.obj.apply(lambda x: x == o or (len(x) > 3 and (x in o or o in x)))]
    if len(c) == 0: continue
    vs = sorted(set(c.v)); v = vs[0] if len(vs) == 1 else np.nan
    out.append(dict(id0=r.id0, video=r.video, cat=r['cat'], kind=r.kind, obj=o, stated=vs, b21=round(b21[r.id0], 4),
                    ratio=round(b21[r.id0] / v, 3) if v == v else np.nan, src=r.source, q=r.question[:70]))
O = pd.DataFrame(out); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
O.to_csv(f'{R}/analysis/v3/ag/r3_simconv/h9_motionfam_rows.csv', index=False)
print('linked scored motion rows:', len(O), O.groupby('cat').size().to_dict())
off = O[(O.ratio - 1).abs() > 0.05]
print('rows > 5% off the stated sibling value:', len(off), off.groupby('cat').size().to_dict())
print(off.to_string())
