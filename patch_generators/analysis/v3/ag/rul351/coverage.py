"""Where the remaining loss is (2026-10-20, main agent; source of HANDOFF_r4_2026-10-20.md, appendix A.3).

Prints:
 1. class-wide multipliers x0.85 .. x1.2 on the 454 labelled rows (validation rows + test rows whose best24v value is a
    validation label; the pre-label answer is audit_test.csv `answer`), by category / source type / size-or-motion;
 2. the share of the validation loss by row-MRA bin;
 3. rule coverage of the scored rows (a row is "rule" if a patch JSON in patches/ or cache/tmp holds its best24v value,
    "r4" if round 4 changed it in best26v, else "none" = plain fusion), by category / source / kind, with category points;
 4. on labelled rows, by fusion source: row MRA of the answer, of geo alone, of CaW alone and of their geometric mean;
 5. the caw_nogeo rows without a rule.
Usage: python analysis/v3/ag/rul351/coverage.py
"""
import os, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import dossier as D

TH = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
N = dict(S2=532, D2=1109, S3=488, D3=881)


def mra(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float); out = np.zeros(len(p))
    ok = np.isfinite(p) & (p > 0)
    out[ok] = ((np.abs(p[ok] / y[ok] - 1))[:, None] < (1 - TH)).mean(1)
    return out


t = D.load()
v = pd.read_csv(f'{R}/analysis/v3/calib/audit_val.csv')
L = t[t.sc & (t.lab != '')].copy(); L['y'] = L.cur; L['ans'] = L.answer
V = v.copy(); V['ans'] = V.answer
cols = ['cat', 'src_type', 'kind', 'ans', 'y', 'g', 'cs', 'source', 'mode']
lab = pd.concat([L[cols], V[cols]]); lab = lab[(lab.ans > 0) & (lab.y > 0)].copy()
lab['m'] = mra(lab.ans, lab.y)
pd.set_option('display.width', 250)

print(f'1. labelled rows: {len(lab)}; multiplier gains (row-MRA summed) by class')
lab['cls'] = lab['cat'] + '_' + lab.src_type.str[:3] + '_' + np.where(lab.kind == 'size', 'size', 'motion')
rows = []
for c, g in lab.groupby('cls'):
    base = mra(g.ans, g.y)
    rows.append(pd.Series({'n': len(g), 'mra': base.mean(), **{f'x{k}': (mra(g.ans * k, g.y) - base).sum()
                                                               for k in (0.85, 0.9, 0.95, 1.05, 1.1, 1.2)}}, name=c))
print(pd.DataFrame(rows).round(3).to_string())
base = mra(lab.ans, lab.y)
print('all labelled rows:', {f'x{k}': round((mra(lab.ans * k, lab.y) - base).sum(), 1) for k in (0.85, 0.9, 0.95, 1.05, 1.1, 1.2)})

print('\n2. validation loss by row-MRA bin')
mv = mra(V.ans, V.y)
b = pd.cut(mv, [-0.01, 0.35, 0.65, 0.95, 1.0], labels=['<=.3', '.4-.6', '.7-.9', '1'])
tb = pd.DataFrame({'bin': b, 'loss': 1 - mv}).groupby('bin', observed=False).agg(n=('loss', 'size'), loss=('loss', 'sum'))
tb['share'] = tb.loss / tb.loss.sum()
print(tb.round(3).to_string())

print('\n3. rule coverage of scored rows')
X = pd.read_csv(f'{R}/analysis/v3/calib/r4/collect_rows.csv')
r4 = set(X[X.sel & (X['cat'] != 'S3')].id0)
s = t[t.sc].copy()
s['rule'] = np.where(s.id0.isin(r4), 'r4', np.where(s.writers != '', 'rule', 'none'))
print(s.rule.value_counts().to_dict())
print('no-rule category points:', {c: round(((s.rule == 'none') & (s['cat'] == c)).sum() / N[c], 3) for c in N})

print('\n4. labelled rows by fusion source (non-patch sources with >= 8 rows): row MRA')
lab['gm'] = np.sqrt(lab.g * lab.cs)
nr = s[s.rule == 'none'].source.value_counts()
rows = []
for src, g in lab.groupby('source'):
    if len(g) < 8 or str(src).startswith('patch'):
        continue
    rows.append((src, len(g), mra(g.ans, g.y).mean(), mra(g.g, g.y).mean(), mra(g.cs, g.y).mean(), mra(g.gm, g.y).mean(),
                 int(nr.get(src, 0))))
print(pd.DataFrame(rows, columns=['source', 'n_lab', 'answer', 'geo', 'caw', 'gm', 'free_no_rule_rows']).round(3).to_string(index=False))

print('\n5. caw_nogeo rows without a rule')
c = s[(s.rule == 'none') & (s.source == 'caw_nogeo')]
print(len(c), c['cat'].value_counts().to_dict())
print(c[['id0', 'video', 'cat', 'kind', 'question', 'cur', 'cs']].to_string(index=False))
