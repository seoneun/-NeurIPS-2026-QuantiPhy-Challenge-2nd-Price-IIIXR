"""Score the blind pilot (key r4_pilot) against the labels it could not see (2026-10-20, main agent).

Truth proxy = the best24v value of rows that hold a validation label (dossier column `lab` non-empty); the pilot saw the
pre-label answer (audit_test.csv `answer`) on those rows. For every proposed row with a truth proxy, prints old / new / truth,
row MRA before and after, and the totals by P bin and by route kind.
Usage: python analysis/v3/ag/rul351/score_pilot.py [rows.csv]
"""
import os, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import dossier as D

TH = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])


def mra(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return ((np.abs(p / y - 1))[:, None] < (1 - TH)).mean(1)


rows = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else f'{R}/analysis/v3/ag/r4_pilot/rows.csv')
t = D.load()
lab = t[(t.lab != '') & t.sc].set_index('id0')
rows = rows[rows.id0.isin(lab.index)].copy()
rows['truth'] = rows.id0.map(lab.cur)
rows['pre'] = rows.id0.map(lab.answer)
rows['m_old'] = mra(rows.pre, rows.truth)
rows['m_new'] = mra(rows.new, rows.truth)
rows['gain'] = rows.m_new - rows.m_old
rows['cat'] = rows.id0.map(lab['cat'])
pd.set_option('display.width', 250)
print(rows[['id0', 'cat', 'pre', 'new', 'truth', 'm_old', 'm_new', 'gain', 'P'] +
           [c for c in ['kind', 'mechanism'] if c in rows]].to_string())
print('n', len(rows), 'gain sum', round(rows.gain.sum(), 2), 'improved', (rows.gain > 0).sum(), 'worse', (rows.gain < 0).sum())
for c in ['kind']:
    if c in rows:
        print(rows.groupby(c).agg(n=('gain', 'size'), gain=('gain', 'sum'), better=('gain', lambda g: (g > 0).sum()),
                                  worse=('gain', lambda g: (g < 0).sum())).to_string())
rows['Pbin'] = pd.cut(rows.P.astype(float), [0, .5, .7, .85, 1.0])
print(rows.groupby('Pbin', observed=False).agg(n=('gain', 'size'), gain=('gain', 'sum'),
                                               better=('gain', lambda g: (g > 0).sum()),
                                               worse=('gain', lambda g: (g < 0).sum())).to_string())
