"""Collect round-4 audit proposals into one table and a build_probe spec (2026-10-20, main agent).

Inputs: analysis/v3/ag/r4_<key>/rows.csv of every audit agent (key != pilot unless --pilot), cross-checked against
cache/tmp/ag_r4_<key>_patch_test.json. Checks: scored rows only; never a label-held row (dossier column lab); one value per
row (conflicts between agents are reported and dropped); copies of one question in a family (plain / s / x / segmented)
get one value (a proposal that leaves a scored copy at the old value is reported).
Expected gain per row = P * mra_if_right + (1 - P) * mra_if_wrong, recomputed here from old / new with the row MRA
(if right: truth = new; if wrong: truth = old), and P scaled by --pscale (calibration from the blind pilot).
Usage: python analysis/v3/ag/rul351/collect.py [--pmin 0.6] [--pscale 1.0] [--keys k1,k2] [--spec NAME]
Writes --table (default analysis/v3/calib/r4/collect_rows_latest.csv; the round-4 record collect_rows.csv, read by
build_r4.py, is never overwritten) and, with --spec (refused if the spec exists), analysis/v3/calib/probespec/<NAME>.tsv plus the merged
patch JSON cache/tmp/ag_r4_<NAME>_patch_test.json (one block per category).
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import dossier as D

TH = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
N = dict(S2=532, D2=1109, S3=488, D3=881)


def mra(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return ((np.abs(p / y - 1))[:, None] < (1 - TH)).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmin', type=float, default=0.0)
    ap.add_argument('--pmin-one', type=float, default=0.0, help='minimum P for one-route rows (two-route rows use --pmin)')
    ap.add_argument('--ptwo', type=float, help='calibrated P for every two-route row (pilot); default: the agent P')
    ap.add_argument('--pone', type=float, help='calibrated P for every one-route row (pilot); default: the agent P')
    ap.add_argument('--pscale', type=float, default=1.0)
    ap.add_argument('--keys')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--base', default='best24v')
    ap.add_argument('--table', default='analysis/v3/calib/r4/collect_rows_latest.csv',
                    help='output table (the round-4 record collect_rows.csv is never overwritten)')
    a = ap.parse_args()
    t = D.load().set_index('id0')
    frames = []
    for d in sorted(glob.glob(f'{R}/analysis/v3/ag/r4_*/rows.csv')):
        key = d.split('/')[-2]
        if key == 'rul351' or (key == 'r4_pilot' and not a.pilot):
            continue
        if a.keys and key not in a.keys.split(','):
            continue
        x = pd.read_csv(d)
        x['key'] = key
        js = f'{R}/cache/tmp/ag_{key}_patch_test.json'
        jv = json.load(open(js)) if os.path.exists(js) else {}
        x['in_json'] = x.id0.astype(str).isin(jv.keys())
        frames.append(x)
    if not frames:
        sys.exit('no rows.csv found')
    X = pd.concat(frames, ignore_index=True)
    X['id0'] = X.id0.astype(int)
    X['new'] = pd.to_numeric(X.new, errors='coerce')
    X['P'] = pd.to_numeric(X.get('P'), errors='coerce').fillna(0.5)
    X['cat'] = X.id0.map(t['cat'])
    X['sc'] = X.id0.map(t.sc).fillna(False).astype(bool)
    X['lab'] = X.id0.map(t.lab).fillna('')
    X['cur'] = X.id0.map(t.cur)
    X['fam'] = X.id0.map(t.fam)
    X['qn'] = X.id0.map(t.question).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    bad = X[~X.sc | (X.lab != '') | X.new.isna() | (X.new <= 0)]
    if len(bad):
        print('DROPPED (unscored, label-held or invalid):', len(bad))
        print(bad[['key', 'id0', 'sc', 'lab', 'new']].to_string())
    X = X[X.sc & (X.lab == '') & X.new.notna() & (X.new > 0)]
    X = X[~np.isclose(X.new, X.cur, rtol=1e-9)]
    # conflicts: two agents writing values more than 0.1 % apart; equal values (within 0.1 %) are merged
    g = X.groupby('id0').new.agg(['min', 'max'])
    conf = g[g['max'] > g['min'] * 1.001].index
    if len(conf):
        print('CONFLICTS (dropped):')
        print(X[X.id0.isin(conf)][['key', 'id0', 'cur', 'new', 'P']].to_string())
    X = X[~X.id0.isin(conf)].drop_duplicates('id0')
    # copies left behind
    T = t.reset_index()
    T['qn'] = T.question.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    miss = []
    for (fam, qn), grp in X.groupby(['fam', 'qn']):
        cp = T[(T.fam == fam) & (T.qn == qn) & T.sc & (T.lab == '')]
        left = cp[~cp.id0.isin(X.id0)]
        left = left[np.isclose(left.cur, grp.cur.iloc[0], rtol=0.02)]
        if len(left):
            miss.append((fam, qn[:60], list(left.id0)))
    if miss:
        print('COPIES NOT CHANGED (same old value):')
        for m in miss:
            print('  ', m)
    X['m_right'] = mra(X.new, X.new) - mra(X.cur, X.new)
    X['m_wrong'] = mra(X.new, X.cur) - mra(X.cur, X.cur)
    kind = X['kind'].astype(str).str.lower()
    # "two-route (shared track)" and similar are NOT independent routes (fix of 2026-10-20 after the best25v S3 reading:
    # row 1664, a claim C row with a shared tracker, had been counted as two-route)
    X['two'] = kind.str.contains('two') & ~kind.str.contains('shared|same track|same tracker|caveat')
    # brief rule 4: rows written by portal-read blocks change only with an independent two-route proposal
    blk = set()
    for f, col, pat in [('analysis/v3/calib/probe8b_rows.csv', 'block', '^claimA'), ('analysis/v3/calib/probe10_rows.csv', 'block',
                         'claimC|farmfront|tspendaq|animlink'), ('analysis/v3/calib/claimC/claimC_sel_rows.csv', None, None)]:
        b = pd.read_csv(f'{R}/{f}')
        blk |= set(b.id0 if col is None else b[b[col].astype(str).str.contains(pat)].id0)
    wr = X.id0.map(t.writers).fillna('')
    X['portal_block'] = X.id0.isin(blk) | wr.str.contains('rul345|rul348|rul451|w1_labradius|w1_deskcorner|s3m_walk|pxwalk|rul335')
    X['Pc'] = (X.P * a.pscale).clip(0, 1)
    if a.ptwo is not None:
        X.loc[X.two, 'Pc'] = a.ptwo
    if a.pone is not None:
        X.loc[~X.two, 'Pc'] = a.pone
    X['ev'] = X.Pc * X.m_right + (1 - X.Pc) * X.m_wrong
    X['ev_pts'] = X.ev / X.cat.map(N)
    X['sel'] = (X.P >= a.pmin) & (X.two | (X.P >= a.pmin_one)) & (X.ev > 0) & (~X.portal_block | X.two)
    os.makedirs(os.path.dirname(os.path.join(R, a.table)), exist_ok=True)
    if os.path.abspath(os.path.join(R, a.table)) == os.path.join(R, 'analysis/v3/calib/r4/collect_rows.csv'):
        sys.exit('refusing to overwrite the round-4 record analysis/v3/calib/r4/collect_rows.csv (build_r4.py reads it)')
    X.to_csv(os.path.join(R, a.table), index=False)
    s = X[X.sel]
    print(f'proposals {len(X)}, selected {len(s)} (pmin {a.pmin}, pscale {a.pscale})')
    tab = s.groupby(['key', 'cat']).agg(n=('id0', 'size'), ev_rowmra=('ev', 'sum'), ev_pts=('ev_pts', 'sum'))
    print(tab.round(4).to_string())
    by = s.groupby('cat').agg(n=('id0', 'size'), ev_pts=('ev_pts', 'sum'),
                              right_pts=('m_right', lambda v: (v / s.loc[v.index, 'cat'].map(N)).sum()),
                              wrong_pts=('m_wrong', lambda v: (v / s.loc[v.index, 'cat'].map(N)).sum()))
    print(by.round(4).to_string())
    print('expected mean gain', round(by.ev_pts.sum() / 4, 4), '| all right', round(by.right_pts.sum() / 4, 4),
          '| all wrong', round(by.wrong_pts.sum() / 4, 4))
    if a.spec:
        if os.path.exists(f'{R}/analysis/v3/calib/probespec/{a.spec}.tsv'):
            sys.exit(f'probespec/{a.spec}.tsv exists; choose a new name')
        lines = [f'# {a.spec}: {a.base} + round-4 audit rows (collect.py --pmin {a.pmin} --pscale {a.pscale})',
                 f'name\t{a.spec}', f'base\t{a.base}']
        for c in ['S2', 'D2', 'S3', 'D3']:
            sc = s[s.cat == c]
            if not len(sc):
                continue
            js = f'cache/tmp/ag_r4_{a.spec}_{c}_patch_test.json'
            json.dump({str(int(r.id0)): {'value': float(r.new), 'why': f'{r.key}: {str(getattr(r, "mechanism", ""))[:120]}'}
                       for r in sc.itertuples()}, open(f'{R}/{js}', 'w'), indent=0)
            lines.append(f'block\tr4_{c}\t{c}\tjson:{js}')
        open(f'{R}/analysis/v3/calib/probespec/{a.spec}.tsv', 'w').write('\n'.join(lines) + '\n')
        print('spec written:', f'analysis/v3/calib/probespec/{a.spec}.tsv')


if __name__ == '__main__':
    main()
