"""A1 time-window patch (rule twin): re-measured geo (twin_rule.py -> cache/tmp/ag_twin_meas_{split}.csv) gated and fused
like the pipeline, written as FINAL answers to cache/tmp/ag_twin_patch_{split}.json; then the labelled-row checks.
 - the pipeline's 2-decade sanity rule (|log10(geo / prior copy h1)| > 2 -> geo invalid) is applied to the new geo too;
   for an acceleration row it is overridden only when the CaW answer is kinematically impossible on the measured track:
   a_caw * span > bound * vref (two-window: span = midpoint gap, vref = max window mean speed, bound 1.5;
   one window: span = window length, vref = the window's mean speed, bound 3 since |v(b) - v(a)| <= 2 vbar)
 - fusion: within 2x of CaW -> geo alone for dist2 / avgspeed (kind of the ORIGINAL parse, so a "moving distance between
   A s to B s" dist2 question measured as a path keeps the dist2 rule), else gm; outside 2x -> geo in strong modes, else CaW
usage: python analysis/v3/ag/twin/twin_patch.py"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/geo')
from lab_rows import labelled
import analyze_caw as A
from rules import UNIT_SI
SUB = 'sub_xt_Nza'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
STRONG = ('2d_ratio', '3d_calib')
GEO_KINDS = ('dist2', 'avgspeed')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 200)


def fuse(g, cs, mode, kind):
    if not np.isfinite(cs) or cs <= 0:
        return g, 'geo_nocaw'
    r = g / cs
    if 0.5 <= r <= 2.0:
        if kind in GEO_KINDS:
            return g, 'agree_geo_kind'
        if kind == 'size' and mode in STRONG:
            return g ** 0.65 * cs ** 0.35, 'agree_wgm'
        return float(np.sqrt(g * cs)), 'agree_gm'
    return (g, 'dis_geo') if mode in STRONG else (cs, 'dis_caw')


L = labelled(SUB)
lab_ids = {(s, int(i)) for s, i in zip(L.split, L.index)}
summary = {}
for split in ('test', 'val'):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv').set_index('Unnamed: 0')
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv').set_index('Unnamed: 0')
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    fn = f'{R}/cache/tmp/ag_twin_meas_{split}.csv'
    M = pd.read_csv(fn) if os.path.exists(fn) and os.path.getsize(fn) > 2 else pd.DataFrame(columns=['id'])
    patch, rows = {}, []
    for _, m in M.iterrows():
        i = int(m['id']); d = det.loc[i]
        src = str(d['answer_source'])
        if src.startswith('patch'):
            print('skip (patched already)', i, src); continue
        if not bool(m.get('ok')) or not np.isfinite(m['val']) or m['val'] <= 0:
            print('skip (no measurement)', i, m.get('why')); continue
        g = float(m['val']); cs = float(d['cs']); u = P[i]['q']['unit']
        kind = P[i]['q'].get('kind') or str(d['kind'])
        h1 = float(raw.loc[i, 'h1_value'])
        gate = ''
        if np.isfinite(h1) and h1 > 0 and abs(np.log10(g / h1)) > 2:
            imp = (kind == 'accel' and np.isfinite(cs) and cs > 0 and np.isfinite(m.get('span', np.nan)) and np.isfinite(m.get('vref', np.nan))
                   and cs * UNIT_SI[u] * m['span'] > m['bound'] * m['vref'])
            if imp:
                gate = (f"; geo fails the 2-decade rule vs prior copy {h1:g} but CaW {cs:g} needs a speed change "
                        f"{cs * UNIT_SI[u] * m['span']:.2f} m/s > {m['bound']:g}x the measured mean speed {m['vref']:.2f} m/s")
                val, how = g, 'dis_geo_kin'
            else:
                val, how = (cs, 'caw_nogeo') if np.isfinite(cs) and cs > 0 else (np.nan, 'none')
                gate = f'; geo fails the 2-decade rule vs prior copy {h1:g}'
        else:
            val, how = fuse(g, cs, str(m['mode']), kind)
        if not np.isfinite(val) or np.isclose(val, d['answer'], rtol=1e-3):
            continue
        t2s = m['t2'] if np.isfinite(m['t2']) else m['dur']
        w = ("A1 " + m['rule'] + ": geo re-measured " +
             (f"over {m['t1']:g}-{t2s:g}s" if m['rule'] != 'two_window' else f"as (vbar{m['w2']} - vbar{m['w1']}) / midpoint gap") +
             (f" on '{m['phrase']}'" if isinstance(m['phrase'], str) and m['phrase'] else '') +
             f" = {g:.4g} (was {d['g']:.4g}); {how} with CaW {cs:.4g}" + gate)
        patch[str(i)] = dict(value=float(val), why=w)
        rows.append(dict(id=i, video_id=d['video_id'], cat=d['cat'], kind=kind, old=d['answer'], old_src=src, g_old=d['g'], g_new=g, cs=cs,
                         new=val, how=how, labelled=(split, i) in lab_ids, q=d['question'][:80]))
    json.dump(patch, open(f'{R}/cache/tmp/ag_twin_patch_{split}.json', 'w'), indent=1)
    T = pd.DataFrame(rows)
    print(f'== {split}: {len(patch)} rows changed')
    if len(T):
        print(T.drop(columns=['q']).round(4).to_string(index=False))
        for _, t in T.iterrows():
            print(f"  {t['id']} {t['video_id']} | {t['q']} | {t['old']:.4g} -> {t['new']:.4g} | {patch[str(t['id'])]['why']}")
        print('by cat', T['cat'].value_counts().to_dict(), '| labelled among them', int(T.labelled.sum()))
        print('median |log10(new/old)|', round(float(np.median(np.abs(np.log10(T.new / T.old)))), 3))

# ---- label evidence 1: the mechanism (windowed vs whole-clip geo on labelled rows whose window WAS parsed)
fn = f'{R}/cache/tmp/ag_twin_mech.csv'
if os.path.exists(fn):
    D = pd.read_csv(fn)
    B = D[D.g_win.notna() & D.g_whole.notna()]
    print(f'\n[mechanism] labelled rows with a parsed sub-window, geo measured inside the window vs over the whole clip: n={len(B)} '
          f'MRA windowed {B.m_win.mean():.3f} whole-clip {B.m_whole.mean():.3f} (CaW {np.nanmean(B.m_cs):.3f}); by split',
          B.groupby('split').apply(lambda x: (len(x), round(x.m_win.mean(), 3), round(x.m_whole.mean(), 3))).to_dict())

# ---- label evidence 2: fusion for in-band path / dist2 rows (geo alone vs gm) on labelled rows
rows = []
for split in ('val', 'test'):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv').set_index('Unnamed: 0')
    ls = L[L.split == split]
    dd = det.reindex(ls.index)
    dd = dd.assign(y=ls.y.values, split=split)
    rows.append(dd)
Z = pd.concat(rows)
Z = Z[Z.g.notna() & Z.cs.notna() & (Z.cs > 0) & (Z.g > 0)]
Z = Z[(Z.g / Z.cs).between(0.5, 2.0)]
Z['m_g'] = A.row_mra(Z.g.values, Z.y.values); Z['m_gm'] = A.row_mra(np.sqrt(Z.g * Z.cs).values, Z.y.values)
Z['m_c'] = A.row_mra(Z.cs.values, Z.y.values)
print('\n[fusion] labelled in-band rows (geo within 2x of CaW), MRA of geo alone / gm / CaW by kind:')
print(Z.groupby(['kind']).agg(n=('y', 'size'), geo=('m_g', 'mean'), gm=('m_gm', 'mean'), caw=('m_c', 'mean')).round(3).to_string())
Zk = Z[Z.kind.isin(['path', 'dist2', 'disp', 'avgspeed', 'accel'])]
print(Zk.groupby(['kind', 'split']).agg(n=('y', 'size'), geo=('m_g', 'mean'), gm=('m_gm', 'mean')).round(3).to_string())
Zi = Zk[Zk.video_id.str.startswith('internet')]
print('internet only:'); print(Zi.groupby(['kind']).agg(n=('y', 'size'), geo=('m_g', 'mean'), gm=('m_gm', 'mean')).round(3).to_string())
