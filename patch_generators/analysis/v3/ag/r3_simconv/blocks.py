"""r3_simconv blocks on best21v (values from our inputs only):
 ffbd    : 36 free-fall speed rows of the drop events (droplet 0198-0201, magnifier 0202-0205, typewriter 0206-0209):
           v = a (round(t fps) - 0.5) / fps  (backward one-frame difference, release at the first video frame, t0 = 0);
           a from the row's own prior, t from the question, fps from test_dataset.parquet.
 birdval : bird family 0088/0090/0091 size rows <- validation labels of simulation_0089 (same scene: 0089 labels 17.98 / 11.0
           equal the stated sibling speeds): big bird length 0.72, big bird width 2.04, small bird width 1.3.
Noise model (latgeo2.md): truth = centre x exp(sigma z), 4000 draws; 'right' and 'wrong' centres given per block.
usage: python blocks.py [write]"""
import json, re, sys
import numpy as np, pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
sys.path.insert(0, f'{R}/analysis/v3/calib'); import fusion_calib as FC
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv'); T = pd.read_parquet(f'{R}/data/test/test_dataset.parquet')
M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv'); sc = M.scored.values.astype(bool)
V = pd.read_csv(f'{R}/analysis/v3/calib/audit_val.csv').set_index('id0')
b21 = pd.read_csv(f'{R}/submissions/sub_xt_best21v_id1.csv').parsed_value.values
N = dict(S2=532, D2=1109, S3=488, D3=881); rng = np.random.default_rng(0); Z = rng.standard_normal((4000, 1))
TQ = re.compile(r'at time ([\d.]+)\s*s', re.I); AC = re.compile(r'acceleration of the .*?=\s*([\d.]+)\s*m/s', re.I)
ff = {}
for i in A.index[A.video.isin([f'simulation_0{n}' for n in range(198, 210)]) & (A.kind == 'speed')]:
    m = TQ.search(A.question[i]); a = float(AC.search(A.prior[i]).group(1)); fps = float(T.fps.iloc[i]); t = float(m.group(1))
    k = round(t * fps); v = a * (k - 0.5) / fps
    assert sc[i] and abs(b21[i] - a * t) < 1e-3 * a * t
    ff[i] = dict(value=round(v, 6), why=f'ffbd: backward one-frame difference of free fall from rest at video frame 0: a (k - 0.5)/fps = {a} x ({k} - 0.5)/{fps:g} = {v:.4f}; was a t = {a * t:.4f} (rule kin)', a=a, t=t, k=k, fps=fps)
bv = {}
lab = {'big_len': V.loc[139, 'y'], 'big_w': V.loc[140, 'y'], 'small_w': V.loc[141, 'y']}
for i in (131, 143, 147): bv[i] = dict(value=float(lab['big_len']), why='birdval: simulation_0089 validation label (length of the big bird); same scene (0089 labels 17.98 / 11.0 = sibling stated speeds)')
bv[132] = dict(value=float(lab['big_w']), why='birdval: simulation_0089 validation label (width of the big bird); same scene')
bv[134] = dict(value=float(lab['small_w']), why='birdval: simulation_0089 validation label (width of the small bird); same scene')
for i in bv: assert sc[i] and abs(b21[i] - bv[i]['value']) > 1e-9
def gain(ids, new, centre, sig):
    ids = np.array(ids); old = b21[ids]; y = centre[None, :] * np.exp(sig * Z)
    g = (FC.row_mra(np.broadcast_to(new, y.shape).ravel(), y.ravel()) - FC.row_mra(np.broadcast_to(old, y.shape).ravel(), y.ravel())).reshape(y.shape).mean(0)
    cats = A['cat'].values[ids]; return {c: round(float(g[cats == c].sum()), 2) for c in sorted(set(cats))}
ids = sorted(ff); new = np.array([ff[i]['value'] for i in ids]); a_ = np.array([ff[i]['a'] for i in ids]); k_ = np.array([ff[i]['k'] for i in ids]); f_ = np.array([ff[i]['fps'] for i in ids]); t_ = np.array([ff[i]['t'] for i in ids])
print('ffbd rows per cat:', pd.Series(A['cat'].values[ids]).value_counts().to_dict())
for s in (0.0, 0.3, 0.4, 0.45, 0.5): print(f'  bound s_true={s}:', gain(ids, new, a_ * (k_ - s) / f_, 0.0))
print('  bound truth = a t exactly (H0):', gain(ids, new, a_ * t_, 0.0))
for sig in (0.02, 0.05):
    S_ = rng.uniform(0.3, 0.5, (4000, 1)); y = a_ * (k_ - S_) / f_ * np.exp(sig * Z); old = b21[ids]
    g = (FC.row_mra(np.broadcast_to(new, y.shape).ravel(), y.ravel()) - FC.row_mra(np.broadcast_to(old, y.shape).ravel(), y.ravel())).reshape(y.shape).mean(0)
    cats = A['cat'].values[ids]
    print(f'  noise sigma {sig}: right (s~U[0.3,0.5]):', {c: round(float(g[cats == c].sum()), 2) for c in ('D2', 'D3')}, '| wrong (H0 a t):', gain(ids, new, a_ * t_, sig))
bi = sorted(bv); bn = np.array([bv[i]['value'] for i in bi])
print('birdval rows:', {i: (round(b21[i], 4), bv[i]['value']) for i in bi})
print('  bound right:', gain(bi, bn, bn, 0.0), '| bound wrong (truth = b21):', gain(bi, bn, b21[bi], 0.0))
for sig in (0.02, 0.05): print(f'  noise right sigma {sig}:', gain(bi, bn, bn, sig))
for sig in (0.20, 0.25, 0.30): print(f'  noise wrong (truth centred on b21) sigma {sig}:', gain(bi, bn, b21[bi], sig))
if 'write' in sys.argv[1:]:
    json.dump({str(i): dict(value=ff[i]['value'], why=ff[i]['why']) for i in ids}, open(f'{R}/cache/tmp/ag_r3_ffbd_patch_test.json', 'w'), indent=1)
    json.dump({str(i): bv[i] for i in bi}, open(f'{R}/cache/tmp/ag_r3_simconv_birdval_patch_test.json', 'w'), indent=1)
    pd.DataFrame([dict(id0=i, cat=A['cat'][i], video=A.video[i], b21=b21[i], new=ff[i]['value'], block='ffbd') for i in ids] +
                 [dict(id0=i, cat=A['cat'][i], video=A.video[i], b21=b21[i], new=bv[i]['value'], block='birdval') for i in bi]).to_csv(f'{R}/analysis/v3/ag/r3_simconv/block_rows.csv', index=False)
    print('written')
