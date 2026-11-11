import sys, os, json
import numpy as np, pandas as pd
Q = os.path.abspath(os.environ["RA_Q"])
WO = os.path.abspath(os.environ["RA_OUT"])
BUILD = sys.argv[1]
TH, CLIP = 1.4, 0.5
GM = {'rul263_dimgm', 'rul263_unstable_gm', 'dis_size_gm'}
GO = {'dis_geo', 'agree_geo_kind', 'geo_cawinvalid'}
def fm(split, vid):
    p = f'{Q}/geo/out/moge_{split}/{vid}.npz'
    if not os.path.exists(p): return np.nan
    z = np.load(p, allow_pickle=True)
    return float(np.nanmedian(np.array(z['fx'], float))) / float(z['W'])
for split in ('test',):
    P = {}
    for ln in open(f'{Q}/geo/out/parse_{split}_v2.jsonl', encoding='utf-8'):
        j = json.loads(ln); P[j['id']] = j
    det = pd.read_csv(f'{Q}/submissions/{BUILD}_detail.csv', index_col=0)
    raw = pd.read_csv(f'{Q}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    for c, cc in [('meas_f', 'f'), ('meas_mode', 'mm'), ('vlm_value', 'vlm'), ('meas_kind', 'kd'), ('geo_value', 'gg'), ('question', 'q')]:
        det[cc] = raw.reindex(det.index)[c]
    ag = pd.read_csv(f'{Q}/cache/tmp/ag_rul001_diag_{split}.csv').drop_duplicates('id').set_index('id')
    patch = {}
    D = det[(det['cat'] == 'D3') & det.video_id.astype(str).str.startswith('simulation') & (det.mm.astype(str) == '3d_calib')]
    for i, r in D.iterrows():
        pr = (P[int(i)].get('priors') or [{}])[0]
        if pr.get('kind') in (None, 'size') or pr.get('gravity'): continue
        if i not in ag.index or not (float(ag.loc[i, 'clip']) >= CLIP): continue
        fmo = fm(split, r.video_id); fc = float(r.f)
        if not (fmo > 0 and fc > 0): continue
        fr = fmo / fc
        if not (r.gg > 0 and r.vlm > 0): continue
        rho = float(r.gg / r.vlm)
        if not ((fr >= TH and rho >= TH) or (fr <= 1 / TH and rho <= 1 / TH)): continue
        src = str(r.answer_source)
        if src.startswith('patch') or src.startswith('kin'): continue
        nv, how = None, ''
        if r.kd in ('size', 'dist2'):
            if src in GM: nv, how = r.answer / np.sqrt(fr), 'gm answer / sqrt(fr)'
            elif src in GO: nv, how = r.answer / fr, 'geo / fr'
        print(split, i, r.video_id, r.kd, src, 'fr %.3f rho_own %.3f clip %.2f' % (fr, rho, float(ag.loc[i, 'clip'])), 'ans %.4g' % r.answer, '->', nv)
        if nv is None or abs(np.log(nv / r.answer)) < np.log(1.02): continue
        patch[str(i)] = dict(value=float(nv), why=f'c65own: own fr={fr:.2f} own geo/CaW {rho:.2f}; {how}; was {r.answer:.4g}')
    json.dump(patch, open(f"{WO}/c65own_{split}.json", "w"), indent=1)
    print(split, 'rows', len(patch), sorted(patch))
