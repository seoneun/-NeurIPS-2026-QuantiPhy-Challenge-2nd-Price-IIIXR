import json, os, sys
import numpy as np, pandas as pd
H1, W = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
QT = 'analysis/v3/compliance/qtrack/'
def mra(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    with np.errstate(all='ignore'): e = np.abs(p - y) / np.abs(y)
    th = np.arange(0.5, 0.96, 0.05)
    return np.where(np.isfinite(e), (e[:, None] < (1 - th)[None, :]).mean(1), 0.0)
def num(s): return pd.to_numeric(s, errors='coerce')
def grp(P): return pd.Series(np.where(P.video_id.str.contains('_segmented'), 'segmentation', np.where(P.video_source == 'captured', 'lab', P.video_source)), index=P.index)
hr = pd.read_csv(f'{H1}/h1_rows.csv').set_index('id0'); rs = pd.read_csv(f'{H1}/rows_status.csv').set_index('id0')
P = pd.read_csv('preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
Pu = pd.read_csv('preds/test_geo_v4ntrj5sQ2usrngsr_d_vw_raw.csv').set_index('Unnamed: 0')
G = pd.read_csv('analysis/v3/ag/r10/cl/clean/geo/geo_GvP1/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
C = json.load(open(QT + 'qdir/prep_test_ctrl.json')); Q = json.load(open(QT + 'qdir/prep_test_qv2.json'))
D = pd.DataFrame(index=P.index)
D['gp'] = num(P.geo_value); D['gpu'] = num(Pu.geo_value); D['gg'] = num(G.geo_value).reindex(D.index)
D['mode'] = P.meas_mode.astype(str); D['mode_u'] = Pu.meas_mode.astype(str); D['ms'] = P.merge_src.astype(str); D['ms_u'] = Pu.merge_src.astype(str)
mech = {i: [] for i in D.index}
for i in D.index[D['mode'].str.contains('fovdefault') | D['mode_u'].str.contains('fovdefault')]:
    mech[i].append('FOCALPOOL')
for k, v in C['rl'].items():
    i = int(k)
    if i not in D.index: continue
    if not (D.at[i, 'ms'] in ('ruler', 'selfruler') or D.at[i, 'ms_u'] in ('ruler', 'selfruler')): continue
    q = Q['rl'].get(k, {}); own = {r['key'] for r in q.get('rulers', [])} | {d[0] for d in q.get('dropped', [])} | {v.get('tkey'), q.get('tkey')}
    foreign = [r['key'] for r in v['rulers'] if r['key'] not in own]
    mech[i].append('RULER_FOREIGN:' + '/'.join(foreign) if foreign else 'RULER_OWNONLY_ok')
cx, qx = set(C['xg']), set(Q['xg'])
for k in cx - qx:
    i = int(k)
    if i in D.index and ('xg_v' in (D.at[i, 'ms'], D.at[i, 'ms_u']) or D.at[i, 'mode'].startswith('vlm_')): mech[i].append('XG_JOINT_ONLY(took v)')
for k in qx - cx:
    i = int(k)
    if i in D.index: mech[i].append('XG_PERQ_ONLY(prod kept d)')
D['mech'] = pd.Series({i: ';'.join(m) for i, m in mech.items()})
ct = pd.read_csv(QT + 'out/compare_tracks_test_qv2_vs_v2_ctrl171.csv')
bad = ct[ct.same_obj == False].assign(ph=lambda d: d.key.str.split('|').str[0].str.lower())
badset = set(zip(bad.video, bad.ph))
for r in map(json.loads, open(HERE + '/../40_rul437/parse_test_v2_rl0.jsonl')):
    i = r['id']; t = (r['q'].get('target') or '').lower()
    if (r['video_id'], t) in badset and i in D.index:
        D.at[i, 'mech'] = (D.at[i, 'mech'] + ';' if D.at[i, 'mech'] else '') + 'TRACK_JOINT_OBJ'
D = D.join(hr, how='left'); D['source'] = rs.source.reindex(D.index)
D['cs'] = pd.read_csv('analysis/v3/ag/r10/work/b_m.csv').set_index('id').cs.reindex(D.index)
fm = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
Pv = pd.read_csv('preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
yv = num(pd.read_csv('data/val/validation_dataset.csv').set_index('Unnamed: 0').ground_truth_posterior).reindex(Pv.index)
gv = grp(Pv); mv = Pv.meas_mode.astype(str).str.replace('^vlm_', '', regex=True); fv = num(Pv.meas_f); gvv = num(Pv.geo_value)
f0 = {}
for g in ['lab', 'simulation', 'segmentation']:
    s = (gv == g) & mv.str.startswith('3d_') & ~mv.str.contains('personf|knownsize|samedepth') & fv.notna() & gvv.notna() & yv.notna()
    grid = np.round(np.arange(0.30, 2.01, 0.01), 2)
    if s.sum():
        sc = np.array([mra(gvv[s] * fv[s] / x, yv[s]).mean() for x in grid])
        f0[g] = float(np.median(grid[sc >= np.nanmax(sc) - 0.005]))
    else:
        f0[g] = np.nan
D['grp'] = grp(P)
mm = pd.read_csv('geo/out/meas_test_v2_v3m_rrj5sQ.csv').set_index('id'); mvv = pd.read_csv('geo/out/meas_test_v2_v3v_rntrj5sQ.csv').set_index('id')
vm = num(mm.value).where(mm.ok == True).reindex(D.index); vv = num(mvv.value).where(mvv.ok == True).reindex(D.index)
RL = [pd.read_csv(f'intermediates/preds/test_geo_rl{k}_raw.csv.gz').set_index('Unnamed: 0') for k in range(3)]
MOT = {"speed", "avgspeed", "hvel", "accel", "disp", "path", "maxheight", "orbitdiam"}
new = D.gp.copy()
for i in D.index[D.mech != '']:
    m = D.at[i, 'mech']; g = D.at[i, 'gp']
    if 'RULER_FOREIGN' in m:
        k = str(i); c = C['rl'][k]; q = Q['rl'].get(k, {}); own = {r['key'] for r in q.get('rulers', [])} | {d[0] for d in q.get('dropped', [])} | {c.get('tkey'), q.get('tkey')}
        vals = []; selfv = np.nan
        for j, r in enumerate(c['rulers'][:3]):
            x = RL[j].loc[i]; ok = x.answer_source == 'meas' and str(x.meas_mode).startswith('2d_ratio')
            v = float(num(pd.Series([x.geo_value]))[0]) if ok else np.nan
            if r['key'] in own and np.isfinite(v): vals.append(v)
            if r['key'] == c.get('tkey') and np.isfinite(v): selfv = v
        mov = Q.get('moving', {}).get(k, np.nan); mot = c['kind'] in MOT
        if len(vals) >= 2 and max(vals) / min(vals) <= 1.3 and (not mot or (mov is not None and mov >= 2)): g = float(np.exp(np.mean(np.log(vals)))) if len(vals) == 2 else float(np.median(vals))
        elif np.isfinite(selfv) and mot and mov is not None and mov >= 2: g = selfv
        else: g = np.nan
    if 'XG_JOINT_ONLY' in m and D.at[i, 'ms'] == 'xg_v':
        g = vm[i] if np.isfinite(vm[i]) else vv[i]
    if 'XG_PERQ_ONLY' in m and D.at[i, 'ms'] == 'd' and np.isfinite(vv[i]):
        g = vv[i]
    if 'TRACK_JOINT_OBJ' in m:
        g = D.at[i, 'gg']
    if 'FOCALPOOL' in m and 'fovdefault' in D.at[i, 'mode'] and 'TRACK_JOINT_OBJ' not in m and np.isfinite(g):
        gr = D.at[i, 'grp']; g = g * fm[gr] / f0[gr]
    new[i] = g
D['new_geo'] = new
GT = {'dis_geo', 'geo_only_mode', 'agree_geo_kind', 'sruler_geo', 'geo_cawinvalid'}; GM = {'agree_gm', 'dis_size_gm', 'rul263_dimgm', 'rul263_unstable_gm'}
rows = []
for i in D.index[(D.mech != '') & D.p17.notna()]:
    r = D.loc[i]; old = r.gp; h = r.h1; pv = h
    if r.status == 'S' or np.isclose(h, r.gp, rtol=1e-6):
        if np.isclose(h, r.gp, rtol=1e-6) or np.isclose(h, r.gpu, rtol=1e-6):
            old = r.gp if np.isclose(h, r.gp, rtol=1e-6) else r.gpu; w = 1.0
        elif r.source in GT: w = 1.0
        elif r.source in GM: w = 0.5
        elif r.source == 'agree_wgm':
            with np.errstate(all='ignore'): w = np.log(h / r.cs) / np.log(r.gp / r.cs) if abs(np.log(r.gp / r.cs)) > 0.02 else 0.5
            w = float(np.clip(w, 0, 1)) if np.isfinite(w) else 0.5
        else: w = None
        ng = r.new_geo
        if w is not None and not np.isclose(ng, old, rtol=1e-6):
            pv = h * (ng / old) ** w if (np.isfinite(ng) and np.isfinite(old) and old > 0 and ng > 0) else r.z1s2c
    rows.append(dict(id0=i, mechanism=r.mech, proposed_value=pv, h1=r.h1))
O = pd.DataFrame(rows)
O['changed'] = ~np.isclose(O.proposed_value, O.h1, rtol=1e-6)
O[['id0', 'mechanism', 'proposed_value', 'changed']].to_csv(W + '/rows.csv', index=False)
X = O[O.changed]
pd.DataFrame({'id0': X.id0, 'value': X.proposed_value, 'rule': 'aud_geo:' + X.mechanism}).to_csv(W + '/03_aud_geo.csv', index=False, lineterminator='\n')
