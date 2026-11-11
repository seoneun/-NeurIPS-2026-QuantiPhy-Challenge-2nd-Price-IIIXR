"""rul113: c45foc52 (border-cut moving-prior focal correction, two label-free witnesses fr = f_MoGe/f_cal and
rho = median own-key geo/CaW) re-run unchanged on the current build, plus a part-2 sibling-witnessed branch for the gated
video's CaW-routed size rows.

Part 1 (= c45foc52, unchanged): video gate (simulation D3, moving non-gravity prior, prior track border-clipped >= 0.5 of
the fit window per rul001 diag, fr and rho both >= 1.4 or both <= 1/1.4 with >= 2 own-key 3d_calib size rows);
size/dist2 rows: GM sources / sqrt(fr), GO sources / fr; speed rows: v' = sqrt(v_rad^2 + (lat * k_edge / fr)^2), re-fused.

Part 2 (new): in a gated video, a size row in mode 3d_calib whose source is a CaW route (d3size_caw, caw_*, dis_caw*) takes
geo / fr when at least one sibling row confirms that value:
  - another base video (_segmented copies count as the same video) of the same recording event:
    identical prior_text and >= 2 shared normalised question texts, OR a K-twin (identical static target depth-key set);
  - same normalised question text;
  - sibling final answer is geo-led: source dis_geo / agree_* / patch:rul074, or mode 3d_personf with answer == geo;
  - sibling final answer within 1.2x of geo / fr.
Patched (patch:/kin) rows are untouched. No literal ids; labels are not used.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul113/rul113.py [build]
writes cache/tmp/ag_c65foc52x_patch_{test,val}.json"""
import sys, os, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
BUILD = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz15'
TH, CLIP, SIBTOL = 1.4, 0.5, 1.2
GM = {'d3size_dimgm', 'd3size_unstable_gm', 'dis_size_gm'}
GO = {'dis_geo', 'agree_geo_kind', 'geo_cawinvalid'}


def nq(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def fm(split, vid):
    p = f'{R}/geo/out/moge_{split}/{vid}.npz'
    if not os.path.exists(p): return np.nan
    z = np.load(p, allow_pickle=True)
    return float(np.nanmedian(np.array(z['fx'], float))) / float(z['W'])


def is_caw_route(src):
    return src == 'd3size_caw' or src.startswith('caw') or src.startswith('dis_caw')


def geo_led(r):
    src = str(r.answer_source)
    if src in ('dis_geo', 'patch:rul074') or src.startswith('agree_'): return True
    if str(r.mm).startswith('3d_personf') and np.isfinite(r.g) and r.g > 0 and abs(np.log(r.answer / r.g)) < np.log(1.02):
        return True
    return False


# ---- load both splits (siblings are searched over test + val inputs; labels never read)
ALL, PP = {}, {}
for split in ('test', 'val'):
    P = {}
    for ln in open(f'{R}/geo/out/parse_{split}_v2.jsonl', encoding='utf-8'):
        j = json.loads(ln); P[j['id']] = j
    det = pd.read_csv(f'{R}/submissions/{BUILD}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    det = det.drop(columns=[c for c in ('y', 'mra', 'm') if c in det.columns])
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    for c, cc in [('meas_f', 'f'), ('meas_mode', 'mm'), ('vlm_value', 'vlm'), ('meas_kind', 'kd'), ('geo_value', 'gg'), ('question', 'q')]:
        det[cc] = raw.reindex(det.index)[c]
    det['split'] = split
    det['base'] = det.video_id.astype(str).str.replace('_segmented', '', regex=False)
    det['nq'] = det.q.map(nq)
    det['pt'] = [str(P[int(i)].get('prior_text', '')).strip() for i in det.index]
    tk = []
    for i in det.index:
        p = P[int(i)]; ks = p.get('target_dkeys') or []
        st = {x['key']: x['d'] for x in (p.get('depth') or []) if x.get('t') is None}
        tk.append(tuple(sorted((k, round(float(st[k]), 3)) for k in ks if k in st)) if ks and all(k in st for k in ks) else ())
    det['tkey'] = tk
    ALL[split] = det; PP[split] = P
U = pd.concat([d.assign(rid=d.index) for d in ALL.values()], ignore_index=True)
QS = U.groupby('base').nq.apply(set).to_dict()
PTS = U.groupby('base').pt.first().to_dict()


def siblings(r):
    """same-event other-camera rows asking the same normalised question"""
    Z = U[(U.nq == r.nq) & (U.base != r.base) & U.base.str.startswith('simulation')]
    keep = []
    for _, s in Z.iterrows():
        ev = (PTS.get(s.base) == r.pt and len(QS.get(s.base, set()) & QS.get(r.base, set())) >= 2)
        kt = bool(r.tkey) and s.tkey == r.tkey
        if ev or kt: keep.append(s)
    return keep


for split in ('test', 'val'):
    det, P = ALL[split], PP[split]
    ag = pd.read_csv(f'{R}/cache/tmp/ag_accguard_diag_{split}.csv')
    te_p = f'{R}/cache/tmp/ag_c20tedge3d_diag_{split}.csv'
    te = pd.read_csv(te_p).set_index('id') if os.path.exists(te_p) else pd.DataFrame()
    patch = {}
    D = det[(det['cat'] == 'D3') & det.video_id.astype(str).str.startswith('simulation')]
    for v, s in D.groupby('video_id'):
        cal = s[s.mm.astype(str) == '3d_calib']
        if len(cal) == 0: continue
        pr = (P[int(cal.index[0])].get('priors') or [{}])[0]
        if pr.get('kind') in (None, 'size') or pr.get('gravity'): continue
        a = ag[(ag.vid == v) & (ag['mode'].astype(str) == '3d_calib')]
        clip = float(a['clip'].max()) if len(a) else np.nan
        if not (clip >= CLIP): continue
        fc = float(cal.f.median()); fmo = fm(split, v)
        if not (fmo > 0 and fc > 0): continue
        fr = fmo / fc
        own = cal[(cal.kd == 'size') & (cal.gg > 0) & (cal.vlm > 0)]
        if len(own) < 2: continue
        rho = float(np.exp(np.median(np.log(own.gg / own.vlm))))
        if not ((fr >= TH and rho >= TH) or (fr <= 1 / TH and rho <= 1 / TH)): continue
        print(split, 'VID', v, 'f_cal', round(fc, 3), 'f_moge', round(fmo, 3), 'fr', round(fr, 3), 'rho', round(rho, 3), 'clip', clip)
        head = f'border-cut moving prior (clip {clip:.2f}); focal off by fr={fr:.2f} (MoGe), own geo/CaW {rho:.2f} agrees'
        for i, r in cal.iterrows():
            src = str(r.answer_source)
            if src.startswith('patch') or src.startswith('kin'): continue
            nv, how = None, ''
            if r.kd in ('size', 'dist2'):
                if src in GM: nv, how = r.answer / np.sqrt(fr), 'gm answer / sqrt(fr)'
                elif src in GO: nv, how = r.answer / fr, 'geo / fr'
                elif r.kd == 'size' and is_caw_route(src) and np.isfinite(r.gg) and r.gg > 0:
                    gfr = r.gg / fr
                    sb = siblings(r)
                    wit = [s_ for s_ in sb if geo_led(s_) and max(s_.answer, gfr) / min(s_.answer, gfr) <= SIBTOL]
                    print('   part2 cand', i, str(r.q)[:50], src, round(r.answer, 3), 'geo/fr', round(gfr, 3), 'sibs',
                          [(s_.split, s_.rid, s_.video_id, s_.answer_source, round(s_.answer, 3)) for s_ in sb], 'wit', len(wit))
                    if wit:
                        nv = gfr
                        how = 'part2: CaW route, geo / fr confirmed by same-event sibling(s) ' + ', '.join(
                            f'{w.video_id} {w.answer:.3g} ({w.answer_source})' for w in wit)
            elif r.kd in ('speed', 'avgspeed') and len(te) and i in te.index and np.isfinite(r.gg) and r.gg > 0:
                t = te.loc[i]
                vr = float(t.v_rad) if np.isfinite(t.v_rad) else 0.0
                k = float(t.new_px) / float(t.old_px) if (float(t.ncut) > 0 and t.old_px > 0 and t.new_px > 0) else 1.0
                lat = np.sqrt(max(r.gg ** 2 - vr ** 2, 0.0))
                v2 = float(np.sqrt(vr ** 2 + (lat * k / fr) ** 2))
                c = float(r.cs)
                if src in ('agree_gm', 'agree_wgm') and c > 0 and max(v2, c) / min(v2, c) <= 2: nv = np.sqrt(v2 * c)
                elif src in GO or src in ('agree_gm', 'agree_wgm'): nv = v2
                how = f"motion: v_rad {vr:.3g}, lateral x k_edge {k:.2f} / fr -> geo' {v2:.4g}, {src} re-fused"
            if nv is None or abs(np.log(nv / r.answer)) < np.log(1.02): continue
            patch[str(i)] = dict(value=float(nv), why=f'rul113: {head}; {how}; was {r.answer:.4g}')
            print('  ', i, str(r.q)[:60], src, round(r.answer, 3), '->', round(nv, 3))
    json.dump(patch, open(f'{R}/cache/tmp/ag_c65foc52x_patch_{split}.json', 'w'), indent=1)
    print(split, 'rul113 rows', len(patch))
