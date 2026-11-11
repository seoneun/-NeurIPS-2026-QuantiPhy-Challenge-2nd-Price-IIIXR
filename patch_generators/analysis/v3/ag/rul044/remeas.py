"""rul044 step 2: re-measure the candidate rows (winparse.py) with geo/measure_v3.measure_video imported read-only, in the
A1/twin chain environment (analysis/v3/ag/twin/twin_rule.py: rj5sQ env, detector tracks v2sc -> v2 -> v1, MoGe, scene ruler,
person focal).  The window-fixed parse goes through MV.load_split_tracks exactly like the pipeline parse (MV.open redirected
for the parse path only).  Every row of a touched video is measured with the current parse (base) and the fixed one (new),
on the row's own track.
usage: python remeas.py test|val -> cache/tmp/ag_c30win_meas_{split}.csv"""
import os, sys, json, copy, builtins
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
ENV = dict(QP_DEPTH_DEDUP='1', QP_SCENE_SA2VA='1', QP_SECOND_FIX='1', QP_SAMECLASS='1', QP_SCENE_RULER='1', QP_FF_ROT='1',
           QP_SIDE_FIX='1', QP_FLOOR='1', QP_STATIC_CAP='25', QP_FF_SUB='1', QP_FF_SUB_SRC='simulation', QP_PARSE2='1',
           QP_NEAREST='1', QP_BELT='1', RL_SELF_GRAV='1', RL_SELF_GRAV_MODE='2d_gself', QP_FPERSON='1', QP_FPERSON_MINN='2',
           QP_REJ_LO='0.2', QP_REJ_HI='5', QP_PRIOR_DRIFT='1.5', QP_PRIOR_ACCGUARD='3', QP_SCENE_HEAD='1', QP_TSHIFT='1')
os.environ.update(ENV)
sys.path.insert(0, R + '/geo')
import numpy as np, pandas as pd
import measure_v3 as MV
import measure as M1
from rules import UNIT_SI
split = sys.argv[1] if len(sys.argv) > 1 else 'test'
OD = f'{R}/analysis/v3/ag/rul044/out'
NEW = f'{OD}/parse_{split}_v2_win.jsonl'
CUR = f'{R}/geo/out/parse_{split}_v2.jsonl'


def eff(r):
    q = r.get('q') or {}
    return dict(kind=q.get('kind'), unit=q.get('unit'), t=q.get('t'), t1=q.get('t1'), t2=q.get('t2'), when=q.get('when'), dim=q.get('dim'),
                part=q.get('part'), target=(r.get('target') or {}).get('phrase'), where=tuple((r.get('target') or {}).get('where') or []),
                target2=(r.get('target2') or {}).get('phrase'), prior_obj=(r.get('prior_obj') or {}).get('phrase'),
                tdk=tuple(r.get('target_dkeys') or []), pdk=tuple(r.get('prior_dkeys') or []),
                depth=tuple((x['key'], x['t'], x['d']) for x in (r.get('depth') or [])),
                priors=tuple((p.get('value_si'), p.get('obj'), p.get('kind'), p.get('t'), p.get('t1'), p.get('t2')) for p in (r.get('priors') or [])),
                fixes=(r.get('side_fix'), r.get('floor_fix'), r.get('parse2_fix')))


def main():
    cand = {int(k) for k in json.load(open(f'{OD}/cand_{split}.json'))}
    out_p = f'{R}/cache/tmp/ag_c30win_meas_{split}.csv'
    if not cand:
        pd.DataFrame([]).to_csv(out_p, index=False)
        print(f'[{split}] no candidate rows'); return
    MV.TRACK_VER = 'v2sc'; MV.FIX = set('abcdef'); MV.HALF_V, MV.HALF_A = 0.25, 0.4
    MV.FOCAL_MIX = ''; MV.PARSE_TAG = ''; MV.AGG3D_P = None; MV.STATIC_FRAC = 0.25
    MV.MEASURE_SCENE = False; MV.WINGED_LEN_SHORT = False; MV.MOGE_OPTS = {'fbrej'}
    fn_ = f'{R}/geo/out/scene_ruler_{split}.json'
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get('ok')})
    fn_ = f'{R}/geo/out/person_focal_{split}.json'
    if os.path.exists(fn_):
        MV.FPERSON.update(json.load(open(fn_)))
    M1.SCALE_DEPTH = True
    cfg = 'e'
    tp = f'{R}/geo/out/owl_threshold_v2.json'
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))['T_owl']
    byv, vjs = MV.load_split_tracks(split, cfg)
    MV.open = lambda p, *a, **k: builtins.open(NEW if str(p) == CUR else p, *a, **k)
    byv2, _ = MV.load_split_tracks(split, cfg)
    del MV.open
    chg = {}
    for v, rs in byv.items():
        m2 = {r['id']: r for r in byv2[v]}
        for r in rs:
            a, b = eff(r), eff(m2[r['id']])
            d = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
            if d:
                chg[r['id']] = (v, d)
    print(f'[{split}] effective-parse changes: {sorted(chg)}; candidates {sorted(cand)}', flush=True)
    for i, (v, d) in sorted(chg.items()):
        print('  ', i, v, d, '' if i in cand else '(not a candidate: ignored)')
    chg = {i: x for i, x in chg.items() if i in cand and set(x[1]) <= {'t1', 't2'}}
    fmed = json.load(open(f'{R}/geo/out/focal_medians_test.json'))['f_over_W']
    allf = float(np.median(list(fmed.values())))
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    vids = sorted({v for v, _ in chg.values()})
    for v in vids:
        p = f'{R}/geo/out/moge_{split}/{v}.npz'
        if os.path.exists(p):
            z = np.load(p)
            MV.MOGE[v] = dict(f=float(np.median(z['fx'])), J=[int(j) for j in z['j']], depth=z['depth'], W=int(z['W']))
    out = []
    for v in vids:
        rs = byv[v]; vj = vjs.get(v)
        if vj is None:
            for i in [i for i, (vv, _) in chg.items() if vv == v]:
                out.append(dict(id=i, video_id=v, ok=False, why='no_track_file'))
            continue
        dur = (max(vj['fidx']) + 1) / float(vj['fps'])
        fdd = {r['video_source']: fmed.get(r['video_source'], allf) * vj['W'] for r in rs}
        o0, _ = MV.measure_video(copy.deepcopy(rs), vj, fdd, cfg)
        m2 = {r['id']: r for r in byv2[v]}
        rs2 = [copy.deepcopy(m2[r['id']]) if r['id'] in chg else copy.deepcopy(r) for r in rs]
        o1, V1 = MV.measure_video(rs2, vj, fdd, cfg)
        base = {x['id']: x for x in o0}; new = {x['id']: x for x in o1}
        print(f'\n== {v} dur {dur:.2f}s version {vj.get("version")} keys {sorted(V1.tr)}')
        for r in rs:
            i = r['id']; u = r['q']['unit']; u2 = m2[i]['q']['unit']
            b0, n1 = base[i], new[i]
            vb = b0['val_si'] / UNIT_SI[u] if b0.get('ok') else np.nan
            vn = n1['val_si'] / UNIT_SI[u2] if n1.get('ok') else np.nan
            tag = 'CHG ' if i in chg else '    '
            print(f"{tag}{i} raw geo {raw.geo_value.get(i)!s:.8} meas {raw.meas_value.get(i)!s:.8} ({raw.meas_mode.get(i)}) | base {vb:.5g} {b0.get('mode')} "
                  f"{b0.get('why')} | new {vn:.5g} {n1.get('mode')} {n1.get('why')} | {r['question'][:80]}")
            if i in chg:
                rec = dict(id=i, video_id=v, base=vb, base_mode=b0.get('mode'), new=vn, mode=n1.get('mode'), why=n1.get('why'), f=n1.get('f'),
                           ok=bool(n1.get('ok')), dur=round(dur, 3), unit=u2, t1=m2[i]['q'].get('t1'), t2=m2[i]['q'].get('t2'))
                if rec['ok']:
                    k = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(u2, 2 if "^2" in u2 else 1)
                    lo, hi = MV.PLAUS[k]
                    rec['plaus'] = bool(lo <= vn * UNIT_SI[u2] <= hi)
                out.append(rec)
    pd.DataFrame(out).to_csv(out_p, index=False)


if __name__ == '__main__':
    main()
