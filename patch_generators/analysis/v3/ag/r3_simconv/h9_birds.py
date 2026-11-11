"""Bird family simulation_0088-0091 (one Blender scene; stated speeds big 17.98 m/s (0088 prior), small 11 m/s (0090/0091
priors) equal the 0089 validation labels 17.98 / 11.0). Our own tracks: per clip, pixel speed of each bird (robust linear
fit of the mask centroid over the clip), scale anchors s_b = 17.98 / v_px_big, s_s = 11 / v_px_small, scale = sqrt(s_b s_s),
pixel centroid distance at 0.3 / 0.5 s -> metres. Validation check: 0089 horizontal distance at 0.5 s (label 14.56)."""
import json, os, sys
import numpy as np, pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
ST = {'big': 17.98, 'small': 11.0}
out = []
for v in ('simulation_0088', 'simulation_0089', 'simulation_0090', 'simulation_0091'):
    for sp in ('test', 'val'):
        for ts in ('v3ntmv', 'v2sc'):
            p = f'{R}/geo/out/tracks_{sp}_{ts}/{v}.json'
            if not os.path.exists(p): continue
            d = json.load(open(p)); fps, st = d['fps'], d['stride']
            objs = {}
            for q in d['queries']:
                ph = q['phrase'].lower(); o = q.get('obj')
                if o is None: continue
                k = 'big' if 'big' in ph else ('small' if 'small' in ph else None)
                if k and k not in objs: objs[k] = o
            if len(objs) < 2: print(v, sp, ts, 'queries', [(q['phrase'], q.get('obj')) for q in d['queries']]); continue
            C, B = {}, {}
            for k, o in objs.items():
                fr = d['objs'][str(o)]
                C[k] = np.array([f['st']['c'] if f.get('st') else [np.nan, np.nan] for f in fr], float)
                B[k] = np.array([f['st']['box'] if f.get('st') else [np.nan] * 4 for f in fr], float)
            n = min(len(C['big']), len(C['small'])); t = np.arange(n) * st / fps
            vpx = {}
            for k in C:
                ok = np.isfinite(C[k][:n, 0]); A_ = np.c_[np.ones(ok.sum()), t[ok]]
                cx = np.linalg.lstsq(A_, C[k][:n][ok, 0], rcond=None)[0][1]; cy = np.linalg.lstsq(A_, C[k][:n][ok, 1], rcond=None)[0][1]
                vpx[k] = float(np.hypot(cx, cy))
            sb, ss = ST['big'] / vpx['big'], ST['small'] / vpx['small']; sc = np.sqrt(sb * ss)
            def dist(tt):
                i = int(round(tt * fps / st)); return float(np.hypot(*(C['big'][i] - C['small'][i]))) if i < n else np.nan
            row = dict(video=v, split=sp, ts=ts, n=n, cover_big=int(np.isfinite(C['big'][:n, 0]).sum()), cover_small=int(np.isfinite(C['small'][:n, 0]).sum()),
                       vpx_big=round(vpx['big'], 1), vpx_small=round(vpx['small'], 1), anchor_ratio=round(sb / ss, 3), scale=sc,
                       d03_px=dist(0.3), d05_px=dist(0.5), d03_m=dist(0.3) * sc, d05_m=dist(0.5) * sc,
                       len_big_px=float(np.nanmedian(np.maximum(B['big'][:n, 2] - B['big'][:n, 0], B['big'][:n, 3] - B['big'][:n, 1]))))
            out.append(row); break
        if out and out[-1]['video'] == v: break
D = pd.DataFrame(out); pd.set_option('display.width', 250)
print(D.round(4).to_string())
D.to_csv(f'{R}/analysis/v3/ag/r3_simconv/h9_birds_meas.csv', index=False)
V = pd.read_csv(f'{R}/analysis/v3/calib/audit_val.csv')
print(V[V.video.str.startswith('simulation_0089')][['id0', 'question', 'y', 'g', 'cs']].to_string())
