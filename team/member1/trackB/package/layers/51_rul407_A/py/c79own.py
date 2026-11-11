import os, sys, json
os.environ['QP_ROOT'] = os.path.abspath(os.environ["RA_Q"])
sys.path.insert(0, os.path.abspath(os.environ["RA_MODS"] + "/rul151"))
import numpy as np, pandas as pd
import rule as C
R = C.R; WO = os.path.abspath(os.environ["RA_OUT"])
SUB = sys.argv[1]
for split in ('test',):
    det = pd.read_csv(f'{R}/submissions/{SUB}_detail.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_{C.RAW}_raw.csv', index_col=0)
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    out = {}
    for r in det.index:
        if r not in P or r not in raw.index: continue
        v = det.loc[r, 'video_id']
        if not str(v).startswith('simulation'): continue
        if raw.loc[r, 'meas_mode'] not in ('2d_ratio', 'vlm_2d_ratio'): continue
        pr = (P[r].get('priors') or [None])[0]
        if not pr or pr.get('kind') != 'size' or pr.get('dim') != 'width': continue
        ph = (P[r].get('prior_obj') or {}).get('phrase'); ver = raw.loc[r, 'meas_tracks']
        if not ph: continue
        q = P[r].get('q') or {}
        if q.get('kind') == 'size' and q.get('dim') == 'height' and (P[r].get('target') or {}).get('phrase') == ph: continue
        S = float(pr['value_si'])
        j = C.load_tracks(split, ver, v); st = C.track_states(j, ph) if j else None
        if not st or len(st) < 5: continue
        b = np.array([s['box'] for s in st], float)
        bw, bh = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1])
        rs = float(np.median([s['rr'][1] for s in st]))
        bord = float(np.mean([s.get('border', 0) != 0 for s in st]))
        mpp = pd.to_numeric(raw.loc[r, 'meas_mpp'], errors='coerce')
        if not (pd.notna(mpp) and mpp > 0): continue
        pxr = S / float(mpp)
        g2 = bw > 1.3 * bh and abs(rs / bh - 1) < 0.08; g4 = bord <= 0.2; g3 = abs(pxr / rs - 1) <= 0.08
        if not (g2 and g4): continue
        print(split, r, v, ph, 'bw %.0f bh %.0f rs %.1f pxr %.1f g3 %s' % (bw, bh, rs, pxr, g3))
        if not g3: continue
        src = str(det.loc[r, 'answer_source'])
        if not (src in ('dis_geo', 'geo') or src.startswith('agree')): continue
        g = det.loc[r, 'g']
        if not (pd.notna(g) and g > 0): continue
        k = rs / bw; g2_ = float(g) * k; cs = det.loc[r, 'cs']
        if pd.notna(cs) and cs > 0 and max(g2_ / cs, cs / g2_) <= 2.0: val, why = float(np.sqrt(g2_ * cs)), f'c79own: own ruler {pxr:.1f}px ~ rs {rs:.1f}; geo {g:.4g}*k{k:.3f}={g2_:.4g} agrees with CaW {cs:.4g} -> gm'
        else: val, why = g2_, f'c79own: own ruler {pxr:.1f}px ~ rs {rs:.1f} (bw {bw:.0f}); geo {g:.4g}*k{k:.3f}'
        out[str(r)] = dict(value=val, why=why)
        print('  CHANGE', split, r, det.loc[r, 'cat'], src, round(float(det.loc[r, 'answer']), 4), '->', round(val, 4))
    json.dump(out, open(f'{WO}/c79own_{split}.json', 'w'), indent=1)
    print(split, 'rows', len(out), sorted(out))
