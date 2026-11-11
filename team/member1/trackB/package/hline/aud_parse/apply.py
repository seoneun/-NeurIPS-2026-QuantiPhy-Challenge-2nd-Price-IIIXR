import csv, json, sys, os, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_fix as F
R = os.getcwd()
H57, H1P, OUT = sys.argv[1:4]


def rows(f): return list(csv.DictReader(open(f, newline='', encoding='utf-8')))


SUB = rows(H57)
H1 = {int(r['id0']): r for r in rows(H1P)}

G = pd.read_csv(R + '/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
PT = {}
for l in open(R + '/geo/out/parse_test_v2.jsonl'):
    d = json.loads(l)
    PT[d['id']] = d
USI = {'m': 1, 'cm': .01, 'mm': .001, 'm/s': 1, 'cm/s': .01, 'm/s^2': 1, 'cm/s^2': .01}


def fl(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def same(a, b):
    return abs(a - b) <= 1e-9 * max(1, abs(b))


def track_obj(vid, phrase):
    f = R + '/geo/out/tracks_test_v2sc/' + vid + '.json'
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    for k, q in enumerate(j.get('queries', [])):
        if q.get('phrase', '').lower() == (phrase or '').lower():
            seq = j['objs'].get(str(q.get('obj', k)))
            if seq is None:
                return None
            c = np.array([s['st']['c'] if (s and s.get('st')) else [np.nan, np.nan] for s in seq], float)
            return np.array(j['fidx']), c
    return None


def remeasure_2d(i, r, q, t1, t2):
    g = G.loc[i]
    if str(g['meas_mode']) != '2d_ratio' or not (fl(g['meas_mpp']) > 0):
        return None, 'no 2d_ratio scale'
    if q['kind'] != 'disp':
        return None, 'kind %s not re-measured' % q['kind']
    tr = track_obj(r['video_id'], (PT[i].get('target') or {}).get('phrase') or q['target'])
    if tr is None:
        return None, 'no track'
    fidx, c = tr
    fps = float(r['fps'])

    def at(t):
        f = int(round(fps * t))
        ok = np.where(~np.isnan(c[:, 0]) & (np.abs(fidx - f) <= 1))[0]
        if len(ok) == 0:
            return None
        return c[ok[np.argmin(np.abs(fidx[ok] - f))]]
    a, b = at(t1), at(t2)
    if a is None or b is None:
        return None, 'frame not tracked'
    return float(np.linalg.norm(b - a)) * fl(g['meas_mpp']) / USI[q['unit']], 'ok'


cands = []
for i, r in enumerate(SUB):
    if i not in H1:
        continue
    h = H1[i]
    q = PT[i]['q']
    v, p17, z1 = fl(r['parsed_value']), fl(h['p17']), fl(h['z1s2c'])
    from_p17 = same(v, p17)
    ft = F.fix_times(r['question'])
    if ft and (q['t'], q['t1'], q['t2']) != (ft['t'], ft['t1'], ft['t2']) and not from_p17 and ft['t1'] is not None and ft['t2'] is not None:
        g_new, why = remeasure_2d(i, r, q, ft['t1'], ft['t2'])
        cw = fl(G.loc[i]['vlm_value'])
        if g_new:
            agree = cw > 0 and max(g_new, cw) / min(g_new, cw) <= 2
            cands.append((i, math.sqrt(g_new * cw) if agree else g_new))
    if F.event_velocity(r['question']):
        te = F.event_time(r['depth_info'])
        if te is not None:
            tgt = set(((PT[i].get('target') or {}).get('phrase') or '').split())
            dl = sorted((t, d) for t, k, d, n in F.depth_lines(r['depth_info']) if t is not None and set(k.split('_')) & (tgt | {'ball'}))
            prev = [x for x in dl if x[0] < te - 1e-9]
            at = [x for x in dl if abs(x[0] - te) < 1e-9]
            if prev and at:
                lb = abs(at[0][1] - prev[-1][1]) / (te - prev[-1][0]) / USI[q['unit']]
                if v < 0.9 * lb:
                    for c in (z1, fl(G.loc[i]['vlm_value'])):
                        if c >= 0.9 * lb:
                            cands.append((i, c))
                            break
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['id0', 'value'])
    w.writerows(cands)
