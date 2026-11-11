import json, os, re, sys
import numpy as np, pandas as pd, cv2

Z1S2C, OUT, PRED_T018 = sys.argv[1], sys.argv[2], sys.argv[3]
D = 'team/member3/pkg/lev108/inputs'
TV = 'geo/out/tracks_test_v35'
TR = 'analysis/v3/ag/retrack/tracks/test_rtMp'
RT = {2351: 'captured_0020__776ee9c62c', 2359: 'captured_0020s__776ee9c62c', 2367: 'captured_0020x__776ee9c62c', 2456: 'captured_0024x__f3667ad532'}
Q = pd.read_parquet('data/test/test_dataset.parquet')
T = {t: [] for t in ('tm19', 'h52m', 'h62m', 'h74D2', 'h74D3a', 'h74D3b', 'h78D2', 'sbfix')}

for t, i, v in pd.read_csv(f'{D}/read_values.csv').itertuples(index=False):
    T[t].append((i, v))


def values(path):
    s = pd.read_csv(path)
    return pd.Series(s.parsed_value.astype(float).values, index=s.id.values - 1)


z, c = values(Z1S2C), pd.read_csv(PRED_T018).set_index('id0').value.astype(float)
for i in map(int, open(f'{D}/gm_rows.txt').read().split()):
    T['h52m'].append((i, np.sqrt(z[i] * c[i])))

F3 = 3.0 * 1920
T['tm19'] += [(1057, np.hypot(2.684, 252 * (46.654 - 0.75 * 2.684) / F3)), (1061, 102 * 42.5 / F3), (1068, 731 * 55.767 / F3),
              (1529, np.mean([(9.6 + 11.2) / 2, (10.2 + 10.7) / 2]))]


def lab_size(path, Z):
    d = json.load(open(path))
    fr = next(x['frames'] for x in d['queries'] if 'target' in (x.get('roles') or []) and x.get('frames'))
    a = np.array([(f['st']['box'][2] - f['st']['box'][0], max(f['st']['rr'][0], f['st']['rr'][1]), min(f['st']['rr'][0], f['st']['rr'][1]))
                  for j, f in enumerate(fr) if f and f.get('st') and not f['st'].get('border') and j < len(d['fidx'])], float)
    f = 0.5959 * d['W']
    return [np.median(a[:, k]) * Z / f * 100 for k in range(3)]


S = {}
for i, obj in ((2351, 'jar'), (2359, 'jar'), (2367, 'jar'), (2456, 'book')):
    Z = float(np.median([float(x) for x in re.findall(rf'distance_{obj}_camera\s*=\s*([\d.]+)\s*m', Q.at[i, 'depth_info'].lower())]))
    S[i] = [lab_size(f'{TV}/{i}.json', Z), lab_size(f'{TR}/{RT[i]}.json', Z)]
for i in (2351, 2359, 2367):
    T['tm19'].append((i, np.mean([s[k] for s in S[i] for k in (0, 1)])))
T['tm19'].append((2456, np.mean([s[2] for s in S[2456]])))

vlat = np.mean(pd.read_csv(f'{D}/lateral_2362.csv').value.values)
T['tm19'].append((2362, np.mean([np.hypot(vlat, 1.9735 * (s0 / 24 - 1) / (15 / 24) * 100) for s0 in (34, 40)])))

cap = cv2.VideoCapture('data/test/simulation_0145_segmented.mp4'); fps = cap.get(cv2.CAP_PROP_FPS); F = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    F.append(fr)
m0 = cv2.dilate((F[0].min(axis=2) < 230).astype(np.uint8), np.ones((5, 5), np.uint8))
K, N, L = [], [], []
for i in range(6, 22):
    mk = (F[i].min(axis=2) < 230).astype(np.uint8) & (1 - m0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mk, 8); k = 1 + np.argmax(st[1:, 4])
    ys, xs = np.where(lab == k); P = np.c_[xs, ys].astype(float); m = P.mean(0)
    vt = np.linalg.svd(P - m, full_matrices=False)[2]; d = vt[0] if vt[0][0] < 0 else -vt[0]
    K.append(i); N.append(P[np.argmax((P - m) @ d)])
    if i >= 12:
        pr = (P - m) @ vt[0]; L.append(pr.max() - pr.min())
N = np.array(N); p = np.polyfit(np.array(K, float), np.hypot(*(N - N[0]).T), 2)
T['tm19'].append((1851, 2 * p[0] * fps ** 2 * (2.9 / np.median(L))))

M = pd.read_csv('team/member3/pkg/chain/inputs/meas/meas_test_v2_v3m_k9V35.csv').set_index('id').value
T['sbfix'] += [(i, M[i]) for i in (270, 2423)]

os.makedirs(OUT, exist_ok=True)
for t, rows in T.items():
    name = 'sbfix_copied_values' if t == 'sbfix' else f'{t}_test'
    pd.DataFrame(rows, columns=['id0', 'value']).sort_values('id0').to_csv(f'{OUT}/{name}.csv', index=False)
