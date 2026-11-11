import json, os, re, sys
import numpy as np, pandas as pd, cv2

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUTD = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
T['cur'] = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv')).parsed_value.values
assert (T.id0.values == np.arange(len(T))).all()


def vals(videos, qsub):
    s = T[T.video.isin(videos) & T.question.str.contains(qsub, regex=False)]
    return s.cur.astype(float).values


def prior_num(video, unit):
    txt = T[T.video == video].prior.iloc[0]
    m = re.search(r'=\s*([0-9]*\.?[0-9]+)\s*([A-Za-z/^0-9]+)', txt)
    assert m and m.group(2) == unit
    return float(m.group(1)), txt


def track(v):
    for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test'):
        p = os.path.join(R, f'geo/out/{d}/{v}.json')
        if os.path.exists(p):
            return json.load(open(p))


def boxes(v, phrase):
    js = track(v)
    q = next(q for q in js['queries'] if q['phrase'] == phrase)
    out = []
    for i, e in enumerate(js['objs'][str(q['obj'])]):
        st = e.get('st') if isinstance(e, dict) else None
        if st and st.get('box'):
            out.append([js['fidx'][i]] + list(st['box']))
    return np.array(out, float), js


def frames(v, idx):
    cap = cv2.VideoCapture(os.path.join(R, f'data/test_480p/{v}.mp4'))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for f in idx(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        if ok:
            out.append(im)
    return out


res, diag = {}, {}

t = T[T.id0 == 629].iloc[0]
src = T[(T.prior == t.prior) & (T.question == t.question) & (T['cat'] == t['cat']) & (T.video != t.video)]
assert sorted(src.video) == ['simulation_0210', 'simulation_0213', 'simulation_0214', 'simulation_0215']
mw = vals(sorted(src.video), 'width of the floor mirror')
res[629] = float(np.median(mw)); diag[629] = f'median of pred_t007 {np.round(mw, 4).tolist()} ({", ".join(sorted(src.video))})'

L1 = {}
for v in ['simulation_0233', 'simulation_0233_segmented']:
    spd, _ = prior_num(v, 'm/s')
    B, js = boxes(v, 'black cat'); W = js['W']; fps = js.get('fps', 24.0)
    ok = (B[:, 1] > 2) & (B[:, 3] < W - 2)
    tt = B[ok, 0] / fps; cx = (B[ok, 1] + B[ok, 3]) / 2
    sp = abs(np.polyfit(tt, cx, 1)[0]); m1 = spd / sp; w = np.median(B[ok, 3] - B[ok, 1])
    L1[v] = (w * m1, sp, w, int(ok.sum()), m1)
r1 = float(np.mean([x[0] for x in L1.values()]))
r2v = vals(['simulation_0231', 'simulation_0232'], 'length of the black cat'); r2 = float(np.mean(r2v))
res[757] = res[1965] = float(np.sqrt(r1 * r2))
diag[757] = diag[1965] = (f'route1 {r1:.5f} (' + '; '.join(f'{k[-14:]}: px speed {x[1]:.2f}, box w {x[2]:.1f}, n {x[3]}'
                                                        for k, x in L1.items()) + f') route2 {r2:.5f} {np.round(r2v, 4).tolist()}')

chalk = sorted(T[T.prior.str.startswith('length of the chalkboard')].video.unique())
assert chalk == ['simulation_0229', 'simulation_0230']
cl = {prior_num(v, 'm')[0] for v in chalk}; assert len(cl) == 1
CL = cl.pop()
CB, _ = boxes('simulation_0231', 'chalkboard'); cbw = np.median(CB[:, 3] - CB[:, 1]); cbh = np.median(CB[:, 4] - CB[:, 2])
wm = CL / cbw
res[735] = float(cbh * wm); diag[735] = 't83'
for i, v in ((746, 'simulation_0231'), (756, 'simulation_0232')):
    C, _ = boxes(v, 'clock'); d = np.median((C[:, 3] - C[:, 1] + C[:, 4] - C[:, 2]) / 2)
    res[i] = float(d * wm); diag[i] = 't85'

sys.path.insert(0, os.path.join(R, 'analysis/v3/ag/rul217'))
os.environ.setdefault('QP_ROOT', R)
import rul217 as CE
x0, x1, yce, yfe, run = CE.wall_edges(CE.median_frame('test', 'simulation_0232'))
hpx = float(yfe - yce)
res[738] = res[749] = float(hpx * wm)
ims = frames('simulation_0232', lambda n: np.linspace(5, n - 5, 5).astype(int)); yc, yf = [], []
for im in ims:
    gy = np.abs(np.median(cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float), cv2.CV_64F, 0, 1, ksize=3)[:, x0:x1], axis=1))
    yc.append(yce - 40 + int(np.argmax(gy[yce - 40:yce + 41]))); yf.append(yfe - 40 + int(np.argmax(gy[yfe - 40:yfe + 41])))
diag[738] = diag[749] = ('t84')

out = {i: round(float(x), 4) for i, x in res.items()}
order = [629, 757, 1965, 735, 746, 756, 738, 749]
with open(os.path.join(OUTD, 'reproduce_out.csv'), 'w') as f:
    f.write('id0,value\n')
    for i in order:
        f.write(f'{i},{float(out[i])!r}\n')
