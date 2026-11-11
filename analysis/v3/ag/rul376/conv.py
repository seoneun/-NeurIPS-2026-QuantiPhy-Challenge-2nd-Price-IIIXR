import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, HERE)
from items import load
TR = f'{R}/geo/out/tracks_test_v3ntmv'


def trk(video, obj):
    d = json.load(open(f'{TR}/{video}.json'))
    fr = d['objs'][str(obj)] if str(obj) in d['objs'] else d['objs'][obj]
    c = np.array([e['st']['c'] if e and e.get('st') else [np.nan, np.nan] for e in fr], float)
    box = np.array([e['st']['box'] if e and e.get('st') else [np.nan] * 4 for e in fr], float)
    rr = np.array([e['st']['rr'][:2] if e and e.get('st') else [np.nan] * 2 for e in fr], float)
    return d, c, box, rr


def parse():
    p = [json.loads(l) for l in open(f'{R}/geo/out/parse_test_v2.jsonl')]
    return {r['id']: r for r in p}


def alternatives(T):
    P = parse(); A = {}
    cur = dict(zip(T.id0, T.c26c)); g = dict(zip(T.id0, T.g)); cs = dict(zip(T.id0, T.cs))
    for i in (200, 1805):
        A[i] = (g[i], 't12')
    for i, v in ((759, 'simulation_0233'), (1967, 'simulation_0233_segmented')):
        _, _, box, _ = trk(v, 1)
        r = np.nanmedian(box[:, 3] - box[:, 1]) / np.nanmedian(box[:, 2] - box[:, 0])
        A[i] = (cur[i] * r, 't13')
    for i, v in ((815, 'simulation_0241'), (845, 'simulation_0249')):
        d = json.load(open(f'{TR}/{v}.json'))
        ob = [q['obj'] for q in d['queries'] if q['phrase'] == 'rabbit'][0]
        _, c, _, _ = trk(v, ob)
        st = np.linalg.norm(np.diff(c, axis=0), axis=1)
        A[i] = (cur[i] * np.nanmean(st) / st[23], 't14')
    for i in T[T['item'] == 7].id0:
        t, fps = P[i]['q']['t'], P[i]['fps']; n = round(t * fps)
        A[i] = (cur[i] * (n - 0.5) / (t * fps), f'item7 ffbd: a(round({t}x{fps:g})-0.5)/fps')
    o45 = json.load(open(f'{R}/analysis/v3/ag/rul349/opt45_patch_test.json'))
    for i in (3084, 3085, 3086, 3088):
        A[i] = (float(o45[str(i)]['value']), 'item10 physb opt45 (analysis/v3/ag/rul349/opt45_patch_test.json)')
    v = 'internet_0074'; fa, fb = 23.976, 30.0
    _, ball, _, _ = trk(v, 2); _, net, _, _ = trk(v, 3); _, p1, _, _ = trk(v, 1); _, p2, _, _ = trk(v, 4)

    def spd(f, fps):
        k = max(1, round(0.25 * fps)); a, b = max(0, f - k), min(len(ball) - 1, f + k)
        return np.linalg.norm(ball[b] - ball[a]) / (b - a) * fps
    _, _, cbox, _ = trk(v, 0); mpp = 6.1 / np.nanmedian(cbox[:, 2] - cbox[:, 0])
    for i, t in ((3283, 1.0), (3286, 3.0)):
        A[i] = (spd(round(t * fa), fa) * mpp, 't15')
    dn = lambda f: abs(ball[f, 1] - net[f, 1])
    r = dn(round(fa)) / dn(round(fb)); A[3284] = (cur[3284] * r, 't7')
    dp = lambda f: min(np.linalg.norm(ball[f] - p1[f]), np.linalg.norm(ball[f] - p2[f]))
    r = dp(round(2 * fa)) / dp(round(2 * fb)); A[3285] = (cur[3285] * r, 't8')
    dq = lambda f: np.linalg.norm(p1[f] - p2[f])
    r = dq(round(4.5 * fa)) / dq(len(p1) - 1)
    A[3288] = (cur[3288] * r, 't9')
    _, c, _, _ = trk('internet_0021', 0)
    A[2977] = (cur[2977] * 30 / 25, 't10')
    r = np.linalg.norm(c[60] - c[0]) / np.linalg.norm(c[50] - c[0])
    A[2979] = (cur[2979] * r, f'item12 own fps: displacement 0-60 / 0-50 frames {r:.3f}')
    for i in (2040, 2041, 2042):
        A[i] = (cur[i] * 1.23 / 1.65, 'item13 val-0001 car scale: x 1.23/1.65')
    _, _, _, rr = trk('internet_0040', 0)
    k = np.nanmedian(rr.min(1)) / np.nanmedian(rr.max(1))
    A[3051] = (cur[3051] * k, f'item14 long-side prior: x short/long {k:.3f}')
    A[3052] = (cur[3052] * k, f'item14 long-side prior: x short/long {k:.3f}')
    A[3050] = (8.398 * k, f'item14 long-side prior: length = short side = 8.398 cm x {k:.3f}')
    for i in (2154, 2159):
        A[i] = (cs[i], 't16')
    r1, r2 = np.sqrt(176.8 * 181.0), np.sqrt(165.7 * 166.3)
    for i in (2747, 2755, 2763):
        A[i] = (float(np.sqrt(r1 * r2)), 'item19 FD1xF / gravityxF geometric mean (r4_labs3.md:86-95)')
    A[8] = (g[8], 't11')
    return A


BLOCKS = [
    ('s2card', 'S2', [14], [3050, 3051, 3052]), ('s2fps74', 'S2', [11], [3283, 3284, 3285, 3286, 3288]),
    ('s2fps21', 'S2', [12], [2977, 2979]), ('s2lane', 'S2', [13], [2040, 2041, 2042]),
    ('s2fps74card', 'S2', [11, 14], [3283, 3284, 3285, 3286, 3288, 3050, 3051, 3052]),
    ('s2fps74card2', 'S2', [11, 14], [3283, 3284, 3285, 3286, 3288, 3051, 3052]),
    ('d2opt45', 'D2', [10], [3084, 3085, 3086, 3088]), ('d2rabbit', 'D2', [5], [815, 845]),
    ('d2ball0', 'D2', [27], [8]), ('d2chairv', 'D2', [4], [759, 1967]), ('d2keel', 'D2', [1], [200, 1805]),
    ('d2ffbd', 'D2', [7], [887, 888, 894, 895, 902, 903, 912, 913, 917, 918, 919, 2006, 2007, 2008]),
    ('d2opt45ball0', 'D2', [10, 27], [3084, 3085, 3086, 3088, 8]),
    ('s3ball40', 'S3', [19], [2747, 2755, 2763]), ('d3deskhit', 'D3', [17], [2154, 2159])]


