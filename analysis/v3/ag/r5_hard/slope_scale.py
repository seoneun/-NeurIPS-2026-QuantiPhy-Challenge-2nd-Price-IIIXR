import json, os, numpy as np
R = '@QP_ROOT@'
FPS = 24.0
SCENE = {'simulation_0181': ('box', 2.338, 0.75), 'simulation_0182': ('box', 2.338, 0.75),
         'simulation_0183': ('tool box', 1.58, 1.25), 'simulation_0184': ('tool box', 1.58, 1.25),
         'simulation_0185': ('gift box', 3.18, 1.0), 'simulation_0186': ('gift box', 3.18, 1.0)}
DEPTH = {'simulation_0182': (9.522, 9.384), 'simulation_0184': (9.194, 9.096), 'simulation_0186': (9.859, 9.741)}

def centroids(v, tset):
    fn = f'{R}/geo/out/tracks_test_{tset}/{v}.json'
    if not os.path.exists(fn): return None
    j = json.load(open(fn)); ph = SCENE[v][0]
    q = [q for q in j['queries'] if q['phrase'] == ph]
    q = [x for x in q if x.get('obj') is not None]
    if not q: return None
    o = j['objs'][str(q[0]['obj'])]
    return np.array([o[i]['st']['c'] if o[i] else [np.nan, np.nan] for i in range(len(o))], float)

def fall(c):
    dy = np.diff(c[:, 1]); land = None
    for i in range(1, len(dy)):
        if dy[i - 1] >= 3 and dy[i] < dy[i - 1]: land = i; break
    fr = np.arange(0, land + 1); t = fr / FPS
    p = np.polyfit(t, c[fr, 1], 2)
    return 2 * p[0], land

def pathlen(c):
    st = np.hypot(*np.diff(c, axis=0).T); st = np.nan_to_num(st)
    return np.concatenate([[0], np.cumsum(st)])

def vpx(s, t, half=0.25):
    fr = np.arange(int(round((t - half) * FPS)), int(round((t + half) * FPS)) + 1); fr = fr[(fr >= 0) & (fr < len(s))]
    p = np.polyfit(fr / FPS, s[fr], 2); return 2 * p[0] * t + p[1]

def aslide(s, t0, t1):
    fr = np.arange(int(round(t0 * FPS)), min(int(round(t1 * FPS)), len(s) - 1) + 1)
    return 2 * np.polyfit(fr / FPS, s[fr], 2)[0]

def table():
    out = {}
    for v, (ph, acc, T) in SCENE.items():
        for tset in ('v3ntmv', 'v2sc'):
            c = centroids(v, tset)
            if c is None: continue
            af, land = fall(c); s = pathlen(c)
            a_sl = aslide(s, T, T + 0.5)
            d = dict(a_fall=af, land=land, s_fall=9.8 / af, a_slide=a_sl, s_slope=acc / a_sl, ratio=(acc / a_sl) / (9.8 / af),
                     v1=vpx(s, 1.0), v15=vpx(s, 1.5), dy=np.diff(c[:10, 1]))
            out[(v, tset)] = d
    return out

if __name__ == '__main__':
    tb = table()
    print('')
    for (v, ts), d in tb.items():
        print(f'{v} {ts:6s} {d["land"]:4d} {d["a_fall"]:6.0f} {d["s_fall"]:.5f} {d["a_slide"]:6.0f}  {d["s_slope"]:.5f} {d["ratio"]:5.2f} | '
              f'{d["v1"]:6.0f} {d["v15"]:6.0f} | {d["v1"] * d["s_fall"]:.3f} {d["v15"] * d["s_fall"]:.3f} | {d["v1"] * d["s_slope"]:.3f} {d["v15"] * d["s_slope"]:.3f} | '
              + ' '.join(f'{x:.0f}' for x in d['dy']))
