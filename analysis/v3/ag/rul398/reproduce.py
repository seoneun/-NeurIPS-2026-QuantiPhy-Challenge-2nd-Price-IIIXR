import json
import numpy as np
R = '@QP_ROOT@'; D = 0.097


def speeds(video, key):
    j = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/{video}.json'))
    o = [q['obj'] for q in j['queries'] if q['key'] == key][0]
    fr = {}
    for pos, x in enumerate(j['objs'][str(o)]):
        st = x.get('st') if isinstance(x, dict) else None
        if st and st.get('rr'): fr[j['fidx'][pos]] = (np.array(st['c'], float), st['rr'])

    def step(f):
        (c1, rr), (c0, _) = fr[f], fr[f - 1]
        a, b, ang = rr; th = np.radians(ang)
        u = np.array([np.cos(th), np.sin(th)]); w = np.array([-np.sin(th), np.cos(th)])
        d = c1 - c0
        return np.hypot((d @ u) * D / a, (d @ w) * D / b) * j['fps']
    return {t: (step(int(t * 24)), np.mean([step(f) for f in range(int(t * 24) - 2, int(t * 24) + 3)])) for t in (1.0, 1.5)}


sp = {'p': speeds('simulation_0224', 'moving chess|'), 's': speeds('simulation_0224_segmented', 'red general chess|')}
new = {t: float(np.mean([sp[c][t][k] for c in 'ps' for k in (0, 1)])) for t in (1.0, 1.5)}
patch = {}
for i, t in ((1945, 1.0), (693, 1.5)):
    nv = new[t]
    patch[str(i)] = {'value': round(nv, 4)}
    print(i, round(nv, 4))
