import argparse, json, os, sys
import numpy as np

R = os.environ.get("QP_ROOT") or os.getcwd()


def load(video, trk='v3ntmv'):
    d = json.load(open(os.path.join(R, f'geo/out/tracks_test_{trk}/{video}.json')))
    return d


def series(d, obj):
    o = d['objs'][str(obj)]
    fps = d['fps']
    T, C, S = [], [], []
    for i, e in enumerate(o):
        if not e or not e.get('st'):
            continue
        st = e['st']
        if st.get('border'):
            continue
        T.append(d['fidx'][i] / fps)
        C.append(st['c'])
        rr = st.get('rr') or [0, 0]
        b = st['box']
        S.append(max(np.sqrt(st['area'] * 4 / np.pi), 0))
    return np.array(T), np.array(C, float), np.array(S, float)


def ray(C, f, cx, cy):
    r = np.stack([(C[:, 0] - cx) / f, (C[:, 1] - cy) / f, np.ones(len(C))], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def smooth(x, k=5):
    return np.array([np.median(x[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(x))])




def fwd(t, P, t0, n=5, dt=1 / 24):
    i = int(np.argmin(np.abs(t - t0)))
    j = i + n
    if j >= len(t):
        return np.nan
    return float(np.linalg.norm(P[j] - P[i]) / (t[j] - t[i]))
