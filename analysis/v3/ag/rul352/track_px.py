import json, os, sys
import numpy as np
R = '@QP_ROOT@'
def load(v):
    for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test'):
        f = f'{R}/geo/out/{d}/{v}.json'
        if os.path.exists(f):
            return d, json.load(open(f))
    return None, None
def px(v):
    d, j = load(v)
    out = {}
    if j is None:
        return out
    for q in j.get('queries', []):
        b = np.array([x for x in q.get('det_boxes', []) if x is not None], float)
        if len(b) == 0:
            continue
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        out[q['key']] = dict(src=d, phrase=q['phrase'], w=np.median(w), h=np.median(h), cx=np.median((b[:, 0] + b[:, 2]) / 2),
                             cy=np.median((b[:, 1] + b[:, 3]) / 2), y2=np.median(b[:, 3]), n=len(b), W=j['W'], H=j['H'])
    return out
if __name__ == '__main__':
    for v in sys.argv[1:]:
        for k, o in px(v).items():
            print(f"{v:28s} {k[:38]:38s} w={o['w']:6.1f} h={o['h']:6.1f} cx={o['cx']:6.1f} cy={o['cy']:6.1f} y2={o['y2']:6.1f} n={o['n']} {o['src']}")
