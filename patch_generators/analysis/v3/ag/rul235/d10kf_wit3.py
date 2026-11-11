"""diagnostic: windowed pixel speed per obj (frames a..b), 0077 v2sc; and 0078 frame strip saved for eyeballing."""
import json, sys, numpy as np, cv2
sys.path.insert(0, 'geo'); import track_common as tc
J = json.load(open('geo/out/tracks_test_v2sc/simulation_0077.json'))
for k, L in J['objs'].items():
    c = np.array([s['st']['c'] if s and s.get('st') else [np.nan]*2 for s in L], float)
    b = np.array([s['st']['box'] if s and s.get('st') else [np.nan]*4 for s in L], float)
    for a, e in ((0, 24), (12, 36), (24, 48), (0, 48), (48, 71)):
        print('0077 obj', k, a, e, 'px/f %.3f' % ((c[e, 0] - c[a, 0]) / (e - a)), 'x1 %.0f x2 %.0f' % (b[e, 0], b[e, 2]))
for v in ('0078', '0076', '0077'):
    fr = tc.load_video(f'data/test_480p/simulation_{v}.mp4')
    cv2.imwrite(f'cache/tmp/ag_d10d2sim_keyfact_{v}.jpg', cv2.cvtColor(np.vstack([fr[i] for i in (0, 24, 48, 70)]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
