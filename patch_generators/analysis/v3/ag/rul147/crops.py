import sys, cv2, numpy as np
sys.path.insert(0, os.path.join(os.environ.get('QP_ROOT', '.'), 'geo'))
import track_common as tc
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
vid = sys.argv[1]; out = sys.argv[2]
fr = tc.load_video(f'{R}/data/test_480p/{vid}.mp4')
fr = fr[0] if isinstance(fr, tuple) else fr
print(len(fr), fr[0].shape)
tiles = []
for f in [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 24]:
    im = np.ascontiguousarray(fr[f][..., ::-1])
    cr = im[300:500, 480:740].copy()
    cv2.putText(cr, f'f{f} t={f/24:.2f}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    tiles.append(cr)
rows = [np.hstack(tiles[i:i + 5]) for i in range(0, 10, 5)] + [np.hstack(tiles[10:14] + [np.zeros_like(tiles[0])])]
cv2.imwrite(out, np.vstack(rows))
