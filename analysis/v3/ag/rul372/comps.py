import os, sys
import cv2, numpy as np
R = '@QP_ROOT@'


def components(v, fi, mode, thr, roi=None, mina=80, close=3):
    cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, im = cap.read()
    assert ok
    if mode == 'dark':
        m = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < thr).astype(np.uint8)
    else:
        m = (im.min(axis=2) > thr).astype(np.uint8)
    if roi is not None:
        z = np.zeros_like(m); x0, y0, x1, y1 = roi; z[y0:y1, x0:x1] = m[y0:y1, x0:x1]; m = z
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if st[i, 4] < mina:
            continue
        ys, xs = np.nonzero(lab == i)
        (cx, cy), (a, b), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        out.append(dict(area=int(st[i, 4]), bbox=[int(x) for x in st[i, :4]], rr=[round(cx, 1), round(cy, 1), round(max(a, b), 1),
                    round(min(a, b), 1), round(ang, 1)], cen=[round(float(c), 1) for c in cen[i]]))
    return im.shape, out


if __name__ == '__main__':
    v, fi, mode, thr = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    roi = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 and sys.argv[5] != '-' else None
    mina = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    shp, out = components(v, fi, mode, thr, roi, mina)
    print(v, fi, mode, thr, 'shape', shp, 'n', len(out))
    for c in sorted(out, key=lambda c: -c['area'])[:12]:
        print('  ', c)
