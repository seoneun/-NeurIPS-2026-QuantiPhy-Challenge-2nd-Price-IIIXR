import sys, cv2, numpy as np
cap = cv2.VideoCapture('data/test/internet_0017.mp4'); P = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    im = f.astype(int); B, G, R = im[..., 0], im[..., 1], im[..., 2]
    sk = ((im.min(2) < 245) & (R - B > 25) & (R > 100)).astype(np.uint8); sk[:, :2950] = 0; sk[:, 3600:] = 0; sk[:700] = 0
    sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, l, st, ce = cv2.connectedComponentsWithStats(sk); k = 1 + np.argmax(st[1:, 4])
    ys, xs = np.where(l == k); top = ys.min(); sel = ys <= top + 10
    P.append((i, xs[sel].mean(), top, st[k][4])); i += 1
np.savetxt(sys.argv[1], np.array(P), fmt='%.1f')
