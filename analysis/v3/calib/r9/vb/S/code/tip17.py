import sys, cv2, numpy as np
cap = cv2.VideoCapture('data/test/internet_0017_segmented.mp4'); n = int(cap.get(7))
P = []
for k in range(n):
    ok, im = cap.read(); im = im.astype(int)
    B, G, R = im[..., 0], im[..., 1], im[..., 2]
    sk = ((im.min(2) < 245) & (R - B > 25) & (R > 100)).astype(np.uint8)
    sk[:, :880] = 0; sk[:, 1060:] = 0
    sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nl, lab, st, cen = cv2.connectedComponentsWithStats(sk); i = 1 + np.argmax(st[1:, 4])
    ys, xs = np.where(lab == i); top = ys.min(); sel = ys <= top + 4
    P.append((k, xs[sel].mean(), top, st[i, 4]))
np.savetxt(sys.argv[1], np.array(P), fmt='%.1f')
