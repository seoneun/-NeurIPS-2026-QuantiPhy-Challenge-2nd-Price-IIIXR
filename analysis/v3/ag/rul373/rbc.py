import os, sys, numpy as np, cv2
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
V = 4.7833
def cells(v):
    cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4')); fps = cap.get(cv2.CAP_PROP_FPS); out = []
    while True:
        ok, im = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); m = (hsv[..., 1] > 90).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        comp = sorted([(st[j, 4], j) for j in range(1, n)], reverse=True)[:2]
        cs = sorted([(cen[j][0], cen[j][1], st[j, 2], st[j, 3], st[j, 4]) for a, j in comp if a > 30])
        out.append(cs)
    return fps, out
def run(v, z1, z2, W):
    fps, out = cells(v)
    i1, i2 = int(round(1 * fps)), int(round(2 * fps))
    for i in (0, i1, i2, len(out) - 1):
        print('')
    L1, L2 = out[i1][0], out[i2][0]
    vlat = np.sqrt(max(V ** 2 - (z2 - z1) ** 2, 0))
    dpx = np.hypot(L2[0] - L1[0], L2[1] - L1[1])
    f = dpx * (z1 + z2) / 2 / vlat if vlat > 0.5 else float('nan')
    D1 = L1[2] * z1 / f; D2 = L2[2] * z2 / f
    fD = (L1[2] * z1 + L2[2] * z2) / 2
    a, b = out[i1][0], out[i1][1]
    zr = z1 * a[2] / b[2]
    lat = np.hypot(b[0] - a[0], b[1] - a[1]) * z1 / f if f == f else float('nan')
    dist = np.sqrt(lat ** 2 + (zr - z1) ** 2) if f == f else float('nan')
    lat_D = np.hypot(b[0] - a[0], b[1] - a[1]) / a[2]; dz_D = (zr - z1) * a[2] / fD * (fD / (a[2] * z1)) * z1 / (fD / a[2])
    dist_D = np.sqrt(lat_D ** 2 + ((zr - z1) / (fD / a[2])) ** 2)
    print('')
    return dict(f=f, D=(D1 + D2) / 2, dist=dist, dist_D=dist_D, hw=a[3] / a[2])
if __name__ == '__main__':
    r9 = run('simulation_0109', 22.0155, 24.0333, 480)
    r10 = run('simulation_0110', 17.9557, 22.7150, 854)
    for D in (r9['D'], 1.642, 1.734):
        print('')
