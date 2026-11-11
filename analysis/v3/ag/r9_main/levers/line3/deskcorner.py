import re
import numpy as np, cv2
from scipy.optimize import brentq, least_squares

W, H = 944, 480
VIDS = ['simulation_0202', 'simulation_0205', 'simulation_0206', 'simulation_0209']
NEAR = re.compile(r'distance_desktop_corner_(?:near|closest_to)_camera = ([\d.]+)')
FAR = re.compile(r'distance_desktop_corner_far_right_camera = ([\d.]+)')


def ray(p, f):
    d = np.array([p[0] - W / 2, p[1] - H / 2, f], float); return d / np.linalg.norm(d)


def ransac_line(x, y, tol=1.5, it=600, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(it):
        i, j = rng.choice(len(x), 2, replace=False)
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i]); b = y[i] - a * x[i]
        inl = np.abs(y - (a * x + b)) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    a, b = np.polyfit(x[best], y[best], 1)
    return a, b, float(best.mean())


def silhouette_corners(M):
    ys, xs = np.where(M)
    xl, xr = xs.min(), xs.max()
    BL = np.array([xl, ys[xs == xl].min()], float); FR = np.array([xr, ys[xs == xr].min()], float)
    top = np.array([np.argmax(M[:, c]) if M[:, c].any() else H for c in range(W)], float)
    xt = int(np.argmin(np.where(np.arange(W) >= (xl + xr) / 2, top, H)))
    cb = np.arange(int(BL[0]) + 20, xt - 15); cr = np.arange(xt + 15, int(FR[0]) - 5)
    a1, b1, s1 = ransac_line(cb.astype(float), top[cb]); a2, b2, s2 = ransac_line(cr.astype(float), top[cr])
    x = (b2 - b1) / (a1 - a2); BR = np.array([x, a1 * x + b1])
    return dict(BL=BL, BR=BR, FR=FR, share_back=s1, share_right=s2)


def rim_corner(frame, M, BL):
    rb = frame[:, :, 2].astype(int) - frame[:, :, 0].astype(int)
    red = ((rb >= 15) & M).astype(np.uint8)
    n, lab = cv2.connectedComponents(red); k = 1 + np.argmax(np.bincount(lab.ravel())[1:]); C = lab == k
    low = np.array([np.where(C[:, x])[0].max() if C[:, x].any() else -1 for x in range(W)], float)
    xs = np.arange(int(BL[0]) + 20, int(BL[0]) + 250); x0 = xs[np.argmax(low[xs])]
    xl = np.arange(x0 - 45, x0 - 11); xf = np.arange(x0 + 15, x0 + 81)
    a1, b1 = np.polyfit(xl, low[xl], 1); a2, b2 = np.polyfit(xf, low[xf], 1)
    x = (b2 - b1) / (a1 - a2)
    return np.array([x, a1 * x + b1])


def pixels(masks):
    z = np.load(masks); out = {}
    for v in VIDS:
        ms, sc = z[f'{v}_masks'], z[f'{v}_scores']
        i = int(np.argmax([s if m.sum() > 0.5 * 677 * 270 else -1 for m, s in zip(ms, sc)])); M = ms[i] > 0
        c = silhouette_corners(M); c['FL'] = rim_corner(z[f'{v}_frame'], M, c['BL']); c['ok_BR'] = c['share_back'] >= 0.6
        out[v] = c
    return out


def solve_case(px, f, dn, dfar):
    dBL, dBR, dFR = ray(px['BL'], f), ray(px['BR'], f), ray(px['FR'], f)

    def pts(l):
        BL, BR, FR = l[0] * dBL, l[1] * dBR, dfar * dFR
        return BL, BR, FR, BL + FR - BR

    def res(l):
        BL, BR, FR, FL = pts(l)
        ang = np.dot(BR - FR, FL - FR) / np.linalg.norm(BR - FR) / np.linalg.norm(FL - FR)
        return [np.linalg.norm(FL) - dn, ang]
    best = None
    for g in [(4.0, 4.0), (5.0, 4.0), (4.0, 5.0), (5.5, 4.5), (3.5, 4.5), (5.0, 5.5)]:
        r = least_squares(res, g, bounds=([0.5, 0.5], [20, 20]))
        if best is None or r.cost < best.cost:
            best = r
    BL, BR, FR, FL = pts(best.x)
    q = FL / FL[2] * f; FLpx = np.array([q[0] + W / 2, q[1] + H / 2])
    e1, e2 = np.linalg.norm(FR - FL), np.linalg.norm(BL - FL)
    return dict(FLpx=FLpx, front=e1, length=max(e1, e2), width=min(e1, e2))


def fit_focal(px, dn, dfar):
    return brentq(lambda fw: solve_case(px, fw * W, dn, dfar)['FLpx'][0] - px['FL'][0], 0.9, 1.8, xtol=1e-5)
