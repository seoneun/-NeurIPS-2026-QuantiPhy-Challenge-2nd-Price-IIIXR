import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); import trk
R = trk.R

def frames(v):
    cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4')); out = []
    while True:
        ok, im = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.array(out)

def period(profile, lo=8, hi=200):
    p = profile - profile.mean(); ac = np.correlate(p, p, 'full')[len(p) - 1:]; ac /= ac[0]
    cand = [(ac[k], k) for k in range(lo, min(hi, len(ac) - 1)) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
    return sorted(cand, reverse=True)[:4]

if __name__ == '__main__':
    v = 'simulation_0142'
    dd, d, s = trk.objseries(v, 'man')
    if not s: dd, d, s = trk.objseries(v, 'person')
    f = np.array([x[0] for x in s]); c = np.array([x[1]['c'] for x in s]); bot = np.array([x[1]['box'][3] for x in s])
    A = np.vstack([f / d['fps'], np.ones(len(f))]).T
    vx = np.linalg.lstsq(A, c[:, 0], rcond=None)[0][0]; vy = np.linalg.lstsq(A, c[:, 1], rcond=None)[0][0]
    vpx = float(np.hypot(vx, vy)); mpp = 1.693 / vpx
    print(v, dd, 'n', len(s), 'man px speed %.1f (vx %.1f vy %.1f) -> mpp %.5f; feet row median %.0f; box h med %.0f -> %.2f m' % (
        vpx, vx, vy, mpp, np.median(bot), np.median([x[1]['box'][3] - x[1]['box'][1] for x in s]),
        mpp * np.median([x[1]['box'][3] - x[1]['box'][1] for x in s])))
    F = frames(v); med = np.median(F, axis=0); H, W = med.shape
    yb = int(np.median(bot))
    for y0 in (yb - 6, yb, yb + 6, yb + 12):
        band = med[max(0, y0 - 3):y0 + 3, 20:W - 20].mean(axis=0)
        g = np.abs(np.diff(band))
        print('')
    cv2.imwrite(os.path.join(R, 'cache/tmp/rul373/med0142.jpg'), med.astype(np.uint8))
