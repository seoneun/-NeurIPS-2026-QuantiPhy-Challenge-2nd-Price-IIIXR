import json, os, sys, numpy as np
R = '@QP_ROOT@'
OUT = f'{R}/cache/tmp/rul380'

def load(video, phrase, kind):
    for ts in ('v2sc', 'v3ntmv'):
        f = f'{R}/geo/out/tracks_test_{ts}/{video}.json'
        if not os.path.exists(f): continue
        d = json.load(open(f))
        q = [x for x in d['queries'] if x['phrase'] == phrase and x.get('obj') is not None]
        if not q: continue
        o = d['objs'][str(q[0]['obj'])]
        T, P = [], []
        for fi, e in zip(d['fidx'], o):
            if not e or not e.get('st'): continue
            st = e['st']
            p = [(st['box'][0] + st['box'][2]) / 2, st['bottom']] if kind == 'bot' else st['c']
            T.append(fi / d['fps']); P.append(p)
        T, P = np.array(T), np.array(P, float)
        np.savetxt(f'{OUT}/traj_{video}_{phrase.replace(" ", "_")}.csv', np.c_[T, P], delimiter=',', header='t,x,y', fmt='%.4f')
        return T, P, ts, d['fps']
    return None

def homog(a, b):
    def norm(p):
        m = p.mean(0); s = np.sqrt(2) / max(1e-9, np.mean(np.linalg.norm(p - m, axis=1)))
        Tm = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]]); return Tm
    Ta, Tb = norm(a), norm(b)
    ah = (Ta @ np.c_[a, np.ones(len(a))].T).T; bh = (Tb @ np.c_[b, np.ones(len(b))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(ah, bh):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ H @ Ta
    pr = (H @ np.c_[a, np.ones(len(a))].T).T; pr = pr[:, :2] / pr[:, 2:3]
    return np.sqrt(np.mean(np.sum((pr - b) ** 2, 1)))

def proj1(ua, ub):
    M = np.c_[ua, np.ones_like(ua), -ua * ub]
    x = np.linalg.lstsq(M, ub, rcond=None)[0]
    pr = (x[0] * ua + x[1]) / (x[2] * ua + 1)
    return np.sqrt(np.mean((pr - ub) ** 2))

def axis_coord(P):
    m = P.mean(0); w = np.linalg.svd(P - m)[2][0]; return (P - m) @ w

def fit(TA, PA, TB, PB, s, rev=False):
    tq = TA + s
    ok = (tq >= TB.min()) & (tq <= TB.max())
    if ok.sum() < 8: return np.nan, np.nan, ok.sum()
    pa = PA[ok][::-1] if rev else PA[ok]
    pb = np.c_[np.interp(tq[ok], TB, PB[:, 0]), np.interp(tq[ok], TB, PB[:, 1])]
    ext = np.ptp(axis_coord(pb)) or 1.0
    return homog(pa, pb) / ext, proj1(axis_coord(pa), axis_coord(pb)) / ext, ok.sum()

def run(pairs, fh):
    for va, pa_, vb, pb_, kind in pairs:
        ka, kb = (kind.split(',') + [kind])[:2]
        A = load(va, pa_, ka); B = load(vb, pb_, kb)
        if A is None or B is None:
            fh.write(f'{va} {pa_} / {vb} {pb_}: missing track A={A is not None} B={B is not None}\n'); continue
        TA, PA, tsa, fa = A; TB, PB, tsb, fb = B
        fh.write(f'\n{va}[{tsa},{fa}fps,{len(TA)}fr,ext {np.ptp(axis_coord(PA)):.0f}px] {pa_} -> {vb}[{tsb},{fb}fps,{len(TB)}fr,ext {np.ptp(axis_coord(PB)):.0f}px] {pb_} ({kind})\n')
        rows = []
        for s in np.round(np.arange(-0.5, 0.501, 0.0625), 4):
            h, p, n = fit(TA, PA, TB, PB, s); rows.append((s, h, p, n))
        h0, p0, n0 = fit(TA, PA, TB, PB, 0.0); hc, pc, _ = fit(TA, PA, TB, PB, 0.0, rev=True)
        sh = min(rows, key=lambda r: r[1] if r[1] == r[1] else 9)[0]; sp = min(rows, key=lambda r: r[2] if r[2] == r[2] else 9)[0]
        fh.write(f'   s=0: H2 {h0:.4f}  P1 {p0:.4f} (n={n0}) | control (A reversed): H2 {hc:.4f}  P1 {pc:.4f} | argmin s: H2 {sh:+.3f}  P1 {sp:+.3f}\n')
        fh.write('   scan H2: ' + ' '.join(f'{r[0]:+.2f}:{r[1]:.3f}' for r in rows[::2]) + '\n')
        fh.write('   scan P1: ' + ' '.join(f'{r[0]:+.2f}:{r[2]:.3f}' for r in rows[::2]) + '\n')

if __name__ == '__main__':
    pairs = [l.split('|') for l in sys.argv[1:]]
    with open(f'{OUT}/anim_check.txt', 'a') as fh: run(pairs, fh)
