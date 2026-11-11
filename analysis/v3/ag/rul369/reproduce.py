import csv
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = '@QP_ROOT@'
D = os.path.join(R, 'analysis/v3/ag/rul369')
ORDER = [1519, 1526, 1530, 1516, 1055, 1645, 1664, 1095]

aud = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
sub = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv'))
assert len(sub) == len(aud) and (aud.id0.values == np.arange(len(aud))).all()
aud['cur'] = sub.parsed_value.values
cur = dict(zip(aud.id0, aud.cur))


def text(video, col):
    s = aud.loc[aud.video == video, col].dropna().astype(str).unique()
    assert len(s) == 1
    return s[0]


def prior_m(video, name):
    m = re.findall(r'(?:^|\n)\s*' + re.escape(name) + r'\s*=\s*([\d.]+)\s*m\b', text(video, 'prior'))
    assert len(m) == 1
    return float(m[0])


def depth_m(video, obj, t):
    hits = [float(c) for a, b, c in re.findall(r't=([\d.]+)s,\s*distance_(\w+?)_camera\s*=\s*([\d.]+)\s*m',
                                                 text(video, 'depth')) if b == obj and float(a) == t]
    assert len(hits) == 1
    return hits[0]


def track(d, v):
    return json.load(open(os.path.join(R, 'geo/out', d, f'{v}.json')))


def qidx(js, phrase):
    return [i for i, q in enumerate(js['queries']) if q['phrase'] == phrase][0]


def tire_rows(js, qi):
    out = []
    for i, f in enumerate(js['objs'][str(js['queries'][qi]['obj'])]):
        st = f.get('st') if isinstance(f, dict) else None
        if st and st.get('box') and not st.get('border'):
            x0, y0, x1, y1 = st['box']; out.append((js['fidx'][i], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return np.array(out, float)


def marking():
    hcam149 = prior_m('simulation_0149', 'diameter of the tire') / 2
    assert hcam149 == 0.382
    t150 = 0.5
    Z150 = depth_m('simulation_0150', 'yellowCarLeftFrontTire', t150)
    assert Z150 == 11.38
    med = {}
    for v, frames in [('simulation_0150', [0, 2, 4, 6]), ('simulation_0149', [44, 46, 48])]:
        js = track('tracks_test_v2sc', v)
        W, H = js['W'], js['H']; f = 50 / 36 * W
        qp = qidx(js, 'pillar'); qt = qidx(js, 'tire')
        tr = tire_rows(js, qt)
        if v == 'simulation_0150':
            fr0 = t150 * js['fps']
            assert fr0 == 12
            r = tr[np.argmin(abs(tr[:, 0] - fr0))]
            ray = np.array([r[1] - W / 2, r[2] - H / 2, f]); Z = Z150 * f / np.linalg.norm(ray)
            Dt = np.median(tr[(tr[:, 3] >= 78), 4]); yc = r[2] + Dt / 2
            hcam = (yc - H / 2) * Z / f; yh = H / 2
            tire = prior_m(v, 'diameter of the tire')
            print('')
        else:
            full = tr[(tr[:, 4] >= 55)]
            yh = np.median(full[:, 2]); hcam = hcam149
            print('')
        cap = cv2.VideoCapture(os.path.join(R, f'data/test/{v}.mp4'))
        res = []
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read()
            sx = im.shape[1] / W; sy = im.shape[0] / H
            pos = int(np.argmin([abs(x - fr) for x in js['fidx']]))
            x0, y0, x1, y1 = js['objs'][str(js['queries'][qp]['obj'])][pos]['st']['box']
            X0, X1, Y0, Y1 = int(x0 * sx), int(x1 * sx), int(y0 * sy), int(min(im.shape[0], (y1 + 5) * sy))
            hsv = cv2.cvtColor(im[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV)
            yel = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90)
            cnt = yel.sum(1)
            rows = np.where(cnt >= 3)[0]
            top = None
            for rr_ in rows:
                if (cnt[rr_:rr_ + int(20 * sy)] >= 3).sum() >= int(5 * sy):
                    top = rr_; break
            if top is None:
                print(''); continue
            ytop = (Y0 + top) / sy
            Zb = hcam * f / (y1 - yh)
            hpx = y1 - ytop
            res.append((fr, ytop, y1, hpx, Zb, hpx * Zb / f, (x1 - x0) * Zb / f))
            print('')
        a = np.array(res)
        med[v] = np.median(a[:, 5])
    m150 = float(f"{med['simulation_0150']:.3f}"); m149 = float(f"{med['simulation_0149']:.3f}")
    mk = round((m150 + m149) / 2, 3)
    meas = (med['simulation_0150'] + med['simulation_0149']) / 2
    print('')
    return mk, float(meas)


def kin():
    v1, v1raw = [], []
    for v, d, qi, pname in [('simulation_0150', 'tracks_test_v2sc', 1, 'diameter of the tire'),
                             ('simulation_0151', 'tracks_test_v3ntmv', 1, 'diameter of the tire of the yellow car')]:
        tire = prior_m(v, pname)
        assert tire == 0.764
        js = track(d, v); q = js['queries'][qi]; fps = js['fps']
        rows = []
        for i, f in enumerate(js['objs'][str(q['obj'])]):
            st = f.get('st') if isinstance(f, dict) else None
            if st and st.get('box'):
                x0, y0, x1, y1 = st['box']
                rows.append((js['fidx'][i], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        a = np.array(rows, float)
        full = a[(a[:, 3] >= 78) & (a[:, 4] >= 78)]
        Dt = np.median(np.r_[full[:, 3], full[:, 4]]); sc = tire / Dt
        t = full[:, 0] / fps; x = full[:, 1]
        for deg in (3, 4):
            p = np.polyfit(t, x, deg); dp = np.polyder(p)
            s = abs(np.polyval(dp, 1.0)) * sc
            v1raw.append(float(s)); v1.append(float(f'{s:.2f}'))
        print('')
    vnew = round(float(np.mean(v1)), 2)
    meas = float(np.mean(v1raw))
    print('')
    return vnew, meas


def rocket():
    v = 'simulation_0144'
    tD = np.array([0.0, 0.5])
    dD = np.array([depth_m(v, 'rocket', tt) for tt in tD])
    assert list(dD) == [43.064, 41.127]
    js = track('tracks_test_v3ntmv', v)
    q = js['queries'][1]; fps = js['fps']; W, H = js['W'], js['H']
    rows = []
    for i, f in enumerate(js['objs'][str(q['obj'])]):
        st = f.get('st') if isinstance(f, dict) else None
        if st and st.get('box') and not st.get('border'):
            x0, y0, x1, y1 = st['box']
            rows.append((js['fidx'][i] / fps, (x0 + x1) / 2, (y0 + y1) / 2))
    a_ = np.array(rows)

    def model(p, fl):
        X0, Y0, Z0, th, ph, acc, t0 = p[:7]
        u = np.array([np.cos(ph) * np.sin(th), np.sin(ph), -np.cos(ph) * np.cos(th)])

        def pos(t):
            s = acc * np.maximum(t - t0, 0) ** 2 / 2
            return np.array([X0, Y0, Z0])[None, :] + s[:, None] * u[None, :]
        return pos

    def resid(p, fixed_f=None):
        fl = fixed_f if fixed_f else p[7]
        pos = model(p, fl)
        P = pos(a_[:, 0])
        px = W / 2 + fl * P[:, 0] / P[:, 2]; py = H / 2 + fl * P[:, 1] / P[:, 2]
        r = list(px - a_[:, 1]) + list(py - a_[:, 2])
        Dd = np.linalg.norm(pos(tD), axis=1)
        r += list((Dd - dD) * 50)
        return np.array(r)

    fdef = 50 / 36 * W
    best = None
    for th0 in (-0.6, 0.0, 0.6):
        for ph0 in (-0.6, 0.0, 0.6):
            p0 = [5.0, 2.0, 42.0, th0, ph0, 18.0, 0.0]
            lb = [-50, -50, 5, -np.pi, -np.pi / 2, 0.5, -0.5]
            ub = [50, 50, 100, np.pi, np.pi / 2, 200, 0.6]
            try:
                r = least_squares(resid, p0, args=(fdef,), bounds=(lb, ub))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    acc, t0 = best.x[5], best.x[6]
    a_def = float(f'{acc:.2f}'); t0_def = float(f'{t0:.3f}')
    r2 = cur[1515] / (1 - t0_def)
    anew = round((a_def + r2) / 2, 2)
    meas = (float(acc) + cur[1515] / (1 - float(t0))) / 2
    print('')
    return anew, meas


def wheel():
    js = track('tracks_test_v2sc', 'simulation_0010_segmented')

    def med_h(phrase):
        q = [x for x in js['queries'] if x['phrase'] == phrase][0]
        hs = [f['st']['box'][3] - f['st']['box'][1] for f in js['objs'][str(q['obj'])]
              if isinstance(f, dict) and f.get('st') and f['st'].get('box')]
        return float(np.median(hs))
    hw, hc = med_h('wheel'), med_h('cart')
    w1 = hw / hc * cur[1054]
    geo = pd.read_csv(os.path.join(R, 'preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv'))
    gr = geo.iloc[1645]
    assert gr['Unnamed: 0'] == 1645 and gr.video_id == 'simulation_0010_segmented' and gr.question == aud.question[1645]
    g_full = float(gr.geo_value)
    g1645 = float(f'{g_full:.4g}')
    wnew = round((w1 + g1645) / 2, 2)
    meas = (w1 + g_full) / 2
    print('')
    return wnew, meas


def walk():
    d0 = depth_m('simulation_0021', 'human', 0.0); d1 = depth_m('simulation_0021', 'human', 1.0)
    assert (d0, d1) == (13.882, 13.768)
    clip, clipraw, loc5, loc9, loc5raw, loc9raw = [], [], [], [], [], []
    for v in ('simulation_0021', 'simulation_0021_segmented'):
        assert (depth_m(v, 'human', 0.0), depth_m(v, 'human', 1.0)) == (d0, d1)
        hp = prior_m(v, 'height of the person')
        assert hp == 1.87
        js = track('tracks_test_v3ntmv', v)
        q = js['queries'][0]; fps = js['fps']
        a = []
        for i, f in enumerate(js['objs'][str(q['obj'])]):
            st = f.get('st') if isinstance(f, dict) else None
            if st and st.get('box'):
                x0, y0, x1, y1 = st['box']
                a.append((js['fidx'][i] / fps, (x0 + x1) / 2, y0, y1, x1 - x0, y1 - y0, st['c'][0]))
        a = np.array(a)
        t, hgt, xm = a[:, 0], a[:, 5], a[:, 6]
        d = d0 + (d1 - d0) * t
        fpx = np.median(hgt * d / hp)
        vrad = abs(d1 - d0) / 1.0
        for lo, hi, dst, dstraw in [(0, 99, clip, clipraw), (0.5, 1.5, loc5, loc5raw), (0.9, 1.1, loc9, loc9raw)]:
            m = (t >= lo) & (t <= hi)
            slm = np.polyfit(t[m], xm[m], 1)[0]
            vlm = abs(slm) * np.median(d[m]) / fpx
            s = np.hypot(vlm, vrad)
            dstraw.append(float(s)); dst.append(float(f'{s:.3f}'))
        print('')
    loc = loc5 + loc9
    vavg = round(float(np.mean(clip)), 2); vat1 = round(float(np.mean(loc)), 2)
    m_avg = float(np.mean(clipraw)); m_at1 = float(np.mean(loc5raw + loc9raw))
    print('')
    print('')
    return vavg, vat1, m_avg, m_at1


mk, m_mk = marking()
vnew, m_v = kin()
anew, m_a = rocket()
wnew, m_w = wheel()
vavg, vat1, m_avg, m_at1 = walk()
vals = {1519: mk, 1526: mk, 1530: vnew, 1516: anew, 1055: wnew, 1645: wnew, 1664: vavg, 1095: vat1}
with open(os.path.join(D, 'reproduce_out.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['id0', 'value'])
    for i in ORDER:
        w.writerow([i, repr(float(vals[i]))])
