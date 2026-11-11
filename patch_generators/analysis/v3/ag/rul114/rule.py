# rul114: SIM D3 'accel at t' rows whose target has its own timed depth keys at t0=0 and t1<=t+0.25.
# If the target track shows a from-rest constant-acceleration onset on its uncut leading edge
# (s(t1/2)/s(t1) in [0.15,0.35], ~0 displacement at the first frame), then the stated radial keys give
# a focal-free lower bound a_lb = 2|d(t1)-d(0)|/t1^2 (displacement >= radial change; s = a t^2/2 from rest).
# If the answer is below a_lb: raise to a_lb * hypot(1, lat/rad) (cap 1.5) with lat from the leading edge
# and meas_f, else to a_lb. Fusion: new geo vs CaW -> gm if within 2x (floored at a_lb), else geo (3d_calib).
import sys, json, os, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul114'


def norm(s):
    return re.sub(r'[\s_]+', '', str(s).lower())


def load_track(split, vid, phrase):
    for td in ('v2sc', 'v2', 'v3nt'):
        p = R + f'/geo/out/tracks_{split}_{td}/{vid}.json'
        if not os.path.exists(p):
            continue
        T = json.load(open(p)); oid = None
        for e in T['queries']:
            if e['key'].split('|')[0] == phrase and e.get('obj') is not None:
                oid = e['obj']; break
        if oid is None or str(oid) not in T['objs']:
            continue
        return td, T, T['objs'][str(oid)]
    return None, None, None


def onset(T, O, fps, t1):
    """leading-edge from-rest check on [0,t1]. returns dict or reason str"""
    W = float(T['W']); H = float(T['H'])
    fr = []; bx = []
    for f, st in zip(T['fidx'], O):
        tt = f / fps
        if tt > t1 + 1.5 / fps:
            break
        if st is None or st.get('st') is None:
            continue
        fr.append(tt); bx.append(st['st']['box'])
    fr = np.array(fr); bx = np.array(bx, float)
    if len(fr) < 6:
        return 'few frames'

    def at(col, tt):
        m = np.abs(fr - tt) <= 1.01 / fps
        return np.median(col[m]) if m.sum() else np.nan
    for tt in (0.0, t1, t1 / 2):
        if np.abs(fr - tt).min() > 1.01 / fps:
            return 'key frames uncovered'
    cut = [(bx[:, 0] <= 2).any(), (bx[:, 1] <= 2).any(), (bx[:, 2] >= W - 2).any(), (bx[:, 3] >= H - 2).any()]
    best = None
    for e, sgn in ((0, -1), (1, -1), (2, 1), (3, 1)):
        if cut[e]:
            continue
        D = at(bx[:, e], t1) - at(bx[:, e], 0)
        if np.sign(D) != sgn:
            continue
        if best is None or abs(D) > abs(best[1]):
            best = (e, D)
    if best is None:
        return 'no uncut leading edge'
    e, D = best
    if abs(D) < 8:
        return f'leading disp {abs(D):.1f}px<8'
    s = lambda tt: abs(at(bx[:, e], tt) - at(bx[:, e], 0))
    first = fr[fr > 0.5 / fps].min()
    r = s(t1 / 2) / abs(D); s1 = s(first) / abs(D)
    return dict(edge='x1 y1 x2 y2'.split()[e], D=float(D), ratio=float(r), s_first=float(s1), W=W,
                ok=bool(0.15 <= r <= 0.35 and s1 <= 0.1))


def run(split, detname):
    P = {}
    for l in open(R + f'/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    raw = pd.read_csv(R + f'/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    D = pd.read_csv(R + '/submissions/' + detname).set_index('Unnamed: 0')
    out = {}; rows = []
    for i, j in P.items():
        if i not in D.index or D.loc[i, 'cat'] != 'D3':
            continue
        if str(raw.loc[i, 'video_source']) != 'simulation':  # (a)
            continue
        q = j['q']
        if q.get('kind') != 'accel' or q.get('t') is None or q.get('t1') is not None or q.get('t2') is not None:
            continue
        t = float(q['t'])
        tk = [norm(k) for k in (j.get('target_dkeys') or [])]
        if len(tk) != 1:
            continue
        ks = sorted([(float(e['t']), float(e['d'])) for e in j.get('depth', []) if e.get('t') is not None and norm(e['key']) == tk[0]])
        k0 = [d for tt, d in ks if abs(tt) < 1e-9]; kp = [(tt, d) for tt, d in ks if tt > 1e-9]
        rec = dict(split=split, id=i, vid=j['video_id'], t=t, ans=D.loc[i, 'answer'], src=D.loc[i, 'answer_source'], g=D.loc[i, 'g'], cs=D.loc[i, 'cs'])
        if 'y' in D.columns:
            rec['y'] = D.loc[i, 'y']
        if not k0 or not kp:
            rec['why'] = '(b) no own keys at 0 and t1'; rows.append(rec); continue
        t1, d1 = kp[0]; d0 = k0[0]
        if t1 > t + 0.25:
            rec['why'] = '(b) t1>t+0.25'; rows.append(rec); continue
        a = float(D.loc[i, 'answer']); src = str(D.loc[i, 'answer_source'])
        if abs(a - 9.8) < 1e-6 or 'rul289' in src or 'grav' in src:
            rec['why'] = '(d) rul289/gravity'; rows.append(rec); continue
        fps = float(raw.loc[i, 'fps'])  # CSV fps, never container fps
        td, T, O = load_track(split, j['video_id'], (j.get('target') or {}).get('phrase'))
        if T is None:
            rec['why'] = 'no track'; rows.append(rec); continue
        on = onset(T, O, fps, t1); rec['td'] = td
        if isinstance(on, str):
            rec['why'] = '(c) ' + on; rows.append(rec); continue
        rec.update({k: on[k] for k in ('edge', 'D', 'ratio', 's_first')})
        if not on['ok']:
            rec['why'] = '(c) onset fail'; rows.append(rec); continue
        rad = abs(d1 - d0); alb = 2 * rad / t1 ** 2
        mf = raw.loc[i, 'meas_f']; fac = 1.0
        if mf == mf and mf > 0 and rad > 0:
            lat = abs(on['D']) * (0.5 * (d0 + d1)) / (mf * on['W']); fac = min(1.5, float(np.hypot(1, lat / rad))); rec['lat'] = lat
        est = alb * fac; rec.update(rad=rad, alb=alb, fac=fac, est=est)
        if a >= alb:
            rec['why'] = 'answer>=a_lb'; rows.append(rec); continue
        cs = D.loc[i, 'cs']
        if cs == cs and cs > 0 and max(est, cs) / min(est, cs) <= 2:
            new = max(np.sqrt(est * cs), alb); how = 'gm'
        else:
            new = est; how = 'geo'
        why = (f'{TAG}: from-rest onset ({on["edge"]} ratio {on["ratio"]:.2f}); a_lb=2*|{d1:.3f}-{d0:.3f}|/{t1:g}^2={alb:.2f}'
               f' x lat {fac:.2f} -> {new:.2f} ({how}); was {a:.3g} {src}')
        rec.update(new=new, why=why); rows.append(rec); out[str(i)] = {'value': float(new), 'why': why}
    return out, pd.DataFrame(rows)


def validate():
    """label-free check: same estimator on PRIOR objects with stated accel / speed-at-t priors and own keys at 0."""
    res = []
    for split in ('test', 'val'):
        P = {}
        for l in open(R + f'/geo/out/parse_{split}_v2.jsonl'):
            j = json.loads(l); P[j['id']] = j
        raw = pd.read_csv(R + f'/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
        seen = set()
        for i, j in P.items():
            if str(raw.loc[i, 'video_source']) != 'simulation':
                continue
            pk = [norm(k) for k in (j.get('prior_dkeys') or [])]
            if len(pk) != 1 or not j.get('priors'):
                continue
            pr = j['priors'][0]
            if pr.get('kind') not in ('accel', 'speed') or pr.get('value_si') is None:
                continue
            if pr['kind'] == 'speed' and pr.get('t') is None:
                continue
            ks = sorted([(float(e['t']), float(e['d'])) for e in j.get('depth', []) if e.get('t') is not None and norm(e['key']) == pk[0]])
            k0 = [d for tt, d in ks if abs(tt) < 1e-9]; kp = [(tt, d) for tt, d in ks if tt > 1e-9]
            if not k0 or not kp:
                continue
            key = (split, j['video_id'], pr['raw'])
            if key in seen:
                continue
            seen.add(key)
            t1, d1 = kp[0]; d0 = k0[0]; fps = float(raw.loc[i, 'fps'])
            td, T, O = load_track(split, j['video_id'], (j.get('prior_obj') or {}).get('phrase'))
            if T is None:
                continue
            on = onset(T, O, fps, t1)
            r = dict(split=split, id=i, vid=j['video_id'], prior=pr['raw'], kind=pr['kind'], pt=pr.get('t'), t1=t1, stated=pr['value_si'])
            if isinstance(on, str):
                r['on'] = on; res.append(r); continue
            rad = abs(d1 - d0); mf = raw.loc[i, 'meas_f']
            lat = abs(on['D']) * (0.5 * (d0 + d1)) / (mf * on['W']) if (mf == mf and mf > 0) else np.nan
            fac = min(1.5, np.hypot(1, lat / rad)) if (rad > 0 and lat == lat) else 1.0
            a_lb = 2 * rad / t1 ** 2
            mult = 1.0 if pr['kind'] == 'accel' else float(pr['t'])  # speed at pt from rest: v = a*pt
            r.update(lb=a_lb * mult, est=a_lb * fac * mult, rad=rad, lat=lat,
                     edge=on['edge'], ratio=on['ratio'], s_first=on['s_first'], ok=on['ok'], fac=fac)
            res.append(r)
    return pd.DataFrame(res)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200); pd.set_option('display.max_rows', 500)
    allx = []
    for split, dn in (('test', 'sub_xt_Nzz15_detail.csv'), ('val', 'sub_xt_Nzz15_valdetail.csv')):
        out, X = run(split, dn)
        json.dump(out, open(R + f'/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
        print(split, 'changed', len(out)); allx.append(X)
    X = pd.concat(allx); print(X.round(3).to_string())
    V = validate(); print('\nVALIDATION on prior objects (stated value vs estimator):')
    if len(V):
        print(V.round(3).to_string())
        sys.path.insert(0, R + '/caw'); import analyze_caw as A
        if 'ok' in V.columns:
            G = V[V.ok == True]
            if len(G):
                print('onset-pass n', len(G), 'MRA est', A.row_mra(G.est.values, G.stated.values).mean().round(3),
                      'MRA lb', A.row_mra(G.lb.values, G.stated.values).mean().round(3),
                      'lb<=stated', (G.lb <= G.stated * 1.02).mean().round(3), 'est/stated median', (G.est / G.stated).median().round(3))
