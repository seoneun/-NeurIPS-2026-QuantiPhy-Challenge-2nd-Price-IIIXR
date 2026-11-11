# rul126: D2 analogue of rul114.
# SIM D2 'speed at t' / 'accel at t' rows (mode 2d_ratio, dynamic prior) whose target box is cut by the frame
# border (>=30% of frames in [0,t]) -> geo's centre/box measure is biased. If the motion-facing (leading) box edge
# is uncut on >=20 frames of [0,t] and follows x = x0 + k (f-f0)^2 with |f0/fps|<=0.05 s, rms<=1 px, and the
# from-rest onset holds (s(t/2)/s(t) in [0.15,0.35], s(first)<=0.1 s(t)), then a = 2k fps^2 mpp and v(t)=a(t-t0).
# Ruler re-check: the same leading-edge fit on the PRIOR object's own track, x mpp, must reproduce the stated prior
# (accel, or speed at its t) within 5%. Gate (f): skip answers set by kin / rul289 / gravity.
# Replace only if |old/new_raw - 1| > 15%; final value follows the pipeline fusion (new geo vs CaW: gm within 2x,
# else geo, mode 2d_ratio is strong).
import sys, json, os, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul126'
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz18'


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


def lead_fit(T, O, fps, tw):
    """leading-edge quadratic fit on frames with time in [0, tw]. returns dict (with 'fail' key on failure)"""
    W = float(T['W']); H = float(T['H'])
    fr = []; bx = []
    for f, st in zip(T['fidx'], O):
        if f / fps > tw + 0.5 / fps:
            break
        if st is None or st.get('st') is None:
            continue
        fr.append(f); bx.append(st['st']['box'])
    fr = np.array(fr, float); bx = np.array(bx, float)
    if len(fr) < 6:
        return dict(fail='few frames')
    touch = ((bx[:, 0] <= 2) | (bx[:, 1] <= 2) | (bx[:, 2] >= W - 2) | (bx[:, 3] >= H - 2))
    res = dict(n=len(fr), border_frac=float(touch.mean()))
    cutm = [bx[:, 0] <= 2, bx[:, 1] <= 2, bx[:, 2] >= W - 2, bx[:, 3] >= H - 2]
    best = None
    for e, sgn in ((0, -1), (1, -1), (2, 1), (3, 1)):
        ok = ~cutm[e]
        if ok.sum() < 6:
            continue
        x = bx[ok, e]; f = fr[ok]
        D = x[-1] - x[0]
        if np.sign(D) != sgn:
            continue
        if best is None or abs(D) > abs(best[1]):
            best = (e, D, ok)
    if best is None:
        res['fail'] = 'no uncut leading edge'; return res
    e, D, ok = best
    f = fr[ok]; x = bx[ok, e]
    res.update(edge='x1 y1 x2 y2'.split()[e], n_uncut=int(ok.sum()), D=float(D))
    if ok.sum() < 20:
        res['fail'] = f'uncut {ok.sum()}<20'; return res
    c2, c1, c0 = np.polyfit(f, x, 2)
    rms = float(np.sqrt(np.mean((np.polyval([c2, c1, c0], f) - x) ** 2)))
    f0 = -c1 / (2 * c2) if c2 != 0 else np.nan
    res.update(k=float(abs(c2)), f0=float(f0), t0=float(f0 / fps), rms=rms, curv_sign=float(np.sign(c2) * sgn_of(e)))
    # onset (raw, median within +-1 frame) relative to the first frame of the window

    def at(tt):
        m = np.abs(fr / fps - tt) <= 1.01 / fps
        m = m & ok
        return np.median(bx[m, e]) if m.sum() else np.nan
    t1 = min(tw, fr[ok].max() / fps)
    x0 = at(0.0); s1 = abs(at(t1) - x0); sh = abs(at(t1 / 2) - x0)
    first = fr[fr > 0.5][0] / fps if (fr > 0.5).any() else np.nan
    sf = abs(at(first) - x0)
    res.update(t1=float(t1), ratio=float(sh / s1) if s1 > 0 else np.nan, s_first=float(sf / s1) if s1 > 0 else np.nan)
    fails = []
    if not (abs(res['t0']) <= 0.05):
        fails.append(f't0 {res["t0"]:.3f}')
    if not (rms <= 1.0):
        fails.append(f'rms {rms:.2f}')
    if not (res['curv_sign'] > 0):
        fails.append('decelerating')
    if not (0.15 <= res['ratio'] <= 0.35):
        fails.append(f'ratio {res["ratio"]:.2f}')
    if not (res['s_first'] <= 0.1):
        fails.append(f's_first {res["s_first"]:.2f}')
    if fails:
        res['fail'] = ','.join(fails)
    return res


def sgn_of(e):
    # leading edge moves in sign sgn; accelerating means c2 has that same sign
    return -1.0 if e in (0, 1) else 1.0


def run(split, detname):
    P = {}
    for l in open(R + f'/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    raw = pd.read_csv(R + f'/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    D = pd.read_csv(R + '/submissions/' + detname).set_index('Unnamed: 0')
    out = {}; rows = []
    for i, j in P.items():
        if i not in D.index or D.loc[i, 'cat'] != 'D2':
            continue
        if str(raw.loc[i, 'video_source']) != 'simulation':  # (a)
            continue
        if ('mode' in D.columns and str(D.loc[i, 'mode']) != '2d_ratio') or str(raw.loc[i, 'meas_mode']) != '2d_ratio':
            continue
        q = j['q']
        if q.get('kind') not in ('speed', 'accel') or q.get('t') is None or q.get('t1') is not None or q.get('t2') is not None:
            continue
        pri = [p for p in (j.get('priors') or []) if p.get('kind') in ('speed', 'accel') and p.get('value_si')]
        if len(pri) != 1:
            continue
        pr = pri[0]
        if pr['kind'] == 'speed' and pr.get('t') is None:
            continue
        t = float(q['t']); a_old = float(D.loc[i, 'answer']); src = str(D.loc[i, 'answer_source'])
        rec = dict(split=split, id=i, vid=j['video_id'], kind=q['kind'], t=t, tgt=(j.get('target') or {}).get('phrase'),
                   prior=pr['raw'], ans=a_old, src=src, g=D.loc[i, 'g'], cs=D.loc[i, 'cs'])
        if 'y' in D.columns:
            rec['y'] = D.loc[i, 'y']
        # (f)
        if 'kin' in src or 'rul289' in src or 'grav' in src or abs(a_old - 9.8) < 1e-6:
            rec['why'] = '(f) kin/rul289/grav'; rows.append(rec); continue
        fps = float(raw.loc[i, 'fps'])  # CSV fps, never container fps
        mpp = float(raw.loc[i, 'meas_mpp'])
        if not (mpp == mpp and mpp > 0):
            rec['why'] = 'no mpp'; rows.append(rec); continue
        tgt = (j.get('target') or {}).get('phrase'); pob = (j.get('prior_obj') or {}).get('phrase')
        if not tgt or not pob or tgt == pob:
            rec['why'] = 'target==prior/none'; rows.append(rec); continue
        td, T, O = load_track(split, j['video_id'], tgt)
        if T is None:
            rec['why'] = 'no track'; rows.append(rec); continue
        F = lead_fit(T, O, fps, t); rec['td'] = td
        rec.update({k: F.get(k) for k in ('border_frac', 'edge', 'n_uncut', 'k', 't0', 'rms', 'ratio', 's_first')})
        if F.get('border_frac', 0) < 0.30:  # (b)
            rec['why'] = '(b) border<30%'; rows.append(rec); continue
        if 'fail' in F:
            rec['why'] = '(c/d) ' + F['fail']; rows.append(rec); continue
        a_new = 2 * F['k'] * fps ** 2 * mpp
        # (e) ruler re-check on the prior's own track
        # window: a stated constant accel holds over the prior's whole track; a speed prior is checked on [0, its t]
        pt = float(pr['t']) if pr['kind'] == 'speed' else 1e9
        tdp, Tp, Op = load_track(split, j['video_id'], pob)
        if Tp is None:
            rec['why'] = '(e) no prior track'; rows.append(rec); continue
        Fp = lead_fit(Tp, Op, fps, pt)
        if 'k' not in Fp:
            rec['why'] = '(e) prior fit ' + Fp.get('fail', '?'); rows.append(rec); continue
        ap = 2 * Fp['k'] * fps ** 2 * mpp
        chk = ap if pr['kind'] == 'accel' else ap * (pt - Fp['t0'])
        Fq = lead_fit(Tp, Op, fps, t)  # diagnostic only: prior fit on the row's own window
        rec.update(prior_chk=chk, prior_stated=pr['value_si'], prior_fail=Fp.get('fail'), prior_rms=Fp.get('rms'),
                   prior_chk_rowwin=(2 * Fq['k'] * fps ** 2 * mpp) if 'k' in Fq else np.nan)
        # the prior fit must itself be a from-rest accelerating quadratic (t0, curvature); its value is gated at 5%
        pf = [x for x in str(Fp.get('fail', '')).split(',') if x and not x.startswith(('ratio', 's_first', 'rms'))]
        if pf:
            rec['why'] = '(e) prior fit ' + ','.join(pf); rows.append(rec); continue
        if abs(chk / pr['value_si'] - 1) > 0.05:
            rec['why'] = f'(e) ruler {chk:.3f} vs {pr["value_si"]}'; rows.append(rec); continue
        new_raw = a_new if q['kind'] == 'accel' else a_new * (t - F['t0'])
        rec['a_new'] = a_new; rec['new_raw'] = new_raw
        if abs(a_old / new_raw - 1) <= 0.15:
            rec['why'] = 'within 15%'; rows.append(rec); continue
        cs = D.loc[i, 'cs']
        if cs == cs and cs > 0 and max(new_raw, cs) / min(new_raw, cs) <= 2:
            new = float(np.sqrt(new_raw * cs)); how = 'gm'
        else:
            new = float(new_raw); how = 'geo'
        why = (f'{TAG}: border-cut {F["border_frac"]:.0%}, leading {F["edge"]} quadratic from rest (n {F["n_uncut"]}, t0 {F["t0"]:+.3f}s, '
               f'rms {F["rms"]:.2f}px, ratio {F["ratio"]:.2f}) a={a_new:.2f}; prior ruler {chk:.3f} vs {pr["value_si"]}; '
               f'{q["kind"]} {new_raw:.2f} -> {new:.2f} ({how}); was {a_old:.3g} {src}')
        rec.update(new=new, how=how, why=why); rows.append(rec); out[str(i)] = {'value': new, 'why': why}
    return out, pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 500)
    allx = []
    for split, dn in (('test', SUB + '_detail.csv'), ('val', SUB + '_valdetail.csv')):
        out, X = run(split, dn)
        json.dump(out, open(R + f'/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
        print(split, 'changed', len(out)); allx.append(X)
        for k, v in out.items():
            print(' ', split, k, v['why'])
    X = pd.concat(allx)
    print(X.drop(columns=['prior']).round(3).to_string())
