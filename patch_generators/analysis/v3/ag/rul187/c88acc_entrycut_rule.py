# rul187: accel-at-t rows in 2D modes where the target is ENTERING the frame at t (box touches the border on
# the motion axis at t) and the first uncut, non-overlap frame t_u lies at most 0.2 s after t, so no uncut frames exist
# before t (rul154 cannot fire).  Grow a straight segment forward from t_u (per-frame +/-2 chord within 8 deg of the
# chord at t_u, gaps < 0.15 s, uncut & non-overlap frames); require >=0.8 s, >=15 frames, perp rms <=1 px, quad rms <=4 px.
# Cubic fit of arc length s(t) -> a_c = |s''(t_u)| (no extrapolation below t_u).  Value = a_c*meas_mpp*k, k = answer/g
# when the answer is an upstream patch that rescaled geo (0.5..1.5), else 1 (then standard fusion vs CaW).
# Fires when |ln(new/answer)| > ln 1.5.  Copies to identical-question twin rows (plain/_segmented) of the same base video.
import sys, json, os, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/analysis/v3/ag/rul154')
import c80uncutacc_rule as C8
TAG = 'rul187'
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz27'
MODES = ('2d_ratio', 'vlm_2d_ratio')


def base(v):
    return str(v).replace('_segmented', '')


def segment(rows, fps, iu):
    f = np.array([r[0] for r in rows], float); B = np.array([r[1] for r in rows], float)
    C = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]
    n = len(rows)

    def chord(k):
        a = max(k - 2, iu); b = min(k + 2, n - 1)
        d = C[b] - C[a]; nr = np.linalg.norm(d)
        return d / nr if nr > 1e-6 else None
    u0 = chord(iu)
    if u0 is None:
        return dict(fail='no chord')
    idx = [iu]
    for k in range(iu + 1, n):
        if (f[k] - f[idx[-1]]) / fps >= 0.15:
            break
        r = rows[k]
        if r[2] or r[3] or r[4]:
            break
        uk = chord(k)
        if uk is None or np.degrees(np.arccos(np.clip(np.dot(uk, u0), -1, 1))) > 8:
            break
        idx.append(k)
    idx = np.array(idx)
    tt = (f[idx] - f[iu]) / fps; P = C[idx] - C[iu]
    s = P @ u0; nrm = np.array([-u0[1], u0[0]]); p = P @ nrm
    res = dict(nseg=len(idx), dur=float(tt[-1]), ax='x' if abs(u0[0]) >= abs(u0[1]) else 'y',
               ux=float(abs(u0[0])), uy=float(abs(u0[1])))
    if len(idx) < 6:
        res['fail'] = 'short'; return res
    res['prms'] = float(np.sqrt(np.mean((p - np.polyval(np.polyfit(s, p, 1), s)) ** 2)))
    res['qrms'] = float(np.sqrt(np.mean((s - np.polyval(np.polyfit(tt, s, 2), tt)) ** 2)))
    c3 = np.polyfit(tt, s, 3)
    res['a_c'] = float(abs(2 * c3[1])); res['a_q'] = float(abs(2 * np.polyfit(tt, s, 2)[0]))
    return res


def run(split, detname):
    P = {}
    for l in open(R + f'/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    raw = pd.read_csv(R + f'/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
    D = pd.read_csv(R + '/submissions/' + detname).set_index('Unnamed: 0')
    out = {}; recs = []
    for i, j in P.items():
        if i not in D.index:
            continue
        q = j['q']
        if q.get('kind') != 'accel' or q.get('t') is None or q.get('t1') is not None or q.get('t2') is not None:
            continue
        md = str(D.loc[i, 'mode']) if 'mode' in D.columns else str(raw.loc[i, 'meas_mode'])
        if md not in MODES:
            continue
        src = str(D.loc[i, 'answer_source']); ans = float(D.loc[i, 'answer']); g = D.loc[i, 'g']; cs = D.loc[i, 'cs']
        rec = dict(split=split, id=i, vid=j['video_id'], cat=D.loc[i, 'cat'], mode=md, t=q['t'], src=src,
                   ans=ans, g=g, cs=cs)
        if 'y' in D.columns:
            rec['y'] = D.loc[i, 'y']
        recs.append(rec)
        fps = float(raw.loc[i, 'fps']); mpp = float(raw.loc[i, 'meas_mpp'])
        if not (mpp == mpp and mpp > 0):
            rec['why'] = 'no mpp'; continue
        tgt = (j.get('target') or {}).get('phrase')
        L = C8.load_boxes(split, j['video_id'], raw.loc[i, 'meas_tracks'], tgt)
        if L is None:
            rec['why'] = 'no track'; continue
        W, H, fidx, objs, tn = L
        C8.W_, C8.H_ = W, H
        rows = C8.frames(W, H, fidx, objs, tn)
        if not rows:
            rec['why'] = 'empty'; continue
        t = float(q['t'])
        unc = [k for k, r in enumerate(rows) if not (r[2] or r[3] or r[4])]
        if not unc:
            rec['why'] = 'no uncut'; continue
        iu = unc[0]; tu = rows[iu][0] / fps; rec['tu'] = round(tu, 3)
        if not (0 <= tu - t <= 0.2 + 1e-6):
            rec['why'] = f'tu-t {tu - t:.2f}'; continue
        S = segment(rows, fps, iu)
        rec.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()})
        if 'fail' in S:
            rec['why'] = S['fail']; continue
        kt = int(np.argmin([abs(r[0] / fps - t) for r in rows]))
        if abs(rows[kt][0] / fps - t) > 1.5 / fps:
            rec['why'] = 'not visible at t'; continue
        # motion axis = any image axis carrying >= half of the unit direction (|u_axis| >= sin 30 deg): entering through
        # an edge needs motion normal to that edge; a diagonal entry is cut on both axes' edges equally validly
        cut_t = (rows[kt][2] and S['ux'] >= 0.5) or (rows[kt][3] and S['uy'] >= 0.5)
        fails = []
        if not cut_t: fails.append('nocut at t')
        if S['dur'] < 0.8 or S['nseg'] < 15: fails.append(f'seg {S["dur"]:.2f}s/{S["nseg"]}')
        if S['prms'] > 1: fails.append(f'prms {S["prms"]:.2f}')
        if S['qrms'] > 4: fails.append(f'qrms {S["qrms"]:.2f}')
        k = 1.0
        if src.startswith('patch') and g == g and g > 0 and 0.5 <= ans / g <= 1.5:
            k = ans / g
        gn = S['a_c'] * mpp * k; rec['g_new'] = gn; rec['k'] = round(k, 3)
        if k != 1.0:
            new = gn; how = 'k-scaled'
        elif cs == cs and cs > 0 and max(gn, cs) / min(gn, cs) <= 2:
            new = float(np.sqrt(gn * cs)); how = 'gm'
        else:
            new = gn; how = 'geo'
        rec['new'] = new
        if abs(np.log(new / ans)) <= np.log(1.5): fails.append('small')
        if fails:
            rec['why'] = ';'.join(fails); continue
        why = (f'{TAG}: target entering (cut at t={t}), first uncut t_u={tu:.2f}; straight seg {S["nseg"]}fr/{S["dur"]:.2f}s '
               f'(prms {S["prms"]:.2f}, qrms {S["qrms"]:.2f}); cubic a(t_u)={S["a_c"]:.0f}px/s2 *mpp*k({k:.3f}) -> {new:.3g} ({how}); was {ans:.3g} {src}')
        rec['why'] = 'OK'; out[str(i)] = {'value': new, 'why': why}
        for k2 in D.index:
            if k2 != i and D.loc[k2, 'question'] == D.loc[i, 'question'] and base(D.loc[k2, 'video_id']) == base(j['video_id']):
                out[str(k2)] = {'value': new, 'why': why + f' (copy from {i}, twin)'}
    return out, pd.DataFrame(recs)


if __name__ == '__main__':
    pd.set_option('display.width', 400); pd.set_option('display.max_rows', 500)
    allx = []
    for split, dn in (('test', SUB + '_detail.csv'), ('val', SUB + '_valdetail.csv')):
        out, X = run(split, dn)
        json.dump(out, open(R + f'/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
        print(split, 'changed', len(out)); allx.append(X)
        for k, v in out.items():
            print(' ', split, k, round(v['value'], 4), v['why'])
    X = pd.concat(allx)
    X.to_csv(R + f'/cache/tmp/ag_{TAG}_recs.csv')
    w = X.why.astype(str)
    print(X[(X.split == 'test') & (X.id.astype(str) == '858')].T.to_string())
    print(X[~w.str.startswith(('tu-t', 'no ', 'empty'))].round(3).to_string())
    print(w.str.split(' ').str[0].value_counts().to_string())
