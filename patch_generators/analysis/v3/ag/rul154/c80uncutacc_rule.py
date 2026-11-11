# rul154: 'accel at t' (single t) rows answered from geo (dis_geo / geo_*), mode 2d_ratio, whose TARGET box
# is cut by the frame border on the motion axis inside t +/- HALF_A: geo's centre fit mixes cut frames (the box centre
# of a half-visible object moves at a different rate) -> refit the centre quadratic on UNCUT frames only.
# Gates: (1) >=2 border-touch frames on the motion axis in the window; (2) enough uncut frames on each side of t
# (uncut = no border contact on the motion axis and no IoU>0 with any other tracked box); (3) straight path
# (heading change < 15 deg); (5) refit stable over HALF_A in {0.3,0.4,0.5} (max/min <= 1.3); (6) free-edge rebuild
# (centre = free edge -/+ L/2, L = median uncut extent with CV <= 3%) within 1.3x; (7) slope witness: own speed
# sibling rows recomputed on uncut frames (>=2 within +/-0.6 s) else local linear speeds at t-0.3 / t+0.3 on uncut
# frames; within 1.3x; (8) |ln(a_u/a_geo_px)| > 0.25.  Value = a_u * meas_mpp; standard fusion vs CaW (gm within 2x,
# else geo).  Copies to an identical-question _segmented twin.
import sys, json, os, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul154'
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz24'
MIN_SIDE = int(os.environ.get('C80_MIN_SIDE', '5'))
HALF_A = 0.4
BTOL = 2.0
TD = {'v1': '', 'v2': '_v2', 'v3': '_v3'}


def tdir(split, mt):
    mt = str(mt)
    suf = TD.get(mt, '_' + mt)
    return R + f'/geo/out/tracks_{split}{suf}'


def load_boxes(split, vid, mt, phrase):
    """returns W,H,fidx,{name: [box or None]}, target name"""
    p = tdir(split, mt) + f'/{vid}.json'
    if not os.path.exists(p):
        return None
    T = json.load(open(p)); W = float(T['W']); H = float(T['H']); fidx = list(T['fidx'])
    objs = {}; tname = None
    if 'objs' in T:
        for k, o in T['objs'].items():
            objs[k] = [(None if (s is None or s.get('st') is None) else s['st']['box']) for s in o]
        for e in T['queries']:
            if e['key'].split('|')[0] == phrase and e.get('obj') is not None and str(e['obj']) in objs:
                tname = str(e['obj']); break
    else:
        for n, e in enumerate(T['queries']):
            objs[str(n)] = [(None if (s is None or s.get('st') is None) else s['st']['box']) for s in e['frames']]
            if tname is None and e['key'].split('|')[0] == phrase:
                tname = str(n)
    if tname is None:
        return None
    return W, H, np.array(fidx, float), objs, tname


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def frames(W, H, fidx, objs, tn):
    """per frame: time-index, box, cut flags per axis, overlap flag"""
    rows = []
    for n, f in enumerate(fidx):
        b = objs[tn][n]
        if b is None:
            continue
        b = [float(x) for x in b]
        cx = (b[0] <= BTOL) or (b[2] >= W - 1 - BTOL)
        cy = (b[1] <= BTOL) or (b[3] >= H - 1 - BTOL)
        ov = False
        for k, o in objs.items():
            if k == tn or o[n] is None:
                continue
            ob = [float(x) for x in o[n]]
            # ignore duplicates of the same instance (another query on the same object)
            inter = iou(b, ob)
            if inter <= 0:
                continue
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter
            if inter / max(ua, 1e-6) > 0.7:
                continue
            ov = True; break
        rows.append((f, b, cx, cy, ov))
    return rows


def qfit(ts, xs):
    c = np.polyfit(ts, xs, 2); return 2 * c[0]


def analyse(rows, fps, t, half=HALF_A):
    w = [r for r in rows if abs(r[0] / fps - t) <= half + 1e-6]
    if len(w) < 5:
        return dict(fail='few window frames')
    f = np.array([r[0] for r in w]); B = np.array([r[1] for r in w])
    C = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]
    dx = abs(C[-1, 0] - C[0, 0]); dy = abs(C[-1, 1] - C[0, 1])
    ax = 0 if dx >= dy else 1
    cut = np.array([r[2] if ax == 0 else r[3] for r in w]); ov = np.array([r[4] for r in w])
    unc = (~cut) & (~ov)
    tt = f / fps - t
    res = dict(ax='xy'[ax], n_w=len(w), n_cut=int(cut.sum()), n_ov=int(ov.sum()),
               n_lo=int((unc & (tt < 0)).sum()), n_hi=int((unc & (tt > 0)).sum()), n_unc=int(unc.sum()))
    if unc.sum() < 5:
        res['fail'] = 'few uncut'; return res
    # heading change between first and second half of the uncut frames
    tu = tt[unc]; Cu = C[unc]
    lo = tu <= np.median(tu); hi = tu >= np.median(tu)
    def head(m):
        if m.sum() < 2:
            return None
        return np.array([np.polyfit(tu[m], Cu[m, 0], 1)[0], np.polyfit(tu[m], Cu[m, 1], 1)[0]])
    h1, h2 = head(lo), head(hi)
    if h1 is not None and h2 is not None:
        cosv = np.dot(h1, h2) / max(np.linalg.norm(h1) * np.linalg.norm(h2), 1e-9)
        res['dhead'] = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    # uncut refit (2D vector accel, motion axis dominates)
    a_vec = np.array([qfit(tu, Cu[:, 0]), qfit(tu, Cu[:, 1])])
    res['a_u'] = float(np.linalg.norm(a_vec)) * 1.0
    res['a_u_ax'] = float(a_vec[ax])
    # free-edge rebuild on all window frames with at least one free edge on the motion axis
    e0, e1 = (0, 2) if ax == 0 else (1, 3)
    L = B[unc, e1] - B[unc, e0]
    res['Lcv'] = float(np.std(L) / max(np.mean(L), 1e-6)); Lm = float(np.median(L))
    lim = (W_ if ax == 0 else H_) - 1 - BTOL
    lo_free = B[:, e0] > BTOL; hi_free = B[:, e1] < lim
    cen = np.where(lo_free & hi_free, (B[:, e0] + B[:, e1]) / 2, np.where(lo_free, B[:, e0] + Lm / 2, np.where(hi_free, B[:, e1] - Lm / 2, np.nan)))
    m = np.isfinite(cen) & ~ov
    res['a_fe'] = float(abs(qfit(tt[m], cen[m]))) if m.sum() >= 5 else np.nan
    return res


def lin_speed(rows, fps, t0, half, ax):
    w = [r for r in rows if abs(r[0] / fps - t0) <= half + 1e-6 and not (r[2] if ax == 0 else r[3]) and not r[4]]
    if len(w) < 3:
        return None
    tt = np.array([r[0] / fps for r in w]); B = np.array([r[1] for r in w])
    C = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]
    # speed at the mean time of the uncut frames actually used (a one-sided window shifts the effective time)
    return float(tt.mean()), float(np.hypot(np.polyfit(tt, C[:, 0], 1)[0], np.polyfit(tt, C[:, 1], 1)[0]))


W_ = H_ = None


def run(split, detname):
    global W_, H_
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
        if ('mode' in D.columns and str(D.loc[i, 'mode']) != '2d_ratio') or str(raw.loc[i, 'meas_mode']) != '2d_ratio':
            continue
        src = str(D.loc[i, 'answer_source'])
        rec = dict(split=split, id=i, vid=j['video_id'], cat=D.loc[i, 'cat'], t=q['t'], src=src, ans=D.loc[i, 'answer'],
                   cs=D.loc[i, 'cs'], tracks=raw.loc[i, 'meas_tracks'])
        if 'y' in D.columns:
            rec['y'] = D.loc[i, 'y']
        recs.append(rec)
        if not (src == 'dis_geo' or src.startswith('geo')):
            rec['why'] = 'src ' + src; continue
        fps = float(raw.loc[i, 'fps']); mpp = float(raw.loc[i, 'meas_mpp']); g0 = float(raw.loc[i, 'geo_value'])
        if not (mpp == mpp and mpp > 0 and g0 == g0 and g0 > 0):
            rec['why'] = 'no mpp/geo'; continue
        tgt = (j.get('target') or {}).get('phrase')
        L = load_boxes(split, j['video_id'], raw.loc[i, 'meas_tracks'], tgt)
        if L is None:
            rec['why'] = 'no track'; continue
        W_, H_, fidx, objs, tn = L
        rows = frames(W_, H_, fidx, objs, tn)
        t = float(q['t'])
        A = analyse(rows, fps, t)
        rec.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.items()})
        a_geo_px = g0 / mpp; rec['a_geo_px'] = a_geo_px
        if 'fail' in A:
            rec['why'] = A['fail']; continue
        a_u = A['a_u']; rec['g_new'] = a_u * mpp
        fails = []
        if A['n_cut'] < 2: fails.append('(1) cut<2')
        if min(A['n_lo'], A['n_hi']) < MIN_SIDE: fails.append(f'(2) side {A["n_lo"]}/{A["n_hi"]}')
        if not (A.get('dhead', 99) < 15): fails.append(f'(3) head {A.get("dhead")}')
        st = []
        for h in (0.3, 0.4, 0.5):
            Ah = analyse(rows, fps, t, h)
            st.append(Ah.get('a_u', np.nan))
        rec['stab'] = [round(x, 1) for x in st]
        if not all(np.isfinite(st)) or max(st) / max(min(st), 1e-9) > 1.3: fails.append('(5) unstable')
        if not (A['Lcv'] <= 0.03): fails.append(f'(6) Lcv {A["Lcv"]:.3f}')
        afe = A['a_fe']
        if not (afe == afe and max(afe, a_u) / max(min(afe, a_u), 1e-9) <= 1.3): fails.append(f'(6) fe {afe}')
        # (7) witness
        ax = 0 if A['ax'] == 'x' else 1
        wit = None; wsrc = None
        sib = []
        for k2, j2 in P.items():
            if j2['video_id'] != j['video_id'] or k2 == i:
                continue
            q2 = j2['q']
            if q2.get('kind') == 'speed' and q2.get('t') is not None and (j2.get('target') or {}).get('phrase') == tgt \
                    and abs(float(q2['t']) - t) <= 0.6:
                v = lin_speed(rows, fps, float(q2['t']), 0.25, ax)
                if v is not None:
                    sib.append((float(q2['t']),) + v)
        rec['sibs'] = [tuple(round(x, 2) for x in s_) for s_ in sib]
        tm_ = sorted(set(round(s_[1], 3) for s_ in sib))
        if len(sib) >= 2 and len(tm_) >= 2 and tm_[-1] - tm_[0] >= 0.15:
            _, ts_, vs_ = np.array(sib).T; wit = abs(np.polyfit(ts_, vs_, 1)[0]); wsrc = f'sib{len(sib)}'
        else:
            v1 = lin_speed(rows, fps, t - 0.3, 0.25, ax); v2 = lin_speed(rows, fps, t + 0.3, 0.25, ax)
            if v1 is not None and v2 is not None and v2[0] - v1[0] >= 0.15:
                wit = abs(v2[1] - v1[1]) / (v2[0] - v1[0]); wsrc = 'local'
        rec['wit'] = wit; rec['wsrc'] = wsrc
        if wit is None or max(wit, a_u) / max(min(wit, a_u), 1e-9) > 1.3: fails.append(f'(7) wit {wit}')
        rec['lnr'] = float(np.log(a_u / a_geo_px))
        if abs(rec['lnr']) <= 0.25: fails.append('(8) small')
        if fails:
            rec['why'] = ';'.join(fails); continue
        gn = a_u * mpp; cs = D.loc[i, 'cs']
        if cs == cs and cs > 0 and max(gn, cs) / min(gn, cs) <= 2:
            new = float(np.sqrt(gn * cs)); how = 'gm'
        else:
            new = float(gn); how = 'geo'
        why = (f'{TAG}: target cut on {A["ax"]} in {A["n_cut"]}/{A["n_w"]} window frames; uncut refit ({A["n_lo"]}+{A["n_hi"]} fr) '
               f'a={gn:.3g} (stab {st[0]:.0f}/{st[1]:.0f}/{st[2]:.0f}px, free-edge {afe:.0f}px, {wsrc} witness {wit:.0f}px) '
               f'-> {new:.3g} ({how}); was {D.loc[i, "answer"]:.3g} {src}')
        rec.update(new=new, how=how, why='OK'); out[str(i)] = {'value': new, 'why': why}
        # segmented / plain twin with the identical question
        for k2 in D.index:
            if k2 != i and D.loc[k2, 'question'] == D.loc[i, 'question'] and \
                    str(D.loc[k2, 'video_id']) in (j['video_id'] + '_segmented',) and not str(D.loc[k2, 'answer_source']).startswith('patch:'):
                out[str(k2)] = {'value': new, 'why': why + f' (copy from {i}, _segmented twin)'}
    return out, pd.DataFrame(recs)


if __name__ == '__main__':
    pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 50); pd.set_option('display.max_rows', 500)
    allx = []
    for split, dn in (('test', SUB + '_detail.csv'), ('val', SUB + '_valdetail.csv')):
        out, X = run(split, dn)
        json.dump(out, open(R + f'/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
        print(split, 'changed', len(out)); allx.append(X)
        for k, v in out.items():
            print(' ', split, k, round(v['value'], 4), v['why'])
    X = pd.concat(allx)
    print(X.round(3).to_string())
