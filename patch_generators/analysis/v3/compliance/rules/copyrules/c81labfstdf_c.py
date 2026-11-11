"""c81labfstdf_c: copy of analysis/v3/ag/rul156/rul156.py (production rul156) with one added fallback.
Production rule (unchanged): one-sided lab focal check from an in-frame standard ball at its own stated (static) depth.
Gates: captured_* video, untimed dist2, meas_mode 3d_calib, both endpoints with a static depth key, a standard ball
(basketball 0.24 m / tennis ball 0.067 m) with its own static depth key and an uncut track (>=80% frames);
f_std = median(max(w,h)) * Z_ball / D.  Fires only when f_cal/f_std > 1.4: geometry re-done with f_std,
3D centroid distance (geo-only), floored at |d1-d2|.

Change (2026-10-17, copyrules helper labd3): production drops a row when an endpoint track in the row's meas_tracks set
(geo/out/tracks_<split>_<meas_tracks>/<video>.json) moves (centre std > 10 px) or shares its object id with a moving
phrase.  On captured_0053x (test row 2828) the v2 detector gave 'book' and 'trash bin' the same object, which follows the
moving object (cstd 82.3 px), so the row was dropped; the v3nt tracker output for the SAME video has a static 'book'
track at the place where the plain video's v2 book track is.  The fixed rule applies the production filter per endpoint:
an endpoint whose meas_tracks track fails (moving, or sharing its object id with a moving phrase) is replaced by the
v3nt track of the same phrase in the row's own video when that v3nt track passes the same filter (and the frame size agrees
within 2 px).  f_std (ball focal) and the ratio gate stay on meas_tracks.  A first version that redid the whole row on v3nt
did not fire on 2828: v3nt maps the moving 'trash bin' phrase onto the static white cup, so the white cup fails the
shared-object filter there; the v2 white cup track passes.  The fallback reads only the row's own video outputs and
applies to every row (plain and copy) that production drops at the filter step (test and val: only row 2828, checked
2026-10-17).  Everything else is the production code.
Run only through analysis/v3/compliance/rerun_shadow.sh (it writes cache/tmp/ag_c81labfstdf_patch_{test,val}.json)."""
import sys, json, re, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))  # repo root (QP_ROOT overrides)
BALLS = {'basketball': 0.24, 'tennis_ball': 0.067}
RATIO = 1.4
ALT = 'v3nt'
DIAG = []

def norm(s): return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')

def find_track(tj, phrase):
    ph = str(phrase).lower().strip()
    for src in ('queries_d', 'queries'):
        for q in tj.get(src) or []:
            if q.get('obj') is None: continue
            if q.get('phrase', '').lower().strip() == ph and not q.get('where'):
                return tj['objs'].get(str(q['obj']))
    return None

def find_q(tj, phrase):
    ph = str(phrase).lower().strip()
    for src in ('queries_d', 'queries'):
        L = tj.get(src) or []
        for q in L:
            if q.get('obj') is None: continue
            if q.get('phrase', '').lower().strip() == ph and not q.get('where'):
                shared = [o.get('phrase') for o in L if o is not q and o.get('obj') == q['obj'] and o.get('phrase', '').lower().strip() != ph]
                return q['obj'], src, shared
    return None, None, []

CSTD = 10.0
def cstd(tr):
    B = boxes(tr); c = (B[:, :2] + B[:, 2:]) / 2
    return float(np.sqrt((c.std(axis=0) ** 2).sum()))

def boxes(tr):
    return np.array([x['st']['box'] for x in tr if x and x.get('st') and x['st'].get('box')], float)

def bp(c, rng, f, cx, cy):
    r = np.array([(c[0] - cx) / f, (c[1] - cy) / f, 1.0]); return r / np.linalg.norm(r) * rng


def measure(split, i, vid, p, q, dep, st, k1, k2, balls, meas_f, mt):
    """production per-row computation on track set mt; returns ('ok', value, why) or (status, None, None)"""
    tp = f'{R}/geo/out/tracks_{split}_{mt}/{vid}.json'
    if not os.path.exists(tp):
        DIAG.append((split, i, vid, 'notracks ' + tp)); return 'notracks', None, None
    tj = json.load(open(tp)); W, H = tj['W'], tj['H']; cx, cy = W / 2, H / 2
    f_cal = float(meas_f) * W
    fstds = []
    for b in balls:
        tr = find_track(tj, b.replace('_', ' '))
        if tr is None: continue
        B = boxes(tr)
        if len(B) < 5: continue
        uncut = (B[:, 0] > 1) & (B[:, 1] > 1) & (B[:, 2] < W - 2) & (B[:, 3] < H - 2)
        if uncut.mean() < 0.8: continue
        B = B[uncut]
        s = np.median(np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]))
        c = np.median((B[:, :2] + B[:, 2:]) / 2, axis=0)
        rng = st[b][0]
        f = s * rng / BALLS[b]
        for _ in range(5):
            nr = np.sqrt(1 + ((c[0] - cx) ** 2 + (c[1] - cy) ** 2) / f ** 2)
            f = s * (rng / nr) / BALLS[b]
        fstds.append(f)
    if not fstds:
        DIAG.append((split, i, vid, 'noballtrack ' + mt)); return 'noballtrack', None, None
    f_std = float(np.median(fstds))
    ratio = f_cal / f_std
    ph1 = p['target']['phrase'] if isinstance(p.get('target'), dict) else q['target']
    ph2 = p['target2']['phrase'] if isinstance(p.get('target2'), dict) else q['target2']
    tr1 = find_track(tj, ph1); tr2 = find_track(tj, ph2)
    DIAG.append((split, i, vid, 'cand %s f_cal/W=%.3f f_std/W=%.3f ratio=%.3f tr1=%s tr2=%s' % (mt, f_cal / W, f_std / W, ratio, tr1 is not None, tr2 is not None)))
    if ratio <= RATIO or tr1 is None or tr2 is None: return 'ratio', None, None
    # endpoint filter, per endpoint: the production filter on the meas_tracks track; if that track fails, the same filter on
    # the ALT (v3nt) track of the same video, whose box replaces it.  f_std and the ratio gate stay on meas_tracks.
    pobj = norm(p['prior_obj']['phrase'] if isinstance(p.get('prior_obj'), dict) else p.get('prior_obj') or '')
    timed = {norm(d['key']) for d in dep if d['t'] is not None}
    def endpoint_bad(tj_, ph_, tag_):
        ob, src, sh = find_q(tj_, ph_); tr_ = find_track(tj_, ph_)
        if tr_ is None or len(boxes(tr_)) == 0: return ['notrack:%s' % ph_], None
        sd = cstd(tr_); bad = []
        DIAG.append((split, i, vid, 'filt %s %s obj=%s src=%s cstd=%.1f shared=%s' % (tag_, ph_, ob, src, sd, sh)))
        if sd > CSTD: bad.append('moving:%s' % ph_)
        mov = [x for x in sh if norm(x) == pobj or norm(x).replace('_', '') in {t.replace('_', '') for t in timed} or norm(x).replace('_', '') in {k.replace('_', '') for k in (p.get('prior_dkeys') or [])}]
        if mov: bad.append('sharedobj:%s~%s' % (ph_, mov))
        return bad, tr_
    used = []; trs = []; bad_all = []
    for ph_ in (ph1, ph2):
        bad, tr_ = endpoint_bad(tj, ph_, mt)
        if bad and mt != ALT:
            tpa = f'{R}/geo/out/tracks_{split}_{ALT}/{vid}.json'
            if os.path.exists(tpa):
                tja = json.load(open(tpa))
                if abs(tja['W'] - W) <= 2 and abs(tja['H'] - H) <= 2:   # v2 853x480 vs v3nt 854x480 on 0053x
                    bad2, tra = endpoint_bad(tja, ph_, ALT)
                    DIAG.append((split, i, vid, 'fallback %s %s: %s -> %s' % (ALT, ph_, ';'.join(bad), ';'.join(bad2) or 'pass')))
                    if not bad2: bad, tr_ = [], tra; used.append('%s=%s' % (ph_, ALT))
        bad_all += bad; trs.append(tr_)
    if bad_all:
        DIAG.append((split, i, vid, 'dropped %s ' % mt + ';'.join(bad_all))); return 'dropped', None, None
    tr1, tr2 = trs
    c1 = np.median([(b[:2] + b[2:]) / 2 for b in boxes(tr1)], axis=0)
    c2 = np.median([(b[:2] + b[2:]) / 2 for b in boxes(tr2)], axis=0)
    d1, d2 = st[k1[0]][0], st[k2[0]][0]
    v = float(np.linalg.norm(bp(c1, d1, f_std, cx, cy) - bp(c2, d2, f_std, cx, cy)))
    vcal = float(np.linalg.norm(bp(c1, d1, f_cal, cx, cy) - bp(c2, d2, f_cal, cx, cy)))
    v = max(v, abs(d1 - d2))
    why = 'rul156:f_std/W=%.3f(f_cal/W=%.3f,geo_fcal=%.3f)' % (f_std / W, f_cal / W, vcal)
    if used:
        why += ' endpoint track fallback %s (meas_tracks %s track failed the static/unshared filter)' % (','.join(used), mt)
    return 'ok', v, why

def run(split):
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[int(j['id'])] = j
    out = {}
    for i, rr in raw.iterrows():
        p = P.get(i)
        if p is None: continue
        vid = str(rr.video_id)
        q = p['q']
        if not vid.startswith('captured_') or q.get('kind') != 'dist2' or q.get('t') is not None: continue
        if q.get('t1') is not None or q.get('t2') is not None or q.get('when') is not None: continue
        if rr.meas_mode != '3d_calib' or pd.isna(rr.meas_f): continue
        dep = p.get('depth') or []
        st = {}
        for d in dep:
            if d['t'] is None: st.setdefault(d['key'], []).append(d['d'])
        k1 = [k for k in (p.get('target_dkeys') or []) if k in st]
        k2 = [k for k in (p.get('target2_dkeys') or []) if k in st]
        if len(k1) != 1 or len(k2) != 1 or k1[0] == k2[0]:
            DIAG.append((split, i, vid, 'nokeys')); continue
        balls = [b for b in BALLS if b in st and len(st[b]) == 1]
        if not balls:
            DIAG.append((split, i, vid, 'noball')); continue
        mt = str(rr.meas_tracks)
        status, v, why = measure(split, i, vid, p, q, dep, st, k1, k2, balls, rr.meas_f, mt)
        if status == 'ok':
            out[str(i)] = {'value': v, 'why': why}
    return out

if __name__ == '__main__':
    for split in ('test', 'val'):
        o = run(split)
        json.dump(o, open(f'{R}/cache/tmp/ag_c81labfstdf_patch_{split}.json', 'w'), indent=1)
        print(split, len(o), o)
    for d in DIAG:
        if d[3].split()[0] in ('cand', 'notracks', 'filt', 'dropped', 'fallback'): print(d)
    print('n diag', len(DIAG), pd.Series([d[3].split()[0] for d in DIAG]).value_counts().to_dict())
