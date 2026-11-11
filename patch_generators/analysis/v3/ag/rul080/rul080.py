"""rul080: speed of a vehicle measured on its own wheel/tire track, scaled by the stated tire diameter.

Mechanism (audit 47, garage template): the stated prior is a PART of the target ('diameter of the tire [of the X car]').
Geo measures the target's box centroid. When the box is clamped by the image border or by a dark near-field occluder
(the box edge stays pinned while the other edge moves), the centroid moves at about half the true speed; when the
prior text names two parts ('tire of the yellow car', 'tire of the blue car'), geo can bind the wrong ruler.
The tire is rigidly attached: its box centre translates with the vehicle, and its VERTICAL box extent is the stated
diameter regardless of yaw. If its size is constant the vehicle does not move in depth, so lateral image speed x
(diameter_m / diameter_px) is the full speed.

Code conditions only (no ids):
  - speed rows (kind == 'speed', time t given); accel rows are NOT touched (window-sensitive);
  - prior: size prior whose object head noun is tire/tyre/wheel; bound to the target by 'of the <target phrase>' when
    several part priors exist, else the single bare part prior; the target phrase itself is not a part;
  - part track: a query track ('queries' or 'queries_d' mapping) in track dirs v2sc, v2, v1, v2full, v3nt whose box
    centre lies inside the SAME file's target box in >= 60% of co-visible frames and >= 8 frames; only those contained
    frames are used (an identity switch to another vehicle's part drops out);
  - route P (part centre track): >= 6 good part frames in [t-0.3, t+0.3] spanning >= 0.2 s (good = not border-cut,
    height within 15% of the part median), size CV < 5% and max/min - 1 < 10% in that window; speed = |quadratic fit
    slope| of the part centre (x, y) over [t-0.25, t+0.25] (>= 7 points, one within 0.1 s of t) x (diameter / median part height in the window);
  - route B (target centroid, only when route P has no window coverage and >= 2 part priors are stated): target box
    uncut and unpinned in [t-0.3, t+0.3], target box height CV < 5% there, part height from all good contained frames
    (CV < 5%), and the target box height at those frames within 10% of the window's (no depth change);
  - mechanism gate: route P needs the target box border-cut or pinned (one edge within 2 px over 4 frames while the
    other moves >= 20 px) somewhere in [t-0.3, t+0.3], OR >= 2 part priors stated; route B needs >= 2 part priors;
  - static camera (both routes): background corners outside the target boxes, tracked t-0.25 s -> t+0.25 s, move < 2 px
    (median; the track json's 'cam' estimate is contaminated by a frame-filling car, so it is not used);
  - change only if new/old outside [0.8, 1.25].
Writes cache/tmp/ag_c47partkin_patch_{test,val}.json"""
import sys, json, os, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul080'
PART = re.compile(r'\b(tire|tyre|wheel)s?\b')
DIRS = ('_v2sc', '_v2', '', '_v2full', '_v3nt')


def tracks_for(j, key):
    """all candidate tracks (n x 5: fidx, x1, y1, x2, y2) for a query key, from 'queries' and 'queries_d'"""
    out = []
    if 'objs' in j:
        seen = set()
        for ql in (j.get('queries') or [], j.get('queries_d') or []):
            for q in ql:
                if isinstance(q, dict) and q.get('key') == key and q.get('obj') is not None and q['obj'] not in seen:
                    seen.add(q['obj'])
                    lst = j['objs'].get(str(q['obj']))
                    if lst:
                        out.append([(f,) + tuple(e['st']['box']) for f, e in zip(j['fidx'], lst) if e and e.get('st') and e['st'].get('box')])
    else:
        for q in j.get('queries') or []:
            if q.get('key') == key and q.get('frames') is not None:
                out.append([(f,) + tuple(e['st']['box']) for f, e in zip(j['fidx'], q['frames']) if e and e.get('st') and e['st'].get('box')])
    return [np.array(a, float) for a in out if len(a) >= 3]


def clipped(pk):
    """part frames clipped by an occluder or the frame: one box edge stays put (<= 2 px) versus an adjacent tracked frame
    while the opposite edge on the same axis moves >= 6 px (the visible extent changes with one side pinned)"""
    n = len(pk)
    out = np.zeros(n, bool)
    for k in range(n):
        for k2 in (k - 1, k + 1):
            if k2 < 0 or k2 >= n or abs(pk[k2, 0] - pk[k, 0]) > 1.5:
                continue
            for e, o in ((1, 3), (3, 1), (2, 4), (4, 2)):
                if abs(pk[k, e] - pk[k2, e]) <= 2 and abs(pk[k, o] - pk[k2, o]) >= 6:
                    out[k] = True
    return out


def qfit(t, y, tq, half=0.25):
    m = (t >= tq - half - 1e-9) & (t <= tq + half + 1e-9)
    if m.sum() < 7 or np.min(np.abs(t[m] - tq)) > 0.1:   # >= 7 points and a frame within 0.1 s of t (no long extrapolation)
        return np.nan, int(m.sum())
    p = np.polyfit(t[m] - tq, y[m], 2)
    return p[-2], int(m.sum())


def cut_or_pin(tk, t, fps, W):
    tt = tk[:, 0] / fps
    m = np.abs(tt - t) <= 0.3
    if not m.any():
        return False, 'no target frames'
    if ((tk[m, 1] <= 2) | (tk[m, 3] >= W - 2)).any():
        return True, 'target box border-cut'
    for e, o in ((1, 3), (3, 1)):
        x = tk[m, e]
        y = tk[m, o]
        for k in range(len(x) - 3):
            if np.ptp(x[k:k + 4]) <= 2 and np.ptp(y[k:k + 4]) >= 20:
                return True, 'target box edge pinned at x=%.0f' % x[k]
    return False, 'target box free'


def cam_static(sp, vid, tk, t, fps, W, H):
    """image-space kinematics need a static camera: track background corners (outside every target box seen in the window,
    10 px margin) from the frame at t-0.25 s to the frame at t+0.25 s with pyramidal LK; static = >= 20 corners tracked and
    their median displacement < 2 px over the 0.5 s span (the vehicle itself moves hundreds of px). A panning camera fails
    (large median shift, or LK loses the corners)."""
    import cv2
    fa, fb = max(0, int(round((t - 0.25) * fps))), int(round((t + 0.25) * fps))
    cap = cv2.VideoCapture(f'{R}/data/{sp}_480p/{vid}.mp4')
    fr = {}
    k = 0
    while k <= fb:
        ok, f = cap.read()
        if not ok:
            break
        if k in (fa, fb):
            fr[k] = cv2.cvtColor(cv2.resize(f, (int(W), int(H))), cv2.COLOR_BGR2GRAY)
        k += 1
    if fa not in fr or fb not in fr:
        return False, 'frames unavailable'
    m = np.full((int(H), int(W)), 255, np.uint8)
    for r in tk[(tk[:, 0] >= fa - 1) & (tk[:, 0] <= fb + 1)]:
        m[max(0, int(r[2]) - 10):int(r[4]) + 10, max(0, int(r[1]) - 10):int(r[3]) + 10] = 0
    p0 = cv2.goodFeaturesToTrack(fr[fa], 200, 0.01, 8, mask=m)
    if p0 is None or len(p0) < 20:
        return False, 'too few background corners'
    p1, st, _ = cv2.calcOpticalFlowPyrLK(fr[fa], fr[fb], p0, None, winSize=(21, 21), maxLevel=3)
    ok_ = st.ravel() == 1
    if ok_.sum() < 20:
        return False, 'background corners lost'
    dmed = float(np.median(np.linalg.norm((p1 - p0).reshape(-1, 2)[ok_], axis=1)))
    return dmed < 2.0, f'background shift {dmed:.2f}px/{fb - fa}fr ({int(ok_.sum())} corners)'


def measure(sp, vid, tph, pkey, psi, t, nparts, csv_fps=None):
    notes = []
    for d in DIRS:
        p = f'{R}/geo/out/tracks_{sp}{d}/{vid}.json'
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        fps = float(csv_fps) if csv_fps else float(j['fps']); W = float(j['W']); H = float(j['H'])   # CSV fps (never the container fps)
        dn = d or 'v1'
        tks = tracks_for(j, tph + '|')
        if not tks:
            notes.append(f'{dn}: no target track')
            continue
        tk = tks[0]
        ft = {int(r[0]): r[1:] for r in tk}
        for pk in tracks_for(j, pkey + '|'):
            ins, keep = [], []
            for r in pk:
                b = ft.get(int(r[0]))
                if b is None:
                    keep.append(False)
                    continue
                cx = (r[1] + r[3]) / 2; cy = (r[2] + r[4]) / 2
                inn = bool(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2)
                ins.append(inn); keep.append(inn)
            if sum(ins) < 8 or np.mean(ins) < 0.6:
                notes.append(f'{dn}: part track not inside target ({np.mean(ins) if ins else 0:.2f}, n={len(ins)})')
                continue
            pk = pk[np.array(keep)]   # only frames where the part sits inside the target box (identity switches drop out)
            pt = pk[:, 0] / fps; ph = pk[:, 4] - pk[:, 2]; pw = pk[:, 3] - pk[:, 1]
            free = (pk[:, 1] > 2) & (pk[:, 3] < W - 2) & (pk[:, 2] > 2) & (pk[:, 4] < H - 2)
            if free.sum() < 5:
                continue
            medh = np.median(ph[free]); medw = np.median(pw[free])
            good = free & (np.abs(ph / medh - 1) < 0.15) & (np.abs(pw / medw - 1) < 0.25) & ~clipped(pk)
            inwin = good & (np.abs(pt - t) <= 0.3)
            cp, why_cp = cut_or_pin(tk, t, fps, W)
            if inwin.sum() >= 6 and np.ptp(pt[inwin]) >= 0.2:
                hw = ph[inwin]
                cv = hw.std() / hw.mean(); drift = hw.max() / hw.min() - 1
                if cv >= 0.05 or drift >= 0.10:
                    notes.append(f'{dn}: part size unstable cv {cv:.3f} drift {drift:.3f}')
                    continue
                if not (cp or nparts >= 2):
                    notes.append(f'{dn}: {why_cp}, single prior')
                    continue
                st_, why_st = cam_static(sp, vid, tk, t, fps, W, H)
                if not st_:
                    notes.append(f'{dn}: camera not static ({why_st})')
                    continue
                cx = (pk[:, 1] + pk[:, 3]) / 2; cy = (pk[:, 2] + pk[:, 4]) / 2
                vx, n = qfit(pt[good], cx[good], t); vy, _ = qfit(pt[good], cy[good], t)
                if not np.isfinite(vx):
                    notes.append(f'{dn}: fit failed')
                    continue
                mpp = psi / np.median(hw)
                return dict(v=float(np.hypot(vx, vy) * mpp), route='P', dir=dn, n=n, dpx=float(np.median(hw)), cv=float(cv),
                            mech=(why_cp if cp else f'{nparts} part priors') + '; ' + why_st), notes
            # route B: target centroid with the bound part's ruler
            if nparts < 2:
                notes.append(f'{dn}: part not visible near t, single prior')
                continue
            if cp:
                notes.append(f'{dn}: route B blocked ({why_cp})')
                continue
            st_, why_st = cam_static(sp, vid, tk, t, fps, W, H)
            if not st_:
                notes.append(f'{dn}: camera not static ({why_st})')
                continue
            tt = tk[:, 0] / fps; th = tk[:, 4] - tk[:, 2]
            m = np.abs(tt - t) <= 0.3
            if m.sum() < 7 or th[m].std() / th[m].mean() >= 0.05:
                notes.append(f'{dn}: target box height unstable near t')
                continue
            hw = ph[good]
            if len(hw) < 5 or hw.std() / hw.mean() >= 0.05:
                notes.append(f'{dn}: part size unstable')
                continue
            thp = np.interp(pt[good], tt, th)
            if abs(np.median(thp) / np.median(th[m]) - 1) >= 0.10:
                notes.append(f'{dn}: target depth differs between part frames and t')
                continue
            cxt = (tk[:, 1] + tk[:, 3]) / 2; cyt = (tk[:, 2] + tk[:, 4]) / 2
            vx, n = qfit(tt, cxt, t); vy, _ = qfit(tt, cyt, t)
            if not np.isfinite(vx):
                notes.append(f'{dn}: fit failed')
                continue
            mpp = psi / np.median(hw)
            return dict(v=float(np.hypot(vx, vy) * mpp), route='B', dir=dn, n=n, dpx=float(np.median(hw)), cv=float(hw.std() / hw.mean()),
                        mech=f'{nparts} part priors, target box free; ' + why_st), notes
    return None, notes


def run(sp):
    D = pd.read_csv(f'{R}/submissions/sub_xt_Nzz3_{"detail" if sp == "test" else "valdetail"}.csv', index_col=0)
    RAW = pd.read_csv(f'{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', usecols=['video_id', 'fps', 'question'])
    FPS = RAW.fps
    P = {}
    for l in open(f'{R}/geo/out/parse_{sp}_v2.jsonl'):
        j = json.loads(l)
        P[j['id']] = j
    out, log, meas = {}, [], {}
    for i, j in P.items():
        q = j.get('q') or {}
        if q.get('kind') != 'speed' or q.get('t') is None or i not in D.index:
            continue
        pri = [p for p in (j.get('priors') or []) if p.get('kind') == 'size' and p.get('obj') and PART.search(p['obj'].lower())]
        if not pri:
            continue
        tph = ((j.get('target') or {}).get('phrase') or '').lower().strip()
        if not tph or PART.search(tph):
            continue
        sel = [p for p in pri if re.search(r'\bof (?:the |a )?' + re.escape(tph) + r'\b', p['obj'].lower())]
        if not sel and len(pri) == 1 and not re.search(r'\bof\b', pri[0]['obj'].lower()):
            sel = pri
        if len(sel) != 1:
            log.append((i, 'no unique bound part prior'))
            continue
        pr = sel[0]
        pkey = PART.search(pr['obj'].lower()).group(0)
        pph = ((j.get('prior_obj') or {}).get('phrase') or '').lower().strip()
        keys = [pph] if pph and PART.search(pph) else []
        if pkey not in keys:
            keys.append(pkey)
        vid = D.loc[i, 'video_id']
        if i >= len(RAW) or str(RAW.video_id.iloc[i]) != str(vid):      # CSV fps must come from the same row
            log.append((i, vid, 'raw row mismatch'))
            continue
        res, notes = None, []
        for k in keys:
            res, nt = measure(sp, vid, tph, k, float(pr['value_si']), float(q['t']), len(pri), FPS.iloc[i])
            notes += nt
            if res:
                break
        old = float(D.loc[i, 'answer'])
        if not res:
            log.append((i, vid, 'no measurement', notes[-4:]))
            continue
        r_ = res['v'] / old
        if 0.8 <= r_ <= 1.25:
            log.append((i, vid, 'within 1.25x of current', round(res['v'], 3), round(old, 3), res['route'], res['dir']))
            continue
        meas[i] = (res, pr, old)
    # pipeline fusion: the part-track value replaces geo; agreement with CaW (within 2x) -> geometric mean; disagreement ->
    # the geo side only where the pipeline already answered from geo (dis_*geo) or the mode is strong, else CaW stays
    fin = {}
    for i, (res, pr, old) in meas.items():
        src = str(D.loc[i, 'answer_source']); cs = D.loc[i, 'cs']; mode = str(D.loc[i, 'mode'])
        v = res['v']
        if np.isfinite(cs) and cs > 0 and 0.5 <= v / cs <= 2:
            val, how = float(np.sqrt(v * cs)), f'agree_gm with CaW {cs:.3g}'
        elif not (np.isfinite(cs) and cs > 0) or src.startswith('dis_') and src.endswith('geo') or mode in ('2d_ratio', '3d_calib'):
            val, how = v, (f'disagrees with CaW {cs:.3g} -> geo' if np.isfinite(cs) else 'no CaW')
        else:
            log.append((i, 'disagreement, pipeline keeps CaW', round(v, 3), cs, src))
            continue
        fin[i] = (val, how)
    # segmented copies answered by copying the plain video's answer (var_ans_plain) keep doing so: they follow the plain row,
    # whether or not their own part track passed the gates; a var_ans_plain row whose plain row is not changed stays as is
    QQ = RAW.question.astype(str)                       # the val detail has no question column; the raw csv has it for both splits
    key = {(str(RAW.video_id.iloc[i]), QQ.iloc[i]): i for i in range(len(RAW))}
    for i in list(fin):
        if str(D.loc[i, 'answer_source']) == 'var_ans_plain':
            del fin[i]
            log.append((i, 'var_ans_plain row: follows its plain row only'))
    cp_ = {}
    for i in D.index[D.answer_source.astype(str) == 'var_ans_plain']:
        if i >= len(RAW):
            continue
        i0 = key.get((str(D.loc[i, 'video_id']).replace('_segmented', ''), QQ.iloc[i]))
        if i0 in fin:
            own = f", own part value {meas[i][0]['v']:.3g}" if i in meas else ''
            cp_[i] = (fin[i0][0], f'copy of plain row {i0} (var_ans_plain){own}')
            if i not in meas:
                meas[i] = (meas[i0][0], meas[i0][1], float(D.loc[i, 'answer']))
    fin.update(cp_)
    for i, (val, how) in fin.items():
        res, pr, old = meas[i]
        out[str(int(i))] = dict(value=val, why=f"{TAG}: speed on the bound part track ('{pr['raw']}') {res['v']:.3g} m/s, route {res['route']} "
                                f"({res['dir']}, n={res['n']}, part {res['dpx']:.0f}px cv {res['cv']:.3f}; {res['mech']}); {how}; was {old:.4g} ({D.loc[i, 'answer_source']})")
    json.dump(out, open(f'{R}/cache/tmp/ag_{TAG}_patch_{sp}.json', 'w'), indent=1)
    return out, log, D


if __name__ == '__main__':
    sys.path.insert(0, R + '/cache/tmp')
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled('sub_xt_Nzz3')
    for sp in ('test', 'val'):
        out, log, D = run(sp)
        print(f'== {sp}: {len(out)} changed; by cat', D['cat'].reindex([int(k) for k in out]).value_counts().to_dict())
        for k, e in out.items():
            print('  CHANGE', k, D.loc[int(k), 'video_id'], D.loc[int(k), 'cat'], '->', round(e['value'], 3), '|', e['why'])
            Ls = L[L.split == sp]
            if int(k) in Ls.index:
                y = Ls.y[int(k)]
                print('   LAB', y, A.row_mra(np.array([D.loc[int(k), 'answer']]), np.array([y])), '->', A.row_mra(np.array([e['value']]), np.array([y])))
        for x in log:
            print('  LOG', x)
