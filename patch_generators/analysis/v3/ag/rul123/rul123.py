"""rul123: coverage gap of rul093 (vehicle acceleration on its own wheel/tire part track).

rul093 measured the part centre's second derivative with a quadratic over half-windows 0.20/0.25/0.33 s and skipped
rows whose windows disagreed (> 1.25x) or whose value was within [0.5, 2] of the current answer. When the vehicle's
acceleration is not constant over the window (it tapers off: raw second differences of the tire centre fall from ~8.6
to ~1.3 m/s^2 over t 0.54-1.04 on the brief's example), a quadratic averages the jerk away differently per window and
the windows disagree. A local CUBIC models a linearly changing acceleration, so its second derivative at t is the local
value and is much less window-dependent.

Code conditions only (no ids). Every rul080/rul093 gate is kept (part prior bound to the target, part track inside
the target box, uncut size-stable frames: size CV < 5% and drift < 10%, mechanism gate, static camera). Changes:
 (1) part frames whose width < 0.93 x median part width, or whose box touches the frame border (x1 <= 3 or
     x2 >= W - 3), are dropped (partly hidden/clipped tire -> its centre shifts);
 (2) acceleration = second derivative at t of a cubic fit of centre x, y vs (time - t), half-windows 0.17/0.21/0.25/0.33 s,
     each needing >= 7 frames and frames on both sides of t; a = hypot(ax, ay) x diameter / median part height;
 (3) >= 2 finite windows with max/min <= 1.25; value = their median.
Applies only to accel rows that rul093's own logic logs as 'window-unstable' or 'within [0.5,2] of current' (and rows
not already in the rul093 patch), and only where answer/new is outside [0.8, 1.25].
Fusion as c53: within 2x of CaW -> geometric mean; else the part value. var_ans_plain rows copy their plain row (their own
segmented track is never used). Place before c63plainkey.
Writes cache/tmp/ag_c68partacc3_patch_{test,val}.json"""
import sys, json, os, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul123'
SUB = os.environ.get('C68_SUB', 'sub_xt_Nzz17')
sys.path.insert(0, R + '/analysis/v3/ag/rul080')
import rul080 as M

# ---------- c53's original quadratic (for reproducing c53's classification) ----------
HW = [0.25]


def qfit_quad(t, y, tq, half=None):
    half = HW[0]
    m = (t >= tq - half - 1e-9) & (t <= tq + half + 1e-9)
    if m.sum() < 7 or np.min(np.abs(t[m] - tq)) > 0.1:
        return np.nan, int(m.sum())
    p = np.polyfit(t[m] - tq, y[m], 2)
    return 2 * p[0], int(m.sum())


def cfit_acc(t, y, tq, half):
    m = (t >= tq - half - 1e-9) & (t <= tq + half + 1e-9)
    if m.sum() < 7 or not ((t[m] < tq - 1e-9).any() and (t[m] > tq + 1e-9).any()):
        return np.nan, int(m.sum())
    p = np.polyfit(t[m] - tq, y[m], 3)
    return 2 * p[1], int(m.sum())


def measure3(sp, vid, tph, pkey, psi, t, nparts, csv_fps):
    """rul080.measure route P with the c68 frame filter and the cubic multi-window acceleration"""
    notes = []
    for d in M.DIRS:
        p = f'{R}/geo/out/tracks_{sp}{d}/{vid}.json'
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        fps = float(csv_fps); W = float(j['W']); H = float(j['H'])
        dn = d or 'v1'
        tks = M.tracks_for(j, tph + '|')
        if not tks:
            notes.append(f'{dn}: no target track'); continue
        tk = tks[0]
        ft = {int(r[0]): r[1:] for r in tk}
        for pk in M.tracks_for(j, pkey + '|'):
            ins, keep = [], []
            for r in pk:
                b = ft.get(int(r[0]))
                if b is None:
                    keep.append(False); continue
                cx = (r[1] + r[3]) / 2; cy = (r[2] + r[4]) / 2
                inn = bool(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2)
                ins.append(inn); keep.append(inn)
            if sum(ins) < 8 or np.mean(ins) < 0.6:
                notes.append(f'{dn}: part track not inside target'); continue
            pk = pk[np.array(keep)]
            pt = pk[:, 0] / fps; ph = pk[:, 4] - pk[:, 2]; pw = pk[:, 3] - pk[:, 1]
            free = (pk[:, 1] > 2) & (pk[:, 3] < W - 2) & (pk[:, 2] > 2) & (pk[:, 4] < H - 2)
            if free.sum() < 5:
                continue
            medh = np.median(ph[free]); medw = np.median(pw[free])
            good = free & (np.abs(ph / medh - 1) < 0.15) & (np.abs(pw / medw - 1) < 0.25) & ~M.clipped(pk)
            # (1) c68: drop narrow (partly hidden) and border-touching part frames
            good &= (pw >= 0.93 * medw) & (pk[:, 1] > 3) & (pk[:, 3] < W - 3)
            inwin = good & (np.abs(pt - t) <= 0.3)
            cp, why_cp = M.cut_or_pin(tk, t, fps, W)
            if not (inwin.sum() >= 6 and np.ptp(pt[inwin]) >= 0.2):
                notes.append(f'{dn}: too few good part frames near t ({int(inwin.sum())})'); continue
            hw = ph[inwin]
            cv = hw.std() / hw.mean(); drift = hw.max() / hw.min() - 1
            if cv >= 0.05 or drift >= 0.10:
                notes.append(f'{dn}: part size unstable cv {cv:.3f} drift {drift:.3f}'); continue
            if not (cp or nparts >= 2):
                notes.append(f'{dn}: {why_cp}, single prior'); continue
            st_, why_st = M.cam_static(sp, vid, tk, t, fps, W, H)
            if not st_:
                notes.append(f'{dn}: camera not static ({why_st})'); continue
            cx = (pk[:, 1] + pk[:, 3]) / 2; cy = (pk[:, 2] + pk[:, 4]) / 2
            mpp = psi / np.median(hw)
            vals = {}
            for h in (0.17, 0.21, 0.25, 0.33):
                ax, n = cfit_acc(pt[good], cx[good], t, h); ay, _ = cfit_acc(pt[good], cy[good], t, h)
                if np.isfinite(ax) and np.isfinite(ay):
                    vals[h] = float(np.hypot(ax, ay) * mpp)
            # diagnostic: the quadratic on the same frames
            qv = {}
            for h in (0.17, 0.25, 0.33):
                HW[0] = h
                ax, _ = qfit_quad(pt[good], cx[good], t); ay, _ = qfit_quad(pt[good], cy[good], t)
                if np.isfinite(ax):
                    qv[h] = round(float(np.hypot(ax, ay) * mpp), 2)
            tend = float(pt[good].max())
            return dict(vals=vals, quad=qv, dir=dn, dpx=float(np.median(hw)), cv=float(cv), drift=float(drift), tend=tend,
                        ngood=int(good.sum()), mech=(why_cp if cp else f'{nparts} part priors') + '; ' + why_st), notes
    return None, notes


def c53_class(sp, vid, tph, keys, pr, t, npri, fps, old):
    """reproduce rul093's classification for a row"""
    M.qfit = qfit_quad
    vals = {}
    for hw in (0.2, 0.25, 0.33):
        HW[0] = hw
        res = None
        for k in keys:
            res, _ = M.measure(sp, vid, tph, k, float(pr['value_si']), t, npri, fps)
            if res:
                break
        if res and np.isfinite(res['v']) and res['v'] > 0:
            vals[hw] = res['v']
    fin = list(vals.values())
    if len(fin) < 2:
        return 'fewer than 2 windows', vals
    if max(fin) / min(fin) > 1.25:
        return 'window-unstable', vals
    a = vals.get(0.25, float(np.median(fin)))
    if 0.5 <= a / old <= 2:
        return 'within [0.5,2] of current', vals
    return 'c53 would change', vals


def run(sp):
    suf = 'detail' if sp == 'test' else 'valdetail'
    D = pd.read_csv(f'{R}/submissions/{SUB}_{suf}.csv', index_col=0)
    D7 = pd.read_csv(f'{R}/submissions/sub_xt_Nzz7_{suf}.csv', index_col=0)
    RAW = pd.read_csv(f'{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', usecols=['video_id', 'fps', 'question'])
    c53p = set(json.load(open(f'{R}/cache/tmp/ag_c53partacc_patch_{sp}.json')))
    P = {}
    for l in open(f'{R}/geo/out/parse_{sp}_v2.jsonl'):
        j = json.loads(l)
        P[j['id']] = j
    meas, log = {}, []
    for i, j in P.items():
        q = j.get('q') or {}
        if q.get('kind') != 'accel' or q.get('t') is None or i not in D.index:
            continue
        if str(D.loc[i, 'answer_source']) == 'var_ans_plain' or str(i) in c53p:
            continue
        pri = [p for p in (j.get('priors') or []) if p.get('kind') == 'size' and p.get('obj') and M.PART.search(p['obj'].lower())]
        if not pri:
            continue
        tph = ((j.get('target') or {}).get('phrase') or '').lower().strip()
        if not tph or M.PART.search(tph):
            continue
        sel = [p for p in pri if re.search(r'\bof (?:the |a )?' + re.escape(tph) + r'\b', p['obj'].lower())]
        if not sel and len(pri) == 1 and not re.search(r'\bof\b', pri[0]['obj'].lower()):
            sel = pri
        if len(sel) != 1:
            continue
        pr = sel[0]
        pkey = M.PART.search(pr['obj'].lower()).group(0)
        pph = ((j.get('prior_obj') or {}).get('phrase') or '').lower().strip()
        keys = [pph] if pph and M.PART.search(pph) else []
        if pkey not in keys:
            keys.append(pkey)
        vid = D.loc[i, 'video_id']
        if i >= len(RAW) or str(RAW.video_id.iloc[i]) != str(vid):
            continue
        t = float(q['t']); fps = RAW.fps.iloc[i]
        old = float(D.loc[i, 'answer'])
        old7 = float(D7.loc[i, 'answer']) if i in D7.index else old
        cls, v53 = c53_class(sp, vid, tph, keys, pr, t, len(pri), fps, old7)
        if cls not in ('window-unstable', 'within [0.5,2] of current'):
            log.append((i, vid, 'c53 class: ' + cls, {k: round(v, 2) for k, v in v53.items()}))
            continue
        res = None; notes = []
        for k in keys:
            res, nt = measure3(sp, vid, tph, k, float(pr['value_si']), t, len(pri), fps)
            notes += nt
            if res:
                break
        if not res:
            log.append((i, vid, cls, 'no c68 measurement', notes[-3:])); continue
        fin = list(res['vals'].values())
        info = (i, vid, cls, 'c53 quad', {k: round(v, 2) for k, v in v53.items()}, 'cubic', {k: round(v, 2) for k, v in res['vals'].items()},
                'quad same frames', res['quad'], 'part %.0fpx cv %.3f drift %.3f n %d tend %.2f t %.2f' % (res['dpx'], res['cv'], res['drift'], res['ngood'], res['tend'], t), 'old', round(old, 3), 'cs', D.loc[i, 'cs'])
        if len(fin) < 2:
            log.append(info + ('fewer than 2 cubic windows',)); continue
        if max(fin) / min(fin) > 1.25:
            log.append(info + ('cubic window-unstable',)); continue
        a = float(np.median(fin))
        if 0.8 <= old / a <= 1.25:
            log.append(info + ('within [0.8,1.25] of answer',)); continue
        log.append(info + ('CHANGE',))
        meas[i] = (a, res, pr, old)
    out = {}
    for i, (a, res, pr, old) in meas.items():
        cs = D.loc[i, 'cs']
        if np.isfinite(cs) and cs > 0 and 0.5 <= a / cs <= 2:
            val, how = float(np.sqrt(a * cs)), 'agree_gm with CaW %.3g' % cs
        else:
            val, how = a, 'disagrees with CaW %.3g -> part value' % cs
        hws = '/'.join('%.2f' % res['vals'][h] if h in res['vals'] else '-' for h in (0.17, 0.21, 0.25, 0.33))
        out[i] = (val, "%s: cubic accel on the bound part track ('%s') %.3g m/s^2 (hw .17/.21/.25/.33: %s; %s, part %.0fpx; %s); %s; was %.4g (%s)"
                  % (TAG, pr['raw'], a, hws, res['dir'], res['dpx'], res['mech'], how, old, D.loc[i, 'answer_source']))
    QQ = RAW.question.astype(str)
    key = {(str(RAW.video_id.iloc[k]), QQ.iloc[k]): k for k in range(len(RAW))}
    for i in D.index[D.answer_source.astype(str) == 'var_ans_plain']:
        if i >= len(RAW):
            continue
        i0 = key.get((str(D.loc[i, 'video_id']).replace('_segmented', ''), QQ.iloc[i]))
        if i0 in out:
            out[i] = (out[i0][0], '%s: copy of plain row %d (var_ans_plain); was %.4g' % (TAG, i0, float(D.loc[i, 'answer'])))
    js = {str(int(i)): dict(value=float(v), why=w) for i, (v, w) in out.items()}
    json.dump(js, open(f'{R}/cache/tmp/ag_{TAG}_patch_{sp}.json', 'w'), indent=1)
    return js, log, D


if __name__ == '__main__':
    sys.path.insert(0, R + '/cache/tmp')
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    for sp in ('test', 'val'):
        js, log, D = run(sp)
        print('==', sp, len(js), 'changed; by cat', D['cat'].reindex([int(k) for k in js]).value_counts().to_dict())
        Ls = L[L.split == sp]
        for k, e in js.items():
            print('  CHANGE', k, D.loc[int(k), 'video_id'], D.loc[int(k), 'cat'], round(float(D.loc[int(k), 'answer']), 3), '->', round(e['value'], 3), '|', e['why'])
            if int(k) in Ls.index:
                y = Ls.y[int(k)]
                print('   LABELLED', y, A.row_mra(np.array([D.loc[int(k), 'answer']]), np.array([y])), '->', A.row_mra(np.array([e['value']]), np.array([y])))
        for x in log:
            print('  LOG', x)
