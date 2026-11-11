import os
"""rul147: an occluder passing in front of a STATIC target shrinks its box for a few frames; with one timed depth key the
size-depth branch (measure.track_depth_fn, SCALE_DEPTH) reads the dip as a radial excursion (far and back) and the speed
at the dip time explodes. Repair (code only, the row's own track): hidden frames s(f) < 0.75*min(med s[f-0.6,f-0.1],
med s[f+0.1,f+0.6]) with s = sqrt(box area), grouped into a run, expanded while s < 0.9*pre-dip median, run <= 0.5 s with
recovery (>= 0.9*pre-dip median) on both sides, box centre moving < 1 median box side over the dip +-0.6 s (and the question
time +-0.6 s). The hidden sizes are replaced by linear interpolation; the row is re-measured with measure_v3.measure_video
(read-only import, same env / focal / keys) and re-fused: 0.5 <= g'/cs <= 2 -> gm(g', cs); else g' in 3d_calib, CaW otherwise.
usage: python occdip.py  -> cache/tmp/ag_c79occdip_patch_{test,val}.json, cache/tmp/ag_c79occdip_rec.json"""
import sys, json, copy
sys.path.insert(0, os.path.join(os.environ.get('QP_ROOT', '.'), 'analysis/v3/ag/rul089'))
import vc
import numpy as np, pandas as pd
MV = vc.MV; M1 = MV.M1; R = vc.R
KINDS = {'speed', 'accel', 'disp', 'avgspeed', 'path'}
ORIG = M1.track_depth_fn
STATE = dict(mode='log', log={}, fix={})


def series(trk):
    t = np.asarray(trk.t, float)
    s = np.sqrt(np.array([trk.st[j]['area'] for j in trk.okc], float))
    c = np.array([trk.st[j]['c'] for j in trk.okc], float)
    return t, s, c


def find_dips(t, s):
    n = len(s)
    if n < 10:
        return []
    hid = np.zeros(n, bool); refs = {}
    for i in range(n):
        pre = s[(t >= t[i] - 0.6) & (t <= t[i] - 0.1)]; post = s[(t >= t[i] + 0.1) & (t <= t[i] + 0.6)]
        if len(pre) >= 3 and len(post) >= 3:
            hid[i] = s[i] < 0.75 * min(np.median(pre), np.median(post))
            refs[i] = float(np.median(pre))
    runs, i = [], 0
    while i < n:
        if hid[i]:
            j = i
            while j + 1 < n and hid[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    out = []
    for a, b in runs:
        pre_med = refs[a]
        lo, hi = a, b
        while lo - 1 >= 0 and s[lo - 1] < 0.9 * pre_med:
            lo -= 1
        while hi + 1 < n and s[hi + 1] < 0.9 * pre_med:
            hi += 1
        L, Rr = lo - 1, hi + 1
        rec = L >= 0 and Rr < n and s[L] >= 0.9 * pre_med and s[Rr] >= 0.9 * pre_med
        span = float(t[hi] - t[lo])
        out.append(dict(lo=int(lo), hi=int(hi), t0=float(t[lo]), t1=float(t[hi]), span=span, pre_med=pre_med,
                        rec=bool(rec), ok=bool(rec and span <= 0.5)))
    return out


def centre_ok(t, s, c, w0, w1, dip):
    ix = np.where((t >= w0) & (t <= w1))[0]
    if len(ix) < 3:
        return False, float('nan'), float('nan')
    nd = [k for k in ix if not (dip['lo'] <= k <= dip['hi'])]
    side = float(np.median(s[nd])) if nd else float(np.median(s[ix]))
    cc = c[ix]
    mv = float(np.linalg.norm(cc.max(0) - cc.min(0)))
    return mv < side, mv, side


def overlap_iou(vj, trk, dip):
    """max IoU of the target box with any other tracked object's box during the dip (diagnostic gate)"""
    best = 0.0
    js = [trk.okc[k] for k in range(dip['lo'], dip['hi'] + 1)]
    for oid, fr in vj['objs'].items():
        for j in js:
            if j >= len(fr) or not fr[j] or not fr[j].get('st'):
                continue
            a = np.array(trk.st[j]['box'], float); b = np.array(fr[j]['st']['box'], float)
            if np.allclose(a, b):
                continue
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
            if u > 0:
                best = max(best, ix / u)
    return best


def repaired(trk, dip):
    t, s, _ = series(trk)
    L, Rr = dip['lo'] - 1, dip['hi'] + 1
    st = list(trk.st)
    for k in range(dip['lo'], dip['hi'] + 1):
        sk = s[L] + (s[Rr] - s[L]) * (t[k] - t[L]) / (t[Rr] - t[L])
        j = trk.okc[k]
        st[j] = dict(st[j], area=float(sk ** 2))
    tr2 = copy.copy(trk)
    tr2.st = st
    return tr2


def wrap(depth, keys, trk):
    if trk is not None:
        kk = (tuple(keys or []), str(trk.q.get('key')))
        if STATE['mode'] == 'log':
            STATE['log'].setdefault(kk, trk)
        elif STATE['mode'] == 'fix' and kk in STATE['fix']:
            return ORIG(depth, keys, repaired(trk, STATE['fix'][kk]))
    return ORIG(depth, keys, trk)


M1.track_depth_fn = wrap


def fuse(g, cs, mode):
    if not np.isfinite(cs) or cs <= 0:
        return g, 'geo_only'
    if 0.5 <= g / cs <= 2:
        return float(np.sqrt(g * cs)), 'agree_gm'
    return (g, 'dis_geo') if str(mode) == '3d_calib' else (cs, 'dis_caw')


def d_str(d):
    return f"{d['t0']:.2f}-{d['t1']:.2f}s pre {d['pre_med']:.1f}"


def run(split):
    raw = pd.read_csv(f'{R}/preds/{split}_{vc.RAW}_raw.csv').set_index('Unnamed: 0')
    det = pd.read_csv(f'{R}/submissions/sub_xt_Nzz24_{"detail" if split == "test" else "valdetail"}.csv').set_index('Unnamed: 0')
    cand = raw[raw.meas_mode.astype(str).str.startswith('3d')]
    vids = sorted(set(cand.video_id))
    fmed = vc.setup(split, set(vids), tracks='v2sc')
    MV.TRACK_VER = 'v2sc'
    byv, vjs = MV.load_split_tracks(split, 'e')
    patch, rec = {}, {}
    nv = nr = 0
    for v in vids:
        if v not in byv:
            continue
        rs = byv[v]
        vj, src = vc.load_vj(split, v, 'v2sc')
        if vj is None:
            continue
        rows = []
        for r in rs:
            i = r['id']; q = r['q']
            if i not in raw.index or not str(raw.meas_mode.get(i)).startswith('3d') or q.get('kind') not in KINDS:
                continue
            tk = r.get('target_dkeys') or []
            timed = [d for d in (r.get('depth') or []) if d['key'] in tk and d['t'] is not None]
            if len(tk) != 1 or len(timed) != 1:
                continue
            rows.append(r)
        if not rows:
            continue
        nv += 1; nr += len(rows)
        STATE['mode'] = 'log'; STATE['log'] = {}
        base, _ = vc.measure(rs, vj, fmed)
        STATE['mode'] = 'off'
        fixes, rowdip = {}, {}
        for r in rows:
            i = r['id']; q = r['q']; tk = tuple(r['target_dkeys'])
            trks = [(kk, tr) for kk, tr in STATE['log'].items() if kk[0] == tk]
            if not trks:
                continue
            kk, tr = trks[0]
            t, s, c = series(tr)
            dips = [d for d in find_dips(t, s) if d['ok']]
            if q.get('t') is not None:
                h = 0.4 if q['kind'] == 'accel' else 0.25
                w0, w1 = q['t'] - h - 0.15, q['t'] + h + 0.15
            elif q.get('t1') is not None and q.get('t2') is not None:
                w0, w1 = q['t1'] - 0.15, q['t2'] + 0.15
            else:
                continue
            for d in dips:
                if d['t1'] < w0 or d['t0'] > w1:
                    continue
                tq = q.get('t')
                c0, c1 = (tq - 0.6, tq + 0.6) if tq is not None else (d['t0'] - 0.6, d['t1'] + 0.6)
                okc, mv, side = centre_ok(t, s, c, c0, c1, d)
                rec[f'{split}:{i}'] = dict(video=v, q=r['question'], kind=q['kind'], key=list(kk), dip=d, centre_move=mv, side=side,
                                           centre_ok=bool(okc), iou=overlap_iou(vj, tr, d))
                if okc:
                    fixes[kk] = d; rowdip[i] = d
                break
        if not rowdip:
            continue
        STATE['mode'] = 'fix'; STATE['fix'] = fixes
        new, _ = vc.measure(rs, vj, fmed)
        STATE['mode'] = 'off'
        for i in rowdip:
            g0 = raw.geo_value.get(i); gb = base[i]['value']; gn = new[i]['value']
            rep = bool(pd.notna(g0) and np.isfinite(gb) and abs(gb / g0 - 1) < 1e-4)
            cs = float(det.cs.get(i)) if pd.notna(det.cs.get(i)) else float('nan')
            src = str(det.answer_source.get(i)); mode = new[i]['mode']
            ans, how = fuse(gn, cs, mode) if (np.isfinite(gn) and gn > 0) else (float('nan'), 'nan')
            R_ = rec[f'{split}:{i}']
            old = float(det.answer.get(i))
            R_.update(raw_geo=g0, base=gb, new=gn, rep=rep, cs=cs, src=src, old=old, ans=ans, how=how, mode=mode,
                      merge_src=str(raw.merge_src.get(i)))
            accept = bool(rep and np.isfinite(ans) and str(raw.merge_src.get(i)) == 'd' and abs(ans / old - 1) > 1e-6
                          and (src.startswith('agree') or src.startswith('dis_') or src in ('geo', 'meas', 'mo_guard:agree_gm')))
            R_['accepted'] = accept
            print(f"[{split}] {i} {v} {R_['q'][:60]} | dip {d_str(R_['dip'])} cmove {R_['centre_move']:.1f}/{R_['side']:.1f} iou {R_['iou']:.2f} "
                  f"| raw {g0} base {gb:.4g} rep {rep} new {gn:.4g} cs {cs} src {src} old {old:.4g} -> {ans:.4g} ({how}) "
                  f"{'ACCEPT' if accept else 'skip'}")
            if accept:
                patch[str(i)] = dict(value=float(ans), why=f"rul147: occlusion dip {R_['dip']['t0']:.2f}-{R_['dip']['t1']:.2f}s "
                                                           f"interpolated, geo {gb:.3g}->{gn:.3g}, {how}")
    print(f'[{split}] videos measured {nv}, gated rows {nr}, dip records {sum(1 for k in rec if k.startswith(split))}, patched {len(patch)}')
    return patch, rec


if __name__ == '__main__':
    allrec = {}
    for sp in ('test', 'val'):
        p, rc = run(sp)
        allrec.update(rc)
        json.dump(p, open(f'{R}/cache/tmp/ag_c79occdip_patch_{sp}.json', 'w'), indent=1)
    json.dump(allrec, open(f'{R}/cache/tmp/ag_c79occdip_rec.json', 'w'), indent=1, default=str)
    for k, v in allrec.items():
        print('REC', k, v['video'], v['kind'], d_str(v['dip']), 'centre', round(v['centre_move'], 1), round(v['side'], 1), v['centre_ok'],
              'iou', round(v['iou'], 2), 'acc', v.get('accepted'), v.get('old'), '->', v.get('ans'), '|', v['q'][:60])
