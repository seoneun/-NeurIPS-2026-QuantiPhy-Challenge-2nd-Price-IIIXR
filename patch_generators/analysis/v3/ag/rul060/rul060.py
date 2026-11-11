"""rul060: knownsize path rejected only by its hard 200 m depth cap.

Scope (code conditions on inputs only; labels are never read here):
  * geo raw meas_why == knownsize_Z(z) (the 3D known-size path ran, but median Z > 200 m tripped the cap),
  * question kind 'size' asking a LENGTH (the extent along the direction of travel),
  * target phrase == prior-object phrase (the asked object is the moving prior object),
  * prior is a time-less speed v ('speed of X = v m/s', no t=),
  * z lies inside [0.8 x min, 1.25 x max] of the video's stated depth_info distances (the cap, not the geometry, failed),
  * the 4-box-window median and the whole-run ratio agree within 1.1x (a stable track; diag: sim_0030 eagle 1.25 vs 5.47),
  * lateral-dominant motion: radial speed / v <= 0.4, where the radial speed is the larger of the steepest timed depth
    key (|dd/dt| over any key given at >= 2 times) and the object's own |d ln(box extent)/dt| x median stated distance.
    (Oblique motion lets the cross-axis extent - wingspan, beam - leak into the along-motion box extent: sim_0009 boat
    (radial/v 0.88) over-reads 1.19x and the D3 'plane 22.16 m/s' renders (0.86) read 35 m vs 15 m in their D2 renders.)
Value: L = median over uncut 4-box windows of (box extent along the motion axis, px) / (px speed, px/s) x v.
  Focal-free and depth-free: for a body moving along its own axis, the image extent of the body and the image velocity
  are the same projection of the same 3D direction, so ext_px / v_px = L / |V| to first order (foreshortening and the
  perspective term cancel).
No CaW hedge: the D3 size->CaW route exists for dynamic-prior focal-calibration noise, which this value does not carry.
usage: python rul060.py [write]"""
import sys, os, json, re
from collections import defaultdict
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
WRITE = len(sys.argv) > 1 and sys.argv[1] == 'write'
RADMAX = 0.4
LEN = re.compile(r"\b(length|long)\b", re.I)


def load_parse(split):
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l)
        P[int(j['id'])] = j
    return P


def phrase(x):
    return (x or {}).get('phrase') if isinstance(x, dict) else x


def boxes(split, vid, ph):
    fn = f'{R}/geo/out/tracks_{split}_v2sc/{vid}.json'
    fn = fn if os.path.exists(fn) else f'{R}/geo/out/tracks_{split}/{vid}.json'
    if not os.path.exists(fn):
        return None, None
    T = json.load(open(fn))
    qs = [x for x in T.get('queries', []) if str(x.get('phrase', '')).lower() == ph.lower() and 'det_boxes' in x]
    if not qs:
        return None, T
    A = np.array([[f / T['fps']] + list(b) for f, b in zip(T['det_frames'], qs[0]['det_boxes']) if b is not None], float)
    return A, T


def estimate(A, W, H, v, win=4):
    A = A[(A[:, 1] > 1) & (A[:, 3] < W - 1) & (A[:, 2] > 1) & (A[:, 4] < H - 1)]      # boxes not cut by the border
    if len(A) < win + 1:
        return None
    cx, cy = (A[:, 1] + A[:, 3]) / 2, (A[:, 2] + A[:, 4]) / 2
    vx, vy = np.polyfit(A[:, 0], cx, 1)[0], np.polyfit(A[:, 0], cy, 1)[0]
    ext = (A[:, 3] - A[:, 1]) if abs(vx) >= abs(vy) else (A[:, 4] - A[:, 2])
    rat = []
    for a in range(len(A) - win + 1):
        s = slice(a, a + win)
        vv = np.hypot(np.polyfit(A[s, 0], cx[s], 1)[0], np.polyfit(A[s, 0], cy[s], 1)[0])
        if vv > 0:
            rat.append(np.median(ext[s]) / vv)
    if not rat:
        return None
    dlnext = np.polyfit(A[:, 0], np.log(ext), 1)[0]
    return dict(L=float(np.median(rat)) * v, Lglob=float(np.median(ext) / np.hypot(vx, vy) * v), n=len(A),
                dlnext=float(dlnext), vpx=float(np.hypot(vx, vy)), ext=float(np.median(ext)))


def radial_speed(j, ds, dlnext):
    K = defaultdict(list)
    for d in j.get('depth') or []:
        if d.get('t') is not None and d.get('d') is not None:
            K[d['key']].append((float(d['t']), float(d['d'])))
    rr = [abs((l[-1][1] - l[0][1]) / (l[-1][0] - l[0][0])) for l in (sorted(x) for x in K.values())
          if len(l) >= 2 and l[-1][0] > l[0][0]]
    rb = abs(dlnext) * float(np.median(ds))          # own box scale change x median stated distance (key-independent)
    if rr:
        return max(max(rr), rb), f'max of timed key {max(rr):.3g} and box scale {rb:.3g} m/s'
    return rb, 'box scale x stated depth'


def build(split, scope='knownsize'):
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    P = load_parse(split)
    out, log = {}, []
    why = raw.meas_why.astype(str)
    cand = raw.index[why.str.match(r'knownsize_Z\(')] if scope == 'knownsize' else raw.index
    for i in cand:
        j = P.get(int(i)) or {}
        q = j.get('q') or {}
        tp, pp = phrase(j.get('target')), phrase(j.get('prior_obj'))
        if q.get('kind') != 'size' or not tp or not pp or tp.lower() != pp.lower():
            continue
        if not (q.get('dim') == 'length' or LEN.search(str(raw.loc[i, 'question']))):
            log.append((int(i), 'not a length')); continue
        prs = j.get('priors') or []
        if len(prs) != 1 or prs[0].get('kind') != 'speed' or any(prs[0].get(k) is not None for k in ('t', 't1', 't2')) \
                or re.search(r'\bt\s*=|\bat\s+\d', str(raw.loc[i, 'ground_truth_prior'])):
            log.append((int(i), 'prior not a time-less speed')); continue
        v = float(prs[0]['value_si'])
        ds = [float(x) for x in re.findall(r'=\s*([\d.]+)\s*m\b', str(raw.loc[i, 'depth_info']))]
        if not ds:
            log.append((int(i), 'no stated distances')); continue
        if scope == 'knownsize':
            z = float(re.findall(r'knownsize_Z\(([\d.eE+-]+)\)', why[i])[0])
            if not (0.8 * min(ds) <= z <= 1.25 * max(ds)):
                log.append((int(i), f'z {z:g} outside stated {min(ds):g}-{max(ds):g}')); continue
        else:
            z = np.nan
        A, T = boxes(split, raw.loc[i, 'video_id'], tp)
        if A is None:
            log.append((int(i), 'no track')); continue
        e = estimate(A, T['W'], T['H'], v)
        if e is None:
            log.append((int(i), 'few uncut boxes')); continue
        if abs(np.log(e['L'] / e['Lglob'])) > np.log(1.1):
            log.append((int(i), f'window {e["L"]:.3g} vs global {e["Lglob"]:.3g} disagree')); continue
        rad, rsrc = radial_speed(j, ds, e['dlnext'])
        if rad / v > RADMAX:
            log.append((int(i), f'oblique: radial {rad:.3g} / v {v:g} = {rad / v:.2f} ({rsrc}); est would be {e["L"]:.3g}')); continue
        out[str(int(i))] = {'value': e['L'], 'why': (
            f'rul060: knownsize rejected only by the 200 m cap (Z {z:g} vs stated {min(ds):g}-{max(ds):g} m); focal-free '
            f'self-speed ratio ext/v_px ({e["ext"]:.0f} px / {e["vpx"]:.0f} px/s) x {v:g} m/s over {e["n"]} uncut boxes; '
            f'radial/v {rad / v:.2f} ({rsrc})')}
        log.append((int(i), f'FIRE {e["L"]:.3f} (global {e["Lglob"]:.3f}, radial/v {rad / v:.2f} {rsrc}, box-scale radial '
                            f'{abs(e["dlnext"]) * np.median(ds) / v:.2f})'))
    return out, log


if __name__ == '__main__':
    for split in ('test', 'val'):
        o, l = build(split)
        print(split, 'fires', len(o))
        for x in l:
            print('  ', x)
        if WRITE:
            json.dump(o, open(f'{R}/cache/tmp/ag_c39ksz_patch_{split}.json', 'w'), indent=1)
    # diagnostic only (never written): the same estimator on every same-object length row with a time-less speed prior
    for split in ('test', 'val'):
        o, l = build(split, scope='all')
        print('[diag all-scope]', split, 'would fire', len(o))
        for x in l:
            if x[1].startswith(('FIRE', 'oblique', 'window')):
                print('  ', x)
