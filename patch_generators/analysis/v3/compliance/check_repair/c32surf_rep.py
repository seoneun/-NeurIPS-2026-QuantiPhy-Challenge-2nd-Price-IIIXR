"""c32surf_rep.py (check_repair grpE, 2026-10-17): analysis/v3/ag/rul048/rul048.py (the production builder of tag rul048;
05_patch_rules.sh:351) with two changes for the per-question rule:
  (1) the base submission is argv[1] (default sub_xt_Nzt, the production code default), so rerun_clean.sh conv argv1 works;
  (2) the mpp fallback 'video's 2D mpp from sibling rows' (video_mpp(): median meas_mpp of the OTHER rows of the same video,
      i.e. scales derived from other questions' priors) is removed; mpp = own row meas_mpp, else 9.8/|a| of the own track.
Everything else (gates, gravity axis, rest surface, time-of-flight check, why text) is unchanged.
--- original docstring follows ---
rul048: gap between the bottom of a dropped object and the support surface it comes to rest on (label-free).
Gates: gravity prior; question 'distance between the bottom of X and <Y> at t'; target2 parsed; geo NaN;
answer_source in {caw_nogeo, dis_caw_routed, caw_c1}.
Measurement on the target's v2sc track (time = fidx / CSV fps):
  1. gravity axis: quadratic fits of the box centre in sliding 6-sample windows; keep windows with pixel speed >= 300 px/s
     and residual <= 12 px (need >= 4); the image axis whose median 2nd derivative is > 3x the other's is the fall axis.
  2. lead edge = box side toward gravity (signed so that larger = closer to the surface).
  3. surface = median lead edge over rest frames: |d lead/dt| < 60 px/s, lead in the top 25% of its 5-95% range,
     second half of the clip.
  4. value = (surface - lead(t)) * mpp * unit; mpp = row meas_mpp, else the video's 2D mpp from sibling rows
     (hgt_B precedent), else 9.8/|a| from step 1. Fires only when the gap > 0.
  5. scale-free cross-check (gravity time of flight): on the flight segment containing t (between contacts / static
     frames) fit lead = c2 t^2 + c1 t + c0; vertex t_v, contact time t_c where the fit reaches the surface;
     gap_tof = g/2 [(t_c - t_v)^2 - (t - t_v)^2]. This uses only times and g, not the pixel scale.
     Fire only if 1/1.5 < value / gap_tof < 1.5.
For dist2 rows the pipeline fusion takes geo whether CaW agrees (agree_geo_kind) or not (strong 2d mode), so the
patched value is the measurement itself.
usage: python rul048.py -> cache/tmp/ag_c32surf_patch_{test,val}.json"""
import json, re, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzt"
UNIT = {'m': 1.0, 'meters': 1.0, 'meter': 1.0, 'cm': 100.0, 'mm': 1000.0}
G = 9.8


def parse(split):
    d = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        d[j['id']] = j
    return d


def load_track(split, vid, phrase, fps):
    try:
        T = json.load(open(f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json"))
    except Exception:
        return None
    phrase = re.sub(r"^(the )?(bottom|top|center|centre|base) of (the )?", "", str(phrase or ""))
    obs = [q.get('obj') for q in T['queries'] if q.get('phrase') == phrase and q.get('obj') is not None]
    if not obs:
        obs = [q.get('obj') for q in T['queries'] if phrase and phrase in str(q.get('phrase')) and q.get('obj') is not None]
    if not obs:
        return None
    A = [(f / fps, *s['st']['box']) for f, s in zip(T['fidx'], T['objs'][str(obs[0])]) if s is not None and s.get('st')]
    return np.array(A) if len(A) else None


def gravity_axis(A):
    t = A[:, 0]
    cx = (A[:, 1] + A[:, 3]) / 2
    cy = (A[:, 2] + A[:, 4]) / 2
    axs, ays = [], []
    for k0 in range(0, len(t) - 6):
        tt = t[k0:k0 + 6] - t[k0]
        vx = np.polyfit(tt, cx[k0:k0 + 6], 2)
        vy = np.polyfit(tt, cy[k0:k0 + 6], 2)
        if np.hypot(vx[1], vy[1]) < 300:
            continue
        res = np.abs(np.polyval(vx, tt) - cx[k0:k0 + 6]).max() + np.abs(np.polyval(vy, tt) - cy[k0:k0 + 6]).max()
        if res > 12:
            continue
        axs.append(2 * vx[0])
        ays.append(2 * vy[0])
    if len(axs) < 4:
        return None
    ax, ay = float(np.median(axs)), float(np.median(ays))
    if abs(ax) > 3 * abs(ay):
        return (A[:, 3] if ax > 0 else -A[:, 1]), abs(ax), ('x+' if ax > 0 else 'x-'), abs(ay), len(axs)
    if abs(ay) > 3 * abs(ax):
        return (A[:, 4] if ay > 0 else -A[:, 2]), abs(ay), ('y+' if ay > 0 else 'y-'), abs(ax), len(axs)
    return None


def rest_surface(t, lead):
    dl = np.abs(np.gradient(lead, t))
    lo, hi = np.percentile(lead, 5), np.percentile(lead, 95)
    rest = np.where((dl < 60) & (lead > hi - 0.25 * (hi - lo)) & (t > t[len(t) // 2]))[0]
    if len(rest) < 5:
        return None, 0
    return float(np.median(lead[rest])), len(rest)


def tof_gap(t, lead, surf, tq):
    """Scale-free gap from gravity time of flight on the flight segment containing tq."""
    n = len(t)
    kq = int(np.argmin(np.abs(t - tq)))
    st = np.abs(np.diff(lead))
    static = np.zeros(n, bool)
    static[1:-1] = (st[:-1] < 2) & (st[1:] < 2)
    contact = np.zeros(n, bool)
    contact[1:-1] = (lead[1:-1] >= lead[:-2]) & (lead[1:-1] >= lead[2:]) & (lead[1:-1] >= surf - 60)
    contact |= lead >= surf - 10
    bnd = static | contact
    if bnd[kq]:
        return None
    a = kq
    while a - 1 >= 0 and not bnd[a - 1]:
        a -= 1
    b = kq
    while b + 1 < n and not bnd[b + 1]:
        b += 1
    idx = np.arange(a, b + 1)
    for _ in range(3):
        if len(idx) < 5:
            return None
        c = np.polyfit(t[idx], lead[idx], 2)
        r = np.abs(np.polyval(c, t[idx]) - lead[idx])
        if r.max() <= 8:
            break
        idx = np.delete(idx, np.argmax(r))
    else:
        return None
    if c[0] <= 0:
        return None
    tv = -c[1] / (2 * c[0])
    roots = np.roots([c[0], c[1], c[2] - surf])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots >= tq - 1e-6]
    if not len(roots):
        return None
    tc = float(roots.min())
    gap = 0.5 * G * ((tc - tv) ** 2 - (tq - tv) ** 2)
    return gap, tv, tc, 2 * c[0], len(idx), (t[idx[0]], t[idx[-1]])


def video_mpp(raw, vid):
    s = raw[(raw.video_id == vid) & raw.meas_mode.astype(str).str.startswith('2d')]
    m = pd.to_numeric(s.meas_mpp, errors='coerce')
    m = m[np.isfinite(m)]
    return (float(m.median()), len(m)) if len(m) else (None, 0)


def measure(split, raw, det, PP, i, gated=True):
    r = raw.iloc[i]
    j = PP.get(i)
    if j is None:
        return None, 'noparse'
    pt = str(r.ground_truth_prior).lower()
    if not ('gravit' in pt or '9.8' in pt):
        return None, 'prior'
    q = j['q']
    unit = UNIT.get(str(q.get('unit')).lower())
    qq = str(r.question).lower()
    if unit is None:
        unit = 100.0 if ' in cm' in qq else 1.0
    src = str(det.loc[i, 'answer_source']).replace('mo_guard:', '')
    if gated:
        if src not in ('caw_nogeo', 'dis_caw_routed', 'caw_c1') or np.isfinite(r.geo_value):
            return None, 'src/geo'
        if not re.search(r"distance between the bottom of", qq) or not q.get('target2'):
            return None, 'phrase'
    if q.get('t') is None:
        return None, 'no_t'
    fps = float(r.fps)
    A = load_track(split, r.video_id, q.get('target'), fps)
    if A is None or len(A) < 20:
        return None, 'track'
    ga = gravity_axis(A)
    if ga is None:
        return None, 'axis'
    lead, amag, axis, aminor, nwin = ga
    t = A[:, 0]
    surf, nrest = rest_surface(t, lead)
    if surf is None:
        return None, 'rest'
    mm = pd.to_numeric(r.meas_mpp, errors='coerce')
    if np.isfinite(mm):   # check_repair: no sibling-row (other question) mpp fallback
        mpp, msrc = float(mm), 'row'
    else:
        mpp, msrc = G / amag, 'grav'
    lt = float(np.interp(q['t'], t, lead))
    gap = (surf - lt) * mpp * unit
    if not gap > 0:
        return None, 'gap<=0'
    tf = tof_gap(t, lead, surf, q['t'])
    info = dict(axis=axis, a=amag, aminor=aminor, nwin=nwin, surf=abs(surf), nrest=nrest, lead_t=abs(lt), mpp=mpp,
                msrc=msrc, mpp_grav=G / amag, unit=unit, src=src, old=float(det.loc[i, 'answer']),
                tof=None if tf is None else tf[0] * unit, tof_detail=tf)
    if tf is None:
        return (gap, info), 'no_tof'
    ratio = gap / (tf[0] * unit)
    info['ratio'] = ratio
    if not (1 / 1.5 < ratio < 1.5):
        return (gap, info), 'tof_disagree'
    return (gap, info), 'ok'


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv")
    PP = parse(split)
    out = {}
    for i in range(len(raw)):
        res, st = measure(split, raw, det, PP, i)
        if res is None or st != 'ok':
            if res is not None:
                print(f"  [{split}] {i} not fired: {st} {res[0]:.4g} {res[1]}")
            continue
        v, d = res
        tf = d['tof_detail']
        out[str(i)] = {'value': float(v), 'why': (
            f"rul048: gravity axis {d['axis']} (a={d['a']:.0f}px/s^2, minor {d['aminor']:.0f}, n={d['nwin']}); "
            f"surface = rest lead edge {d['surf']:.0f}px (n{d['nrest']}); lead(t)={d['lead_t']:.0f}px; mpp {d['mpp']:.6f} "
            f"({d['msrc']}); gravity time-of-flight check {d['tof']:.4g} (vertex {tf[1]:.3f}s, contact {tf[2]:.3f}s); "
            f"was {d['old']:.4g} {d['src']}")}
    json.dump(out, open(f"{R}/cache/tmp/ag_c32surf_patch_{split}.json", 'w'), indent=1)
    print(f"== {split}: {len(out)} rows")
    for k, v in out.items():
        print('  ', k, round(v['value'], 4), v['why'])
    return out


if __name__ == '__main__':
    for s in ('test', 'val'):
        run(s)
