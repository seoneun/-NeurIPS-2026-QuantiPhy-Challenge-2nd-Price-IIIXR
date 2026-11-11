"""rul059: a held (duplicated) first frame in a simulation render biases the early-time local velocity fit low.
Rule (label-free, code conditions only):
  scope: simulation video whose frame 0 is a held copy of frame 1 (MAD(f0,f1) < 0.3 x median MAD of the next three
         frame pairs); question kind speed/hvel at t <= 0.3 s ('initial speed' = t 0); NOT a kin row; answer source
         geo-derived (agree_*, dis_geo, geo_*, dis_size_gm); planar scene (no depth_info) because the scale-free step
         assumes pixel motion proportional to world motion.
  step 1: drop the leading held frame(s) from the target track.
  step 2: target == prior object and the prior is a time-less non-gravity acceleration -> one quadratic on the uncut
          boxes of frames >= n_held (R^2 >= 0.995, span >= 20 px, >= 12 boxes): v(t) = a_prior * v_px(t) / a_px
          (scale-free; only the CSV fps enters).  Otherwise scale geo by k = v_px(held dropped) / v_px(all frames)
          (local fit of measure_v3.vel_at_c), only when |ln k| > 0.1.
  step 3: re-fuse: new geo within 2x of CaW -> geometric mean, else new geo (strong 2d mode keeps geo).
  copies: rows that took this plain row's answer through --var_ans_plain (same base recording, same old answer,
          source var_ans_plain) take the new answer, as the pipeline's patch-copy step intends (its key does not link
          "initial speed" to "speed at 0.0s").
Writes cache/tmp/ag_c39dup0_patch_{test,val}.json."""
import sys, os, json, re, math
import numpy as np, pandas as pd, cv2
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/geo'); sys.path.insert(0, R + '/analysis/v3')
import measure_v3 as MV
from variant_keys import split_variant
SUB = os.environ.get("C39_SUB", "sub_xt_Nzx")
RAWN = "geo_v4ntrj5sQ2srngsr_d_vw"
SRC_OK = re.compile(r'^(agree_|dis_geo|geo_|dis_size_gm)')
NO2D = os.environ.get("C39_ALLOW3D", "") == ""

_dc = {}
def held_frames(path):
    """(ratio d01/base, number of leading held frames) from the first 8 decoded frames."""
    if path in _dc: return _dc[path]
    cap = cv2.VideoCapture(path); F = []
    for _ in range(8):
        ok, f = cap.read()
        if not ok: break
        F.append(f.astype(np.float32))
    cap.release()
    if len(F) < 5:
        _dc[path] = (None, 0); return _dc[path]
    d = [float(np.mean(np.abs(F[k + 1] - F[k]))) for k in range(len(F) - 1)]
    nd = 0
    while nd < len(d) - 3:
        base = float(np.median(d[nd + 1:nd + 4])) + 1e-6
        if d[nd] < 0.3 * base: nd += 1
        else: break
    base0 = float(np.median(d[1:4])) + 1e-6
    _dc[path] = (d[0] / base0, nd); return _dc[path]

def load_track(sp, vid, phrase, pref):
    order = [pref] + [x for x in ('v2sc', 'v2full', 'v2', '') if x != pref]
    for tv in order:
        p = f"{R}/geo/out/tracks_{sp}{'_' + tv if tv else ''}/{vid}.json"
        if not os.path.exists(p): continue
        j = json.load(open(p)); best = None
        for q in j['queries'] + j.get('queries_d', []):
            if q.get('obj') is None or q.get('phrase') != phrase: continue
            seq = j['objs'].get(str(q['obj']))
            if seq is None: continue
            pts = [(fi / j['fps'], *s['st']['c'], *s['st']['box'], fi) for fi, s in zip(j['fidx'], seq) if s and s.get('st')]
            if best is None or len(pts) > len(best): best = pts
        if best: return np.array(best, float), j['W'], j['H'], tv
    return None, None, None, None

def vpx(A, t0, horiz):
    t = A[:, 0]; P = A[:, 1:3]; sz = float(np.median(np.maximum(A[:, 5] - A[:, 3], A[:, 6] - A[:, 4])))
    sg = MV.segment(t, P, sz); v = MV.vel_at_c(t, P, t0, sg)
    if v is None: return None
    return float(abs(v[0])) if horiz else float(np.linalg.norm(v))

def run(sp):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if sp == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    RAW = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", usecols=['Unnamed: 0', 'video_path', 'fps', 'depth_info', 'meas_tracks']).set_index('Unnamed: 0')
    out = {}; log = []
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); i = r['id']; vid = r['video_id']
        if not vid.startswith('simulation') or i not in D.index or i not in RAW.index: continue
        q = r['q']; kind = q.get('kind')
        if kind not in ('speed', 'hvel'): continue
        t0 = q.get('t')
        if t0 is None and q.get('when') == 'start' and q.get('t1') is None: t0 = 0.0
        if t0 is None or float(t0) > 0.3: continue
        t0 = float(t0)
        d = D.loc[i]; src = str(d.get('answer_source', '')).replace('mo_guard:', '')
        if not SRC_OK.match(src): continue
        ratio, nd = held_frames(RAW.loc[i, 'video_path'])
        if ratio is None or nd == 0:
            continue
        if NO2D and isinstance(RAW.loc[i, 'depth_info'], str) and RAW.loc[i, 'depth_info'].strip():
            log.append((i, vid, 'skip: depth_info (3D)')); continue
        g = d.get('g'); cs = d.get('cs')
        if not (isinstance(g, float) and g > 0): continue
        tp = (r.get('target') or {}).get('phrase')
        mt = str(RAW.loc[i, 'meas_tracks']); pref = 'v2sc' if mt == 'v2' else 'v2full'
        A, W, H, tv = load_track(sp, vid, tp, pref)
        if A is None or len(A) < 8:
            log.append((i, vid, 'skip: no target track')); continue
        B = A[A[:, 7] >= nd]
        pr = (r.get('priors') or [{}])[0]
        same = (r.get('prior_obj') or {}).get('phrase') == tp
        horiz = kind == 'hvel'
        newg = None; how = ''
        if same and not horiz and pr.get('kind') == 'accel' and pr.get('t') is None and pr.get('t1') is None and not pr.get('gravity'):
            unc = (B[:, 3] > 1) & (B[:, 4] > 1) & (B[:, 5] < W - 1) & (B[:, 6] < H - 1)
            C = B[unc]
            if len(C) >= 12:
                t = C[:, 0]; cx = np.polyfit(t, C[:, 1], 2); cy = np.polyfit(t, C[:, 2], 2)
                res = sum(float(np.sum((C[:, k] - np.polyval(c, t)) ** 2)) for k, c in ((1, cx), (2, cy)))
                tot = sum(float(np.sum((C[:, k] - C[:, k].mean()) ** 2)) for k in (1, 2))
                r2 = 1 - res / tot if tot > 0 else 0
                apx = 2 * math.hypot(cx[0], cy[0])
                v0 = math.hypot(np.polyval(np.polyder(cx), t0), np.polyval(np.polyder(cy), t0))
                span = float(np.hypot(*(C[-1, 1:3] - C[0, 1:3])))
                if r2 >= 0.995 and apx > 0 and span >= 20:
                    newg = abs(pr['value_si']) * v0 / apx; how = f'quad r2={r2:.4f} v0px={v0:.1f} apx={apx:.1f} n={len(C)} trk={tv}'
                else:
                    how = f'quad rejected r2={r2:.4f} span={span:.0f}; '
        if newg is None:
            v_all = vpx(A, t0, horiz); v_d = vpx(B, t0, horiz)
            if v_all and v_d:
                k = v_d / v_all
                if abs(math.log(k)) > 0.1: newg = g * k; how += f'k={k:.3f} trk={tv}'
                else: how += f'k={k:.3f} small'
        if newg is None:
            log.append((i, vid, 'nochange', how)); continue
        if isinstance(cs, float) and cs > 0 and max(newg, cs) / min(newg, cs) <= 2:
            val = math.sqrt(newg * cs); fus = 'gm'
        else:
            val = newg; fus = 'geo'
        out[str(i)] = {'value': float(val), 'why': f'rul059 held frame0 (n={nd}, r={ratio:.3f}); {how}; newgeo {newg:.4g} {fus} cs {cs} (old {d.answer:.4g} {src})'}
        log.append((i, vid, round(float(d.answer), 4), round(newg, 4), round(val, 4), how))
    # var_ans_plain copies of a patched plain row
    sv = D.video_id.astype(str).map(split_variant)
    base = sv.str[0]; var = sv.str[1]
    for k_ in list(out):
        i = int(k_)
        if var[i] != 'plain': continue
        old = float(D.loc[i, 'answer'])
        m = (base == base[i]) & (var != 'plain') & D.answer_source.astype(str).str.replace('mo_guard:', '').eq('var_ans_plain') & (np.abs(D.answer.astype(float) - old) < 1e-9 * max(1, old))
        for j in D.index[m]:
            out[str(j)] = {'value': out[k_]['value'], 'why': f'rul059 var_ans_plain copy of {i}'}
            log.append((j, D.loc[j, 'video_id'], 'copy of', i, round(old, 4), round(out[k_]['value'], 4)))
    for x in log: print(sp, x)
    if os.environ.get("C39_DRY", "") == "":
        json.dump(out, open(f"{R}/cache/tmp/ag_c39dup0_patch_{sp}.json", "w"), indent=1)
    return out

if __name__ == '__main__':
    for sp in ('test', 'val'):
        o = run(sp); print(sp, 'patched', len(o))
