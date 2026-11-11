"""rul107: a render that HOLDS a frame mid-clip (frame i pixel-identical to frame i-1) puts a fake zero-motion
sample into the target track. MV.segment's BIC (tiny sigma on clean sims) splits the trajectory at that sample and
vel_at_c then fits a short or kinked piece, so the local velocity at t comes out biased.
Rule (label-free, code conditions only):
  held(vid) = {i : mean|gray_i - gray_{i-1}| < 0.02  and  median_j mean|gray_j - gray_{j-1}| > 0.1}
             (gray of the frames returned by tc.load_video).
  scope : simulation video; parse kind speed/hvel/vvel with a single time t (no t1/t2; never accel kinds);
          answer source geo-fused (agree_*, dis_geo, geo_*), not already overridden by an accepted patch.
  track : v2sc if meas_tracks=='v2' else v3ntmv, then v2full; among queries whose phrase equals the parsed target or
          shares its head noun, the one with the largest path length (>= 12 boxes).
  k     = |v(t) held frames dropped| / |v(t) all frames|, MV.segment re-run on each set, MV.vel_at_c.
  gates : 1 <= #held frames with |t_h - t| < 0.35 s <= 2; #segments after drop <= before; |ln k| > 0.1; 0.5 < k < 2.
  output: g' = g k; within 2x of CaW -> geometric mean (agree_wgm weights kept if the row used them), else mode routing
          (2d_ratio / 3d_calib keep geo, other modes keep CaW -> no change).
  copies: var_ans_plain copies of a patched plain row take the new answer.
Writes cache/tmp/ag_c62held_patch_{test,val}.json."""
import sys, os, json, re, math
import numpy as np, pandas as pd, cv2
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/geo'); sys.path.insert(0, R + '/analysis/v3')
import measure_v3 as MV
import track_common as tc
from variant_keys import split_variant
SUB = os.environ.get("C62_SUB", "sub_xt_Nzz12")
RAWN = "geo_v4ntrj5sQ2srngsr_d_vw"
SRC_OK = re.compile(r'^(agree_|dis_geo|geo_)')
WIN = float(os.environ.get("C62_WIN", "0.35"))
ALLV = os.environ.get("C62_ALLV", "") != ""

_hc = {}
def held(path):
    if path in _hc: return _hc[path]
    F = tc.load_video(path); d = []
    prev = None
    for f in F:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.int16)
        if prev is not None: d.append(float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(d); med = float(np.median(d)) if len(d) else 0.0
    h = [i + 1 for i, x in enumerate(d) if x < 0.02 and med > 0.1]
    _hc[path] = (h, med); return _hc[path]

def best_track(sp, vid, target, mt, fps):
    order = ['v2sc', 'v3ntmv', 'v2full'] if mt == 'v2' else ['v3ntmv', 'v2full']
    for dd in order:
        fn = f'{R}/geo/out/tracks_{sp}_{dd}/{vid}.json'
        if not os.path.exists(fn): continue
        j = json.load(open(fn)); best = None; bp = -1; bph = None
        for qq in j['queries'] + j.get('queries_d', []):
            ph = qq.get('phrase')
            if qq.get('obj') is None or not ph: continue
            if ph != target and ph.split()[-1] != target.split()[-1]: continue
            seq = j['objs'].get(str(qq['obj']))
            if not seq: continue
            pts = [(fi, fi / fps, *s['st']['c'], *s['st']['box']) for fi, s in zip(j['fidx'], seq) if s and s.get('st')]
            if len(pts) < 12: continue
            pa = np.array(pts, float); path = float(np.linalg.norm(np.diff(pa[:, 2:4], axis=0), axis=1).sum())
            if path > bp: best, bp, bph = pa, path, ph
        if best is not None: return best, bph, dd
    return None, None, None

def comp(v, kind):
    if kind == 'hvel': return abs(float(v[0]))
    if kind == 'vvel': return abs(float(v[1]))
    return float(np.linalg.norm(v))

def run(sp):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if sp == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    RAW = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", usecols=['Unnamed: 0', 'video_path', 'fps', 'meas_tracks', 'meas_mode']).set_index('Unnamed: 0')
    out = {}; log = []
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); i = r['id']; vid = r['video_id']
        if i not in D.index or i not in RAW.index: continue
        if not ALLV and not vid.startswith('simulation'): continue
        q = r['q']; kind = q.get('kind')
        if kind not in ('speed', 'hvel', 'vvel') or q.get('t') is None or q.get('t1') is not None or q.get('t2') is not None: continue
        t0 = float(q['t']); tgt = q.get('target') or (r.get('target') or {}).get('phrase')
        if not tgt: continue
        d = D.loc[i]; src = str(d.get('answer_source', '')).replace('mo_guard:', '')
        rr = RAW.loc[i]
        h, med = held(rr.video_path)
        if not h: continue
        fps = float(rr.fps)
        nwin = int(np.sum(np.abs(np.array(h) / fps - t0) < WIN))
        if nwin < 1: continue
        A, ph, dd = best_track(sp, vid, tgt, str(rr.meas_tracks), fps)
        if A is None: log.append((i, vid, 'no track')); continue
        fi = A[:, 0].astype(int); t = A[:, 1]; P = A[:, 2:4]
        sz = float(np.median(np.maximum(A[:, 6] - A[:, 4], A[:, 7] - A[:, 5])))
        dm = np.isin(fi, h)
        if dm.sum() == 0: continue
        try:
            sg = MV.segment(t, P, sz); va = MV.vel_at_c(t, P, t0, sg)
            sg2 = MV.segment(t[~dm], P[~dm], sz); vb = MV.vel_at_c(t[~dm], P[~dm], t0, sg2)
        except Exception as e:
            log.append((i, vid, 'err', str(e))); continue
        if va is None or vb is None: continue
        a, b = comp(va, kind), comp(vb, kind)
        if a <= 0: continue
        k = b / a; n1, n2 = len(sg['segs']), len(sg2['segs'])
        info = f'held={h[:8]} nwin={nwin} segs {n1}->{n2} k={k:.3f} trk={dd}:{ph}'
        gates = []
        if not (1 <= nwin <= 2): gates.append('nwin')
        if n2 > n1: gates.append('segs')
        if not (abs(math.log(k)) > 0.1 and 0.5 < k < 2): gates.append('k')
        if src.startswith('patch:'): gates.append('patched')
        if not SRC_OK.match(src): gates.append('src')
        g = d.get('g'); cs = d.get('cs')
        if not (isinstance(g, float) and g > 0): gates.append('nog')
        if gates:
            log.append((i, vid, d['cat'], kind, t0, 'GATED', ','.join(gates), info, src, round(float(d.answer), 4))); continue
        newg = g * k
        mode = str(d.get('mode', rr.meas_mode))
        if isinstance(cs, float) and cs > 0 and max(newg, cs) / min(newg, cs) <= 2:
            if src.startswith('agree_wgm') and abs(float(d.answer) - math.sqrt(g * cs)) > 1e-6 * float(d.answer) and g != cs:
                # keep the row's own agree weight: answer = g^w cs^(1-w)
                w = math.log(float(d.answer) / cs) / math.log(g / cs)
                val = newg ** w * cs ** (1 - w); fus = f'wgm w={w:.2f}'
            else:
                val = math.sqrt(newg * cs); fus = 'gm'
        elif mode in ('2d_ratio', '3d_calib'):
            val = newg; fus = 'geo'
        else:
            log.append((i, vid, 'routes to CaW, no change', info)); continue
        out[str(i)] = {'value': float(val), 'why': f'rul107 held render frames dropped from target track ({info}); geo {g:.4g}->{newg:.4g}, {fus} with CaW {cs} (old {float(d.answer):.4g} {src})'}
        log.append((i, vid, d['cat'], kind, t0, 'PATCH', info, src, round(float(d.answer), 4), round(newg, 4), round(val, 4)))
    sv = D.video_id.astype(str).map(split_variant); base = sv.str[0]; var = sv.str[1]
    for k_ in list(out):
        i = int(k_)
        if var[i] != 'plain': continue
        old = float(D.loc[i, 'answer'])
        m = (base == base[i]) & (var != 'plain') & D.answer_source.astype(str).str.replace('mo_guard:', '').eq('var_ans_plain') & (np.abs(D.answer.astype(float) - old) < 1e-9 * max(1, old))
        for j in D.index[m]:
            out[str(j)] = {'value': out[k_]['value'], 'why': f'rul107 var_ans_plain copy of {i}'}
            log.append((j, D.loc[j, 'video_id'], 'copy of', i))
    for x in log: print(sp, x)
    print(sp, 'videos decoded', len(_hc), 'with held frames', sum(1 for v in _hc.values() if v[0]))
    if os.environ.get("C62_DRY", "") == "":
        json.dump(out, open(f"{R}/cache/tmp/ag_c62held_patch_{sp}.json", "w"), indent=1)
    return out

if __name__ == '__main__':
    for sp in ('test', 'val'):
        o = run(sp); print(sp, 'patched', len(o), list(o)[:20])
