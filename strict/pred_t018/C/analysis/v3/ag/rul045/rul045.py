import sys, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
K, THR, LO = 0.8, 1.5, 0.8


def keypts(p):
    tk = set(p.get('target_dkeys') or [])
    return sorted({(float(x['t']), float(x['d'])) for x in (p.get('depth') or [])
                   if isinstance(x, dict) and x.get('key') in tk and x.get('t') is not None and x.get('d') is not None})


def track_xy(sp, vid, phrase):
    for sub in ('_v3nt', '_v2', ''):
        p = f"{R}/geo/out/tracks_{sp}{sub}/{vid}.json"
        if os.path.exists(p):
            break
    else:
        return None
    T = json.load(open(p))
    kn = str(phrase).replace('_', '').replace(' ', '').lower()
    qs = list(T.get('queries_d') or []) + list(T.get('queries') or [])
    ob = next((e['obj'] for e in qs if kn and (kn in str(e.get('phrase')).replace(' ', '').lower() or str(e.get('phrase')).replace(' ', '').lower() in kn)), None)
    if ob is None or str(ob) not in T['objs']:
        return None
    O = T['objs'][str(ob)]
    A = np.array([(f / T['fps'], O[k]['st']['c'][0], O[k]['st']['c'][1]) for k, f in enumerate(T['fidx']) if O[k] and O[k].get('st')])
    return (A, T['W'], T['H']) if len(A) else None


def straight_line(tr, t0, z0, t1, z1, wend=0.167):
    A, W, H = tr
    f = 0.72 * W

    def c_at(t):
        j = np.abs(A[:, 0] - t) <= 0.025
        return A[j][:, 1:3].mean(0) if j.any() else None
    c0, c1 = c_at(t0), c_at(t1)
    if c0 is None or c1 is None:
        return np.nan, np.nan, np.nan
    P0 = np.array([(c0[0] - W / 2) / f * z0, (c0[1] - H / 2) / f * z0, z0])
    P1 = np.array([(c1[0] - W / 2) / f * z1, (c1[1] - H / 2) / f * z1, z1])
    S = np.linspace(-0.3, 1.3, 1601)
    PS = P0[None] + S[:, None] * (P1 - P0)[None]
    UV = np.c_[PS[:, 0] / PS[:, 2] * f + W / 2, PS[:, 1] / PS[:, 2] * f + H / 2]
    m = (A[:, 0] >= t0 - 1e-6) & (A[:, 0] <= t1 + 1e-6)
    ss, res = [], []
    for row in A[m]:
        d2 = ((UV - row[1:3]) ** 2).sum(1)
        k = int(np.argmin(d2)); ss.append(S[k]); res.append(np.sqrt(d2[k]))
    ss, res, tt = np.array(ss), np.array(res), A[m][:, 0]
    span = float(np.hypot(*(c1 - c0)))
    j1 = tt >= t1 - wend - 1e-6
    r_end = float(np.polyfit(tt[j1], ss[j1], 1)[0] * (t1 - t0)) if j1.sum() >= 3 else np.nan
    return float(np.median(res)) / max(span, 1e-6), r_end, span


def candidates(sp):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv", index_col=0)
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    rows = []
    for i in det.index:
        p = PP.get(i)
        q = (p or {}).get('q') or {}
        if p is None or str(det.cat[i]) not in ('S3', 'D3') or q.get('kind') != 'speed' or q.get('t') is None:
            continue
        if not str(raw.video_source.get(i, '')).lower().startswith('lab'):
            continue
        if 'patch' in str(det.answer_source[i]):
            continue
        if any(isinstance(x, dict) and (x.get('kind') == 'accel' or x.get('gravity')) for x in (p.get('priors') or [])):
            continue
        g = pd.to_numeric(det.g[i], errors='coerce')
        a = float(det.answer[i])
        pts = keypts(p)
        ts = sorted({x[0] for x in pts})
        if len(pts) != 2 or len(ts) != 2 or not np.isfinite(g) or g <= 0:
            continue
        t = float(q['t'])
        if abs(t - ts[1]) > 1e-6:
            continue
        (t0, z0), (t1, z1) = pts
        u = str(q.get('unit') or '')
        vr = abs(z1 - z0) / (t1 - t0) * (100 if u.startswith('cm') else (1000 if u.startswith('mm') else 1))
        if not vr > 0:
            continue
        vid = str(raw.video_id.get(i, ""))
        tr = track_xy(sp, vid, q.get('target') or '')
        relres, r_end, span = straight_line(tr, t0, z0, t1, z1) if tr is not None else (np.nan, np.nan, np.nan)
        uniform = bool(np.isfinite(relres) and span >= 20 and relres <= 0.05 and r_end >= 0.9)
        rows.append(dict(rid=int(i), ans=a, t1=t1, ratio=float(g) / vr, uniform=uniform))
    return pd.DataFrame(rows)


C = candidates('test')
sel = C[(C.ratio > LO) & (C.ratio <= THR) & ~C.uniform]
out = {str(r.rid): dict(value=r.ans * K, why='t147')
       for r in sel.itertuples()}
json.dump(out, open(f"{R}/cache/tmp/ag_rul045_patch_test.json", "w"), indent=0)
print('[test] rul045: rows', len(out))
