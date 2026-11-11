import json, ast, numpy as np
TD = 'geo/out/tracks_test_qv3ntmv_mq2off'
def load(video, phrase, fn=None):
    j = json.load(open(fn or f'{TD}/{video}__all.json'))
    qs = [q for q in j['queries'] if q['phrase'].lower() == phrase.lower()] or [q for q in j['queries'] if phrase.lower() in q['phrase'].lower()]
    if not qs: raise SystemExit('')
    q = qs[0]; o = j['objs'][str(q['obj'])]
    fps = j['fps']; fidx = j['fidx']
    rows = []
    for f, e in zip(fidx, o):
        if not e: continue
        st = e.get('st') if isinstance(e, dict) else None
        if isinstance(st, str): st = ast.literal_eval(st)
        if not st: continue
        b = st['box']; rows.append((f, f / fps, st['c'][0], st['c'][1], b[2] - b[0], b[3] - b[1], st['area'], st.get('border', 0), st.get('pca', [np.nan, np.nan])))
    return j, q, rows
def fit(T, X, t, H, deg=2):
    m = np.abs(T - t) <= H + 1e-9
    if m.sum() < deg + 2: return np.nan, np.nan, int(m.sum())
    p = np.polyfit(T[m] - t, X[m], deg)
    return p[-2], 2 * p[-3] if deg >= 2 else 0.0, int(m.sum())
def kin(rows, t, H):
    T = np.array([r[1] for r in rows]); X = np.array([r[2] for r in rows]); Y = np.array([r[3] for r in rows])
    vx, ax, n = fit(T, X, t, H); vy, ay, _ = fit(T, Y, t, H)
    return np.hypot(vx, vy), np.hypot(ax, ay), n, (vx, vy, ax, ay)
