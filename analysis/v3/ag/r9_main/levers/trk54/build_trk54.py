import sys, re, json, numpy as np, pandas as pd
D = 'analysis/v3/ag/r9_main/levers/trk54/in'
FW = 0.596
FIX = ['captured_0054a', 'captured_0054as', 'captured_0054ax', 'captured_0054b']
T = pd.read_parquet('data/test/test_dataset.parquet').reset_index(drop=True)
T['id0'] = T.index
T = T[T.video_id.isin(FIX)]


def track(v, phrase, fallback):
    j = json.load(open(f'{D}/tracks/{v}.json'))
    if phrase in j['objs']:
        return j['objs'][phrase], j['fps'], j['W'], j['H']
    if not fallback:
        return None
    j = json.load(open(f'geo/out/tracks_test_qv3ntmv_mq2off/{v}__all.json'))
    q = {p['phrase']: p['obj'] for p in j['queries']}
    if q.get(phrase) is None:
        return None
    return j['objs'][str(q[phrase])], j['fps'], j['W'], j['H']


def pos(tr, tk, Zk):
    o, fps, W, H = tr
    c = np.full((len(o), 2), np.nan); d = np.full(len(o), np.nan)
    for i, e in enumerate(o):
        st = e.get('st') if isinstance(e, dict) else None
        if st and st.get('area', 0) > 0 and not st.get('border'):
            c[i] = st['c']; d[i] = np.sqrt(4 * st['area'] / np.pi)
    nk = int(round(tk * fps)); dk = np.nanmedian(d[max(0, nk - 1):nk + 2]); fp = FW * W
    Z = Zk * dk / d; X = (c[:, 0] - W / 2) * Z / fp; Y = (c[:, 1] - H / 2) * Z / fp
    return np.stack([X, Y, Z], 1), fps


def keys(r):
    return {m.group(2).replace('_', ' '): (float(m.group(1)), float(m.group(3)))
            for m in re.finditer(r't=([\d.]+)s, distance_([a-z_]+?)_camera = ([\d.]+)m', r.depth_info)}


rows = []
for _, r in T.iterrows():
    q = r.question; K = keys(r)
    mv = re.search(r'velocity of the ([a-z ]+?) at ([\d.]+)s', q)
    md = re.search(r'distance between the ([a-z ]+?) and the ([a-z ]+?) at ([\d.]+)s', q)
    new = np.nan
    if mv:
        ph, t = mv.group(1), float(mv.group(2))
        kph = 'white ball' if ph == 'pink ball' else ph
        if ph not in ('white ball', 'green ball', 'pink ball') or kph not in K: continue
        tr = track(r.video_id, ph, False)
        if tr is None: continue
        P, fps = pos(tr, *K[kph]); n = int(round(t * fps)); vs = []
        for k in (1, 2, 3):
            a, b = n - k, n + k
            if a >= 0 and b < len(P) and not np.isnan(P[[a, b]]).any(): vs.append(np.linalg.norm(P[b] - P[a]) / (2 * k / fps))
        new = float(np.median(vs)) if vs else np.nan
    elif md:
        a, b, t = md.group(1), md.group(2), float(md.group(3))
        ka, kb = ('white ball' if a == 'pink ball' else a), ('white ball' if b == 'pink ball' else b)
        if not ({a, b} & {'white ball', 'pink ball'}) or ka not in K or kb not in K or ka == kb: continue
        ta = track(r.video_id, a, True); tb = track(r.video_id, b, True)
        if ta is None or tb is None: continue
        Pa, fps = pos(ta, *K[ka]); Pb, _ = pos(tb, *K[kb]); n = int(round(t * fps))
        new = float(np.linalg.norm(Pa[n] - Pb[n]))
    rows.append(dict(id0=r.id0, q=q, value=new))
M = pd.DataFrame(rows)
M = M[M.value.notna()]
white = M.q.str.contains('white ball'); pink = M.q.str.contains('pink ball'); green = M.q.str.contains('green ball')
b = M.id0.map(T.video_id).eq('captured_0054b')
for name, sel in (('trk54', white), ('trk54g', ~white & ~pink & b & green), ('trk54p', ~white & pink)):
    out = M[sel]
    pd.DataFrame(dict(id0=out.id0, value=out.value.round(6))).to_csv(f'{sys.argv[1]}/{name}_test.csv', index=False)
