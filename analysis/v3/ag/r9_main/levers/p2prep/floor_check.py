import sys, glob, json, re, numpy as np, pandas as pd
FW = 0.596; D = 'geo/out/tracks_test_qv3ntmv_mq2off'
base, out_t = sys.argv[1:3]
T = pd.read_csv(base); T['id0'] = T.id - 1; T['cat'] = T.inference_type.str[0] + T.video_type.str[1]
H = pd.read_csv('team/member1/trackB/build/inputs/h1_rows.csv').set_index('id0').status
T['status'] = T.id0.map(H)
S = T[(T.video_source == 'lab') & T.cat.isin(['D3', 'S3']) & T.status.notna() & (T.status != 'X_LAB_VAL')]
pat = re.compile(r"(?:velocity|speed) of (?:the )?([a-z\- ]+?) (?:at|when) ([\d.]+) ?s|(?:the )?([a-z\- ]+?)'s (?:velocity|speed) at ([\d.]+) ?s")
res = []
for _, r in S.iterrows():
    m = pat.search(r.question.lower())
    if not m: continue
    obj = (m.group(1) or m.group(3)).strip(); t = float(m.group(2) or m.group(4))
    km = re.search(r't=([\d.]+)s, distance_' + re.escape(obj.replace(' ', '_')) + r'_camera = ([\d.]+) ?m', str(r.depth_info))
    if not km: continue
    tk, Zk = float(km.group(1)), float(km.group(2)); Ls, Vs = [], []
    for f in sorted(glob.glob(f'{D}/{r.video_id}__*.json')):
        j = json.load(open(f)); q = [(p['phrase'], p['obj']) for p in j['queries'] if p['obj'] is not None]
        ob = [o for p, o in q if p == obj]
        if not ob or sum(o == ob[0] for _, o in q) > 1: continue
        o = j['objs'][str(ob[0])]; fps = j['fps']; W, Hh = j['W'], j['H']; fp = FW * W
        c = np.full((len(o), 2), np.nan); d = np.full(len(o), np.nan)
        for i, e in enumerate(o):
            st = e.get('st') if isinstance(e, dict) else None
            if st and st.get('area', 0) > 0: c[i] = st['c']; d[i] = np.sqrt(4 * st['area'] / np.pi)
        nk = int(round(tk * fps)); n = int(round(t * fps))
        if n + 3 >= len(d) or n - 3 < 0: continue
        dk = np.nanmedian(d[max(0, nk - 1):nk + 2]); Z = Zk * dk / d; X = (c[:, 0] - W / 2) * Z / fp; Y = (c[:, 1] - Hh / 2) * Z / fp
        L = np.hypot(*(c[n + 1] - c[n - 1])) / (2 / fps) * Z[n] / fp
        V = [np.sqrt((X[n + k] - X[n - k]) ** 2 + (Y[n + k] - Y[n - k]) ** 2 + (Z[n + k] - Z[n - k]) ** 2) / (2 * k / fps) for k in (1, 2, 3)]
        if np.isfinite(L) and np.isfinite(V).all(): Ls.append(L); Vs.append(float(np.median(V)))
    if not Ls: continue
    unit = 100.0 if 'cm/s' in r.question else 1.0
    res.append(dict(id0=r.id0, cur=r.parsed_value, L=np.median(Ls) * unit, V=np.median(Vs) * unit))
O = pd.DataFrame(res); F = O[O.cur < 0.7 * O.L]
pd.DataFrame({'id0': F.id0, 'value': F.V}).to_csv(out_t, index=False)
