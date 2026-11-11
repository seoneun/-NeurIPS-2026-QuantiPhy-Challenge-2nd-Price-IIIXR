import re, math
NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")
def _f(x):
    try:
        x = float(x); return x if math.isfinite(x) and x > 0 else None
    except Exception: return None
def _u1(row, kin, need_u):
    if row['cat'] != 'S2' or row['s1_src'] not in ('agree_gm', 'agree_wgm'): return None
    g, c, cur = _f(row['s1_g']), _f(row['s1_cs']), _f(row['cur'])
    if not (g and c and cur) or not (0.5 <= g / c <= 2): return None
    geo = row['geo']
    if str(geo.get('meas_mode')) != '2d_ratio' or not bool(geo.get('meas_valid')): return None
    if not (min(g, c) * 0.999 <= cur <= max(g, c) * 1.001) or abs(math.log(cur / g)) < 1e-9: return None
    q = ((row.get('parse') or {}).get('q') or {}); k = q.get('kind')
    if k == 'size': pass
    elif k in kin and not NONRIGID.search(str(row['question']).lower()): pass
    else: return None
    if need_u:
        gu = _f(row['geo_u'].get('meas_value')); gm_ = _f(geo.get('meas_value'))
        if not (gu and gm_) or abs(math.log(gu / gm_)) > math.log(1.05): return None
    return g, f"s2plan_geo: S2 planar 2d_ratio agree row ({row['s1_src']}, kind {k}) geo {g:.4g} vs CaW {c:.4g} -> own geo (was {cur:.4g})"
def U1_V1(row): return _u1(row, ('speed', 'accel', 'disp'), False)
def U1_V2(row): return _u1(row, ('speed', 'disp'), False)
def U1_V3(row): return _u1(row, ('speed', 'disp'), True)
def _f(x):
    try:
        x = float(x); return x if math.isfinite(x) and x > 0 else None
    except Exception: return None
def _u2(row, var):
    if row['cat'] != 'S2' or 'rul202' not in str(row['s1_src']): return None
    p = row.get('parse') or {}; k = (p.get('q') or {}).get('kind')
    if k not in ('speed', 'avgspeed', 'accel', 'disp'): return None
    g, cur = _f(row['s1_g']), _f(row['cur'])
    if not (g and cur) or abs(math.log(cur / g)) < 1e-9: return None
    if var in ('V1', 'V3'):
        if str(row['geo'].get('merge_src')) != 'sruler': return None
    else:
        po = str(((p.get('prior_obj') or {}).get('phrase')) or '').strip().lower(); tg = str(((p.get('target') or {}).get('phrase')) or '').strip().lower()
        if not po or not tg or po == tg: return None
    v = math.sqrt(cur * g) if var == 'V3' else g
    return v, f"s2edge_sruler[{var}]: {k} scaled by a scene/other ruler; target border-extent ratio does not enter speed -> own geo {g:.4g} (was {cur:.4g})"
def U2_V1(row): return _u2(row, 'V1')
def U2_V2(row): return _u2(row, 'V2')
def U2_V3(row): return _u2(row, 'V3')

ACCEPTED = [('ag_r9_S2_U1_s2plan_geo', U1_V2), ('ag_r9_S2_U2_s2edge_sruler', U2_V1)]
