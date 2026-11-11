import os, re, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
WING_RX = re.compile(r"wing ?span|width of (the|its) wings", re.I)
MIN_FR, MIN_FRAC, MIN_PX, R_MAX = 12, 0.5, 12.0, 1.5

def is_wingspan(r):
    return bool(WING_RX.search(r['question']))

def _zfit(r):
    p = r.get('parse') or {}; keys = set(p.get('target_dkeys') or [])
    pts = [(float(d['t']), float(d['d'])) for d in (p.get('depth') or []) if d.get('key') in keys and d.get('t') is not None and d.get('d')]
    if len({t for t, _ in pts}) < 2: return None
    t, z = np.array(pts).T; return np.polyfit(t, z, 1)

def ratio(r, pct=90):
    cp = f"{HERE}/cache/w_{r['split']}_{r['id']}.json"
    if not os.path.exists(cp): return None
    o = json.load(open(cp)); n = o.get('n', 0)
    fr = [f for f in o.get('frames', []) if not f['border'] and not f['spill']]
    if len(fr) < MIN_FR or len(fr) < MIN_FRAC * max(n, 1): return None
    L = np.array([f['L'] for f in fr])
    if np.median(L) < MIN_PX: return None
    X = L
    if r['video_type'][1] == '3':
        c = _zfit(r)
        if c is None: return None
        fps = float(r['fps']) if r.get('fps') else 24.0
        Z = np.polyval(c, np.array([f['k'] for f in fr]) / fps)
        if (Z <= 0).any(): return None
        X = L * Z
    q = float(np.percentile(X, pct) / np.percentile(X, 50))
    return q if 1.0 <= q <= R_MAX else None

def _scope(r):
    if not is_wingspan(r) or 'wing' in str(r.get('s1_src', '')): return None
    cur = r['cur']
    gv = [float(g.get('geo_value')) for g in (r.get('geo') or {}, r.get('geo_u') or {})
          if g.get('meas_kind') == 'size' and bool(g.get('meas_valid')) and g.get('geo_value') is not None and np.isfinite(float(g.get('geo_value')))]
    if any(abs(math.log(cur / g)) <= 0.01 for g in gv if g > 0): return 'direct'
    s = str(r.get('s1_src', ''))
    if (s == 'agree_gm' or s.endswith('_gm')) and any(0.5 <= g / cur <= 2.0 for g in gv if g > 0): return 'gm'
    return None

def make(pct, allow_gm):
    def fn(r):
        sc = _scope(r)
        if sc is None or (sc == 'gm' and not allow_gm): return None
        q = ratio(r, pct)
        if q is None: return None
        f = q if sc == 'direct' else math.sqrt(q)
        return r['cur'] * f, f"wingspan flap P{pct}/P50={q:.3f} own video ({sc})"
    return fn

ws_p90 = make(90, True)
ws_p95 = make(95, True)
ws_p90_dir = make(90, False)
VARIANTS = [("ag_r9_R1_ws_p90", ws_p90), ("ag_r9_R2_ws_p95", ws_p95), ("ag_r9_R3_ws_p90_dir", ws_p90_dir)]
ACCEPTED = [("ag_r9_R1_ws_p90", ws_p90)]
