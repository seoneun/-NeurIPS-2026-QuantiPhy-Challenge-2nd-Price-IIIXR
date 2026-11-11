import math, re, importlib.util
import os
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "G_gkwide", "rules.py")
_s = importlib.util.spec_from_file_location("gkw_table_m_s2", _P); GK = importlib.util.module_from_spec(_s); _s.loader.exec_module(GK)
def _f(x):
    try:
        x = float(x); return x if math.isfinite(x) and x > 0 else None
    except Exception: return None
def _u3(row, var):
    if row['cat'] != 'S2' or str(row['video_source']).startswith('simulation'): return None
    P = row.get('parse') or {}; Q = P.get('q') or {}
    if Q.get('kind') != 'size': return None
    g, c, cur = _f(row['s1_g']), _f(row['s1_cs']), _f(row['cur'])
    if not (g and c and cur) or abs(math.log(cur / c)) > 0.02 or abs(math.log(g / c)) <= math.log(2): return None
    geo = row['geo']
    if str(geo.get('meas_mode')) != '2d_ratio' or not bool(geo.get('meas_valid')): return None
    hit = GK.lookup(row)
    if hit is None:
        if var != 'V2': return None
        q = str(row.get('question', '')).lower(); _, _, ph = GK._head(Q.get('target') or (P.get('target') or {}).get('phrase'))
        if re.search(GK.GEXC, re.sub(r'toilet paper', '', ph + ' ' + q)): return None
        why = 'no class'
    else:
        nm, lo, hi, src = hit
        if not (lo <= g <= hi): return None
        if var == 'V3' and not (lo <= c <= hi): return None
        why = f"{nm} [{lo:.4g},{hi:.4g}]"
    return g, f"s2dis_gkgeo[{var}]: S2 planar size, geo {g:.4g} vs CaW {c:.4g} > 2x, geo class-plausible ({why}) -> own geo"
def U3_V1(row): return _u3(row, 'V1')
def U3_V2(row): return _u3(row, 'V2')
def U3_V3(row): return _u3(row, 'V3')
