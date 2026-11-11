import math
import numpy as np


def _ok(x):
    try: return x is not None and np.isfinite(float(x)) and float(x) > 0
    except Exception: return False


def d3gnf_caw(r):
    if r['cat'] != 'D3' or (((r.get('parse') or {}).get('q') or {}).get('kind')) != 'size': return None
    g = r['geo'].get('meas_value'); c = r.get('s1_cs')
    if not (_ok(g) and bool(r['geo'].get('meas_valid')) and _ok(c)): return None
    m = str(r['geo'].get('meas_mode'))
    if 'gravnofall' not in m or math.log(float(g) / c) < math.log(1.5): return None
    return float(c), f"d3gnf_caw: {m} g/cs={float(g)/c:.2f} -> own CaW"


ACCEPTED = [("ag_r9_d3gnf_caw", d3gnf_caw)]
