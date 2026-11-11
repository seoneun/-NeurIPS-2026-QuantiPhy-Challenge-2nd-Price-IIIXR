import os, sys, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1] if len(sys.argv) > 1 else 'pred_t107'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
MAST = re.compile(r'\b(sail\w*|yacht|schooner|sloop|ketch)\b', re.I)
TDIRS = {'v1': ['', '_v1'], 'v2': ['_v2sc', '_v2'], 'v3': ['_v3nt', '_v3'], 'v3nt': ['_v3nt']}
SPEEDK = ('avgspeed', 'speed', 'instspeed', 'maxspeed', 'velocity')


def track_states(j, phrase):
    qs = j.get('queries')
    if isinstance(qs, list):
        for q in qs:
            if q.get('phrase') == phrase or q.get('key', '').split('|')[0] == phrase:
                if 'obj' in q and q['obj'] is not None:
                    L = j.get('objs', {}).get(str(q['obj'])) or []
                    return [x['st'] for x in L if x and x.get('st')]
                return [x['st'] for x in q.get('frames', []) if x and x.get('st')]
        return None
    if isinstance(qs, dict):
        for key, o in qs.items():
            if key.split('|')[0] == phrase:
                L = j.get('objs', {}).get(str(o)) or j.get('objs', {}).get(o) or []
                return [x['st'] for x in L if x and x.get('st')]
    return None


def fuse(kind, g, cs):
    if not (pd.notna(cs) and cs > 0):
        return g, 'geo'
    r = max(g / cs, cs / g)
    if kind == 'size':
        if r <= 2:
            return float(np.exp(0.7 * np.log(g) + 0.3 * np.log(cs))), 'agree_wgm'
        return g, 'dis_geo'
    if r > 2:
        return g, 'dis_geo'
    return float(np.sqrt(g * cs)), 'agree_gm'




