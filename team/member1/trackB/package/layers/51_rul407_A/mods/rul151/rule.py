import os, sys, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1] if len(sys.argv) > 1 else 'pred_t107'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
TDIR = {'v1': '', 'v2': '_v2', 'v3': '_v3'}


def load_tracks(split, ver, v):
    suf = TDIR.get(ver, '_' + str(ver))
    p = f'{R}/geo/out/tracks_{split}{suf}/{v}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def track_states(j, phrase):
    qs = j.get('queries')
    if isinstance(qs, list):
        for q in qs:
            if q.get('phrase') == phrase or q.get('key', '').split('|')[0] == phrase:
                return [x['st'] for x in q.get('frames', []) if x and x.get('st')]
        return None
    if isinstance(qs, dict):
        for key, o in qs.items():
            if key.split('|')[0] == phrase:
                L = j.get('objs', {}).get(str(o)) or j.get('objs', {}).get(o) or []
                return [x['st'] for x in L if x and x.get('st')]
    return None
