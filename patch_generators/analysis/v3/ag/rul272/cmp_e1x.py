"""check that every e1x<x> patch equals the abplan ab_<x> patch (same rows, same values)"""
import json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
bad = 0
for t in ('s2rigidmo_sim', 's2rigidmo_inet', 'd2gselfgm', 's3firstkey', 'd3tdepth', 'd3kint0'):
    for sp in ('test', 'val'):
        a = json.load(open(f'{R}/cache/tmp/ag_ab_{t}_patch_{sp}.json')); e = json.load(open(f'{R}/cache/tmp/ag_e1x{t}_patch_{sp}.json'))
        same = set(a) == set(e) and all(abs(a[k]['value'] - e[k]['value']) == 0 for k in a)
        bad += not same
        print(f'e1x{t} {sp}: {len(e)} rows, ab_{t} {len(a)} rows, identical values: {same}')
print('ALL_IDENTICAL' if bad == 0 else f'MISMATCH {bad}')
