import os, re, numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def dist(text, obj):
    out = {}
    for m in re.finditer(r't=([\d.]+)s,\s*distance_' + obj + r'_camera\s*=\s*([\d.]+)', str(text)):
        out[float(m.group(1))] = float(m.group(2))
    return out


def values():
    t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv').set_index('id0')
    out = {}
    r = t.loc[1550]
    assert 'shark' in r.question and 'acceleration of the turtle = 2.639' in r.prior
    a_turtle = float(re.search(r'=\s*([\d.]+)', r.prior).group(1))
    ds, dt = dist(r.depth, 'shark'), dist(r.depth, 'turtle')
    a_shark = a_turtle * (ds[0.0] - ds[0.5]) / (dt[0.0] - dt[0.5])
    out[1550] = a_shark
    out[1551] = a_shark * 1.0
    g38l, g38w, g37l = (float(t.loc[i, 'g']) for i in (1732, 1733, 1730))
    assert 'length of the rock' in t.loc[1732, 'question'] and 'width of the rock' in t.loc[1733, 'question']
    assert 'length of the rock' in t.loc[1730, 'question'] and t.loc[1730, 'video'] == 'simulation_0037_segmented'
    L = float(np.sqrt(g38l * g37l))
    out[1732] = L
    out[1733] = g38w * L / g38l
    return out


if __name__ == '__main__':
    for k, v in values().items():
        print(k, round(v, 4))
