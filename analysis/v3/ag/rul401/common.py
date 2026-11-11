import re, numpy as np, pandas as pd
R = '@QP_ROOT@'
O = f'{R}/cache/tmp/rul401'
FAMS = ['0022', '0028', '0029', '0030', '0031', '0032', '0033', '0036', '0037']

def free_rows():
    f = pd.read_csv(f'{R}/cache/tmp/r5/free_rows.csv')
    return f[f.video.str.extract(r'captured_(\d{4})')[0].isin(FAMS)].copy()

def load_b(name):
    rows, foc = [], None
    for line in open(f'{R}/cache/tmp/r4_labs3/{name}.out'):
        if line.startswith('focal per entry'):
            foc = [float(x) for x in re.findall(r'[\d.]+', line.split(']')[0])]
        m = re.match(r'^(\d\.\d{3})\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d)\s+([-\d.]+)\s+([-\d.]+)', line)
        if m:
            rows.append([float(x) for x in m.groups()])
    return pd.DataFrame(rows, columns=['t', 'u', 'v', 'dpx', 'bd', 'Dlin', 'Dsz']), foc

def prod_geo(id0):
    p = pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
    return float(p.loc[p['Unnamed: 0'] == id0, 'geo_value'].iloc[0])

def row_F(id0):
    t = pd.read_pickle(f'{R}/analysis/v3/ag/rul346/tab.pkl')
    return float(t[(t.split == 'test') & (t.id0 == id0)].F.iloc[0])

def ray_point(u, v, f, D, cx=427.0, cy=240.0):
    r = np.array([u - cx, v - cy, f], float)
    return D * r / np.linalg.norm(r)
