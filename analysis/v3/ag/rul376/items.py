import os
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
ITEMS = {
    1: [200, 1805], 2: [1060, 1061, 1062, 4, 1647, 5, 1648, 6, 1649], 3: [1960, 1961, 1962, 1963, 1964],
    4: [759, 1967], 5: [807, 815, 838, 845, 827, 831],
    6: [777, 778, 782, 783, 786, 787, 794, 795, 800, 801, 810, 811, 1621, 1969, 1970, 1972, 1973],
    7: [887, 888, 894, 895, 902, 903, 912, 913, 917, 918, 919, 2006, 2007, 2008], 8: [465], 9: [3145],
    10: [3084, 3085, 3086, 3088], 11: [3283, 3284, 3285, 3286, 3288], 12: [2103, 2977, 2105, 2979],
    13: [2039, 2040, 2041, 2042], 14: [3050, 3051, 3052], 15: [2835, 2847, 2858, 2870, 2882, 2894],
    16: [2433, 2434, 2443, 2444, 2453, 2454], 17: [2154, 2159], 18: [2269, 2274, 2279], 19: [2747, 2755, 2763],
    20: [2517, 2518], 21: [2496, 2501, 2506], 22: [2239, 2242, 2250, 2253, 2261, 2264], 23: [206, 1811], 24: [1384],
    25: [401, 403, 405, 407, 1858, 1860, 1862, 1864], 26: [1052, 1053, 1054, 1056, 1642, 1643, 1644, 1646], 27: [8],
    28: [1025, 1027, 1029, 1030]}
N = {'S2': 532, 'D2': 1109, 'S3': 488, 'D3': 881}


def load():
    F = pd.read_csv(f'{T5}/free_rows.csv')
    L = pd.read_csv(f'{T5}/labelled_rows.csv')
    rows = []
    for it, ids in ITEMS.items():
        for i in ids:
            rows.append({'item': it, 'id0': i})
    T = pd.DataFrame(rows)
    T['held'] = T.id0.isin(L[L.set == 'held'].id0)
    T = T.merge(F, on='id0', how='left')
    return T


if __name__ == '__main__':
    T = load()
    os.makedirs(f'{R}/cache/tmp/rul376', exist_ok=True)
    T.to_csv(f'{R}/cache/tmp/rul376/items_rows.csv', index=False)
    print('rows', len(T), 'unique', T.id0.nunique(), 'held', int(T.held.sum()), 'not in free', int(T.c26c.isna().sum()))
    print(T.groupby('cat').size().to_dict())
    pd.set_option('display.width', 250)
    T['q'] = T.question.str[:52]
    T['w'] = T.writers.fillna('').str[:28]
    print(T[['item', 'id0', 'video', 'cat', 'q', 'c26c', 'c25', 'g', 'cs', 'w', 'r4', 'source']].to_string(
        index=False, float_format=lambda x: f'{x:.4g}'))
