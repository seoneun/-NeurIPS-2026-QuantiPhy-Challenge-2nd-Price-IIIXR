import os, re, sys, shutil
import numpy as np, pandas as pd

R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
SRC = f'{R}/cache/tmp/r5_tool'
INP = f'{R}/analysis/v3/ag/rul391/frozen/inputs' if R5FZ and '--snapshot' not in sys.argv else f'{R}/cache/tmp/rul391/inputs'
HERE = f'{R}/analysis/v3/ag/rul391'
TRACKS = {'g18box': 'captured_0018', 'g24book': 'captured_0024', 'g23book': 'captured_0023', 'c23sbook': 'captured_0023s',
          'c23xbook': 'captured_0023x', 'c54apink': 'captured_0054a', 'c54axpink': 'captured_0054ax'}
FW_SPEC = 0.596
W = 853
FPS = 24.0
G = 9.8


def snapshot():
    os.makedirs(INP, exist_ok=True)
    for tag, v in TRACKS.items():
        os.makedirs(f'{INP}/tracks_test_{tag}', exist_ok=True)
        shutil.copy2(f'{SRC}/tracks_test_{tag}/{v}_extents.csv', f'{INP}/tracks_test_{tag}/{v}_extents.csv')
        m = f'{SRC}/measure_test_{v}_{tag}.csv'
        if os.path.exists(m):
            shutil.copy2(m, f'{INP}/measure_test_{v}_{tag}.csv')
    os.makedirs(f'{INP}/det', exist_ok=True)
    for n in ['captured_0018_box', 'captured_0024_book', 'captured_0023_book', 'captured_0062_blue-shopping-bag',
              'internet_0006_white-car', 'internet_0006_motorcycle', 'internet_0073_yellow-bag', 'internet_0073_blue-bag']:
        shutil.copy2(f'{SRC}/det/test_{n}_owlv2.csv', f'{INP}/det/test_{n}_owlv2.csv')


def ext(tag):
    return pd.read_csv(f'{INP}/tracks_test_{tag}/{TRACKS[tag]}_extents.csv')


def meas(tag):
    return pd.read_csv(f'{INP}/measure_test_{TRACKS[tag]}_{tag}.csv').set_index('id0')


def depths(depth_text):
    out = {}
    for name, val in re.findall(r'distance_(\w+?)_camera\s*=\s*([\d.]+)\s*m', str(depth_text)):
        out.setdefault(name, float(val))
    return out


def main():
    if '--snapshot' in sys.argv:
        snapshot()
    free = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
    lab = pd.read_csv(f'{T5}/labelled_rows.csv').set_index('id0')
    f = FW_SPEC * W
    rows = {}

    e = ext('g18box'); dep = depths(free.loc[2295, 'depth'])
    d_held, d_floor = dep['soccer_ball'], dep['far_person']
    held = e[(e.frame <= 28)]; floor = e[(e.frame >= 42)]
    L_held = held.rr_long.median() * d_held / f * 100
    L_floor = floor.rr_long.median() * d_floor / f * 100
    length = float(np.sqrt(L_held * L_floor))
    width = float(held.rr_short.median() * np.sqrt(d_held * d_floor) / f * 100)
    for i in (2295, 2314, 2333): rows[i] = length
    for i in (2296, 2315, 2334): rows[i] = width

    a = float(re.search(r'acceleration\s*=\s*([\d.]+)', str(free.loc[2407, 'prior'])).group(1)) * 100
    for tag, r12, r14 in (('g23book', 2406, 2407), ('c23sbook', 2415, 2416), ('c23xbook', 2424, 2425)):
        x = ext(tag); st = np.hypot(np.diff(x.cx.values), np.diff(x.cy.values)); fr = x.frame.values[1:]
        k_last = max(k for k, s in zip(fr, st) if s >= 1.0 and k <= 45)
        t_stop = (k_last + 0.5) / FPS
        v12 = float(meas(tag).loc[r12, 'newM'])
        route_a = v12 - a * 0.2
        route_b = a * (t_stop - 1.4)
        rows[r14] = float((route_a + route_b) / 2)

    for tag, rid in (('c54apink', 2839), ('c54axpink', 2862)):
        x = ext(tag); s = x[(x.frame >= 28) & (x.frame <= 35)]; t = s.frame.values.astype(float)
        px = np.polyfit(t, s.cx.values, 1); py = np.polyfit(t, s.cy.values, 2)
        ay = 2 * py[0]; tc = 37.5
        v = np.hypot(px[0], 2 * py[0] * tc + py[1])
        rows[rid] = float(v * G / (ay * FPS))

    rat = []
    for tag, rid in (('g23book', 2409), ('c23sbook', 2418), ('c23xbook', 2427)):
        rat.append(float(lab.loc[rid, 'y']) / float(meas(tag).loc[rid, 'newM']))
    w24 = float(meas('g24book').loc[2436, 'newM'])
    for i in (2436, 2446, 2456): rows[i] = w24 * float(np.mean(rat))

    for k in sorted(rows):
        print(f'{k},{rows[k]:.6g}')
    return (rows,)




if __name__ == '__main__':
    main()
