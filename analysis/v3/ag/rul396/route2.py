import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
RT2 = f'{R}/analysis/v3/ag/rul396/frozen' if R5FZ else f'{R}/cache/tmp/rul396'
sys.path.insert(0, f'{R}/analysis/v3/ag/rul351')
import dossier as D

t = pd.read_pickle(f'{R}/analysis/v3/ag/rul396/frozen/dossier_load.pkl.gz') if R5FZ else D.load()
t['c26c'] = pd.read_csv(f'{R}/submissions/pred_t014_id1.csv').parsed_value.values
T = t.set_index('id0')


def g(i, col='g'):
    return float(T.loc[i, col])


S_FALL, F_FALL, F_DEF, K = 0.00940, 976.0, 1186.0, 0.907
D182 = 9.522
box = dict(d2_w=145 * S_FALL, d2_h=104 * S_FALL, d3def_w=176 * D182 / F_DEF, d3def_h=97 * D182 / F_DEF)

rows = []


def add(i, new, route1, r1v, route2, r2v, indep, rec=None):
    old = g(i, 'c26c')
    agree = (r2v is not None and r1v is not None and abs(r2v / r1v - 1) <= 0.15)
    big = (r1v is not None and r2v is not None and abs(r1v / old - 1) > 0.25 and abs(r2v / old - 1) > 0.25)
    two = bool(indep.startswith('yes') and agree and big)
    if rec is None:
        rec = 'adopt as two-route' if two else 'keep as one-route'
    rows.append(dict(id0=i, proposed=new, route1=route1, route2=route2, recommendation=rec))


camb = [g(2876), g(2888), g(2900)]
add(2841, 0.121179, '', 0.121179,
    '',
    float(np.median(camb)), 'yes')

add(1748, 9.42, '', 9.42,
    '', (g(1148) + g(1725)) / 2,
    'yes')

add(1170, 19.9914, '', 19.9914,
    '', g(1169),
    'yes')

add(1697, 19.9914, '', 19.9914,
    '', g(1169),
    'yes',
    rec='keep as one-route')

add(1732, 24.4708, '', g(1732),
    '', g(1730, 'c26c'),
    'yes')

add(1733, 21.8753, '', 21.8753,
    '', g(1169),
    'yes',
    rec='keep as one-route')

add(438, 1.2362, '', box['d2_w'] * K,
    '', box['d3def_w'],
    'yes',
    rec='keep as one-route')

add(1426, 1.3877, '', box['d2_w'],
    '', box['d3def_w'],
    'yes', rec='keep as one-route')

for i in (439, 1427):
    add(i, 0.8711, '',
        box['d2_h'], '', box['d3def_h'],
        'yes')

binv = [g(2783), g(2791), g(2799)]
for i in (2929, 2937, 2945):
    add(i, 43.2008, '', 43.2008,
        '', float(np.median(binv)),
        'yes')

for i, r2, txt in ((1550, g(368, 'c26c'), 't59'),
                   (1551, g(369, 'c26c'), 't60')):
    add(i, 7.2972, '', 7.2972, txt, r2,
        'yes')

c = pd.read_csv(f'{R}/analysis/v3/calib/r4/collect_rows.csv').set_index('id0')


def r4(i, r2txt, r2v, indep, rec=None):
    rr = c.loc[i]
    add(i, float(rr.new), '', float(rr.new), r2txt, r2v, indep, rec)


r4(1395, '', None, 'no')
r4(1068, '', (g(4) + g(1647)) / 2,
   'yes', rec='drop')
for i, j in ((3138, None), (3139, None)):
    r4(i, '', float(np.sqrt(g(i, 'cs') * g(i, 'qwen'))),
       'yes', rec='drop')
r4(3151, '', g(3151),
   'partly', rec='drop')
r4(3147, '', float(np.sqrt(g(3147, 'cs') * g(3147, 'qwen'))),
   'yes', rec='drop')
ex = pd.read_csv(f'{RT2}/s126puck_extents.csv').dropna(subset=['cx', 'cy'])
dg = pd.read_csv(f'{RT2}/s126_puck_gdino.csv')
dg = dg.assign(cx=(dg.x0 + dg.x1) / 2, cy=(dg.y0 + dg.y1) / 2, w=dg.x1 - dg.x0)
hand = dg[(dg.w > 40) & (dg.w < 60) & (dg.cx > 300)]
mx, my = hand.cx.median(), hand.cy.median()
dpx = ex.box_w.median()
dmin = float(np.hypot(ex.cx - mx, ex.cy - my).min())
r285 = dmin * 0.074 / dpx
top = dg[(dg.w > 40) & (dg.w < 60) & (dg.cx < 260) & (dg.frame > 0)]
r281 = float(((top.x1 - top.x0 + top.y1 - top.y0) / 2).median() * 0.074 / dpx)
r4(285, '', r285,
   'yes')
small = {601: 0.43, 588: 0.43, 626: 0.43, 1599: None, 281: r281, 345: 1.95, 1148: g(1148), 1725: g(1725),
         1475: None, 1470: None, 711: 1.112, 758: 0.702, 1966: 0.702, 706: 1.135, 1957: 1.135, 949: None,
         3271: None, 251: None, 253: None, 254: None, 1628: None, 1985: None, 1988: 85 * 0.02563, 1987: 72 * 0.02563,
         1990: 78 * 0.02563, 1991: 102 * 0.02563, 1217: None, 112: 6.019, 339: 12.64, 268: None, 1377: 1.642}
for i, v in small.items():
    rr = c.loc[i]
    r4(i, '', v, '')

out = pd.DataFrame(rows)
ad = out[out.recommendation == 'adopt as two-route']
patch = {str(int(r.id0)): {'value': float(r.proposed), 'why': f'rul396 two-route: route1 {r.route1}; route2 {r.route2}'}
         for r in ad.itertuples()}
json.dump(patch, open(f'{R}/cache/tmp/ag_rul396_patch_test.json', 'w'), indent=1)
