import csv, io, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '../../../..'))
t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
b = pd.read_csv(f'{R}/submissions/pred_t007_id1.csv')
assert len(t) == len(b)
t['cur'] = b.parsed_value.values
T = t.set_index('id0')


def sib(video, qsub):
    g = t[(t.video == video) & t.question.str.contains(qsub, regex=False)]
    assert len(g) == 1
    return float(g.cur.iloc[0]), int(g.id0.iloc[0])


def target(i, video, qsub):
    assert T.loc[i, 'video'] == video and qsub in T.loc[i, 'question']


def csv_roundtrip(x):
    return float(pd.read_csv(io.StringIO('v\n' + repr(float(x)) + '\n')).v.iloc[0])


out = {}
target(1560, 'simulation_0202', 'height of the typewriter')
v207, _ = sib('simulation_0207', 'height of the typewriter')
v208, _ = sib('simulation_0208', 'height of the typewriter')
out[1560] = float(np.median([v207, v208]))
target(1584, 'simulation_0209', 'length of the typewriter')
out[1584] = sib('simulation_0206', 'length of the typewriter')[0]
target(1597, 'simulation_0211', 'diameter of the lampshade')
out[1597] = sib('simulation_0212', 'diameter of the lampshade')[0]
v70, _ = sib('simulation_0070', 'width of the turning yellow car')
v71, _ = sib('simulation_0071', 'width of the turning yellow car')
yw = float(np.mean([v70, v71]))
for i, v in ((1285, 'simulation_0062'), (1353, 'simulation_0067')):
    target(i, v, 'width of the turning yellow car')
    out[i] = yw
out = {i: csv_roundtrip(v) for i, v in out.items()}

with open(f'{HERE}/reproduce_out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id0', 'value'])
    for i, v in out.items():
        w.writerow([int(i), repr(float(v))])
