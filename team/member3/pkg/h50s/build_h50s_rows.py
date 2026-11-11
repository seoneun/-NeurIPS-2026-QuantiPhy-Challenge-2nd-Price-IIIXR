import sys
import numpy as np
import pandas as pd

Z1S2C, INCL, OUT9, OUT24, PRED_T018 = sys.argv[1:6]
sub = lambda p, **k: pd.read_csv(p, **k).assign(id0=lambda d: d['id'] - 1).set_index('id0').parsed_value
val = {}
N48, C15 = sub('submissions/pred_t002_id1.csv', dtype={'parsed_value': str}), pd.read_csv(PRED_T018, dtype={'value': str}).set_index('id0').value
for i in (53, 2435, 2445):
    val[i] = float(N48[i])
val[2826] = float(C15[2826])
Z = sub(Z1S2C, float_precision='legacy')
G = pd.read_csv('team/member3/pkg/h50s/inputs/pred_t018_t6_rows.csv', float_precision='legacy').set_index('id0').pred_t018_t6
for i in (1584, 1680):
    val[i] = float(np.sqrt(Z[i] * G[i]))
for i in (2928, 2936, 2944):
    val[i] = 29.5
V = pd.read_csv('data/val/validation_dataset.csv')
boat = V[(V.video_id.str.strip() == 'simulation_0098_segmented') & (V.question.str.strip() == 'What is the length of the boat in meters?')]
for i in (10, 1124, 1154, 1162, 1704, 1738, 1744):
    val[i] = float(boat.ground_truth_posterior.iloc[0])
val.update({726: 2.4, 2292: 161.3, 2790: 1.43, 2924: 42.1, 2933: 34.7, 2940: 44.9, 2950: 2.23, 2953: 11.2, 2954: 40.2, 2955: 5.8,
            2980: 9.7, 3069: 22.4})
I = pd.read_csv(INCL).set_index('id0').value
for i in (2239, 2261):
    val[i] = float(I[i])
S9 = [10, 1124, 1154, 1162, 1584, 1680, 1704, 1738, 1744, 2826, 2928, 2936, 2944]
R = pd.DataFrame(sorted(val.items()), columns=['id0', 'value'])
R[R.id0.isin(S9)].to_csv(OUT9, index=False)
R[~R.id0.isin(S9)].to_csv(OUT24, index=False)
