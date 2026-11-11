import os, sys, pandas as pd
base, out = sys.argv[1:3]
W = 'output/work/p2prep'; os.makedirs(W, exist_ok=True)
b = pd.read_csv(base); b['id0'] = b.id - 1
cur = dict(zip(b.id0, b.parsed_value)); cat = dict(zip(b.id0, b.inference_type.str[0] + b.video_type.str[1]))
k = pd.read_csv('analysis/v3/ag/r9_main/levers/p2prep/in/kinc_rows.csv', float_precision='round_trip')
k['cur'] = k.id0.map(cur); k = k[((1 - k.cur / k.a_t) * k.tk * k.fps - 0.5).abs() < 0.05]
pd.DataFrame({'id0': k.id0, 'value': k.a_t * (1 - 0.2 / (k.tk * k.fps))}).to_csv(f'{W}/kinc_c05.csv', index=False)
k = pd.read_csv(f'{W}/kinc_c05.csv')
k[k.id0.map(cat) == 'D2'].to_csv(out, index=False)
