import json, os
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
IDS = [470, 471, 475, 476, 1917, 1918, 1922, 1923, 921, 922, 923]
p = [json.loads(l) for l in open(f'{R}/geo/out/parse_test_v2.jsonl')]
P = {r['id']: r for r in p}
cur = pd.read_csv(f'{R}/submissions/pred_t014_id1.csv').parsed_value.astype(float).values
out = {}
for i in IDS:
    r = P[i]; t, fps = r['q']['t'], r['fps']; n = round(t * fps)
    out[i] = cur[i] * (n - 0.5) / (t * fps)
for i, v in out.items():
    print(f'{i},{v!r}')
