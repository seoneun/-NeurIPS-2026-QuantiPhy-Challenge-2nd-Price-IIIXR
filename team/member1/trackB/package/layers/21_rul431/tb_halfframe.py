import csv, re, sys
import pandas as pd
base, OUT = sys.argv[1:3]
d = pd.read_parquet('data/test/test_dataset.parquet')
H6 = {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(base, newline='', encoding='utf-8'))}
rows = []
for i, r in d.reset_index(drop=True).iterrows():
    q = re.search(r'speed of the ([a-z ]+?) at ([\d.]+)\s*s in m/s', r.question.lower())
    p = re.search(r'(?:acceleration of the [a-z ]+?(?: at [\d.]+\s*s)?|gravity acc)\s*=\s*([\d.]+)\s*m/s\^?2', r.prior.lower())
    if not q or not p: continue
    t = float(q.group(2)); a = float(p.group(1)); fps = int(float(r.fps)); n = round(t * fps)
    was = H6[i]
    if n > 12 or abs(float(was) - a * t) > 1e-9: continue
    rows.append([i, a * (n - 0.5) / fps, f'tb_halfframe: a(round(t fps)-0.5)/fps, a={a} t={t} fps={fps}; was a*t={a * t:g}'])
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); w.writerows(rows)
print(len(rows), 'rows ->', OUT)
