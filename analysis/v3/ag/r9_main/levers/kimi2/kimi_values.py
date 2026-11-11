import json, re, sys, pandas as pd
rows, src, out = sys.argv[1:4]
NUM = r'([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)'
o = []
for k in open(rows).read().split():
    c = json.load(open(f'{src}/{k}.json'))['response']['choices'][0]['message']['content']
    o.append((int(k[1:]), float(re.findall(r'ANSWER\s*[:=]\s*\**\s*[~≈]?\s*' + NUM, c, re.I)[-1].replace(',', ''))))
pd.DataFrame(o, columns=['id0', 'value']).to_csv(out, index=False)
