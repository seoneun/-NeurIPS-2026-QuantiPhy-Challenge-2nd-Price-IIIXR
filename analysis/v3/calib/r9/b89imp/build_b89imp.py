import sys, json, pandas as pd
detail, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(detail, index_col=0)
rows = [(i, float(d.cs[i])) for i in (int(x) for x in open('analysis/v3/calib/r9/b89imp/in/gate_rows.txt').read().split())]
rows.append((2876, json.load(open('analysis/v3/calib/r9/b89imp/in/c96_O5_test.json'))['2876']['value']))
pd.DataFrame(rows, columns=['id0', 'value']).sort_values('id0').to_csv(out, index=False)
