import sys, pandas as pd
detail, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(detail, index_col=0)
i = [int(x) for x in open('analysis/v3/ag/r9_main/levers/shrink_nolab/in/rows.txt').read().split()]
pd.DataFrame({'id0': i, 'value': d.cs.loc[i].values}).to_csv(out, index=False)
