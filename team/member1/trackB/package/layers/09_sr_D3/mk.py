import pandas as pd
import os; O = os.environ['OUTDIR'] + '/'
f = pd.read_csv(O + 'fired_test.csv')
ad = f[~f.tag.str.contains('lodkcw')]
ad[['id0', 'new', 'tag']].rename(columns={'new': 'value', 'tag': 'rule'}).to_csv(O + 'cand.csv', index=False)
