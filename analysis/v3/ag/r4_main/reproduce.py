import os
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv').set_index('id0')
out = []
for i in (656, 672, 655, 671):
    g, cs = float(t.loc[i, 'g']), float(t.loc[i, 'cs'])
    assert max(g, cs) / min(g, cs) < 2
    out.append((i, g))
pd.DataFrame(out, columns=['id0', 'value']).to_csv(os.path.join(os.path.dirname(__file__), 'reproduce_out.csv'),
                                                   index=False, float_format='%.17g')
print(out)
