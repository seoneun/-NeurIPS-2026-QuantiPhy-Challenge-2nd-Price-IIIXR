import sys, pandas as pd
out, base = sys.argv[1], sys.argv[2]
B = pd.read_csv(base); v = B.parsed_value.astype(float).values.copy()
for p in sys.argv[3:]:
    T = pd.read_csv(p)
    v[T.id0.astype(int).values] = T.value.astype(float).values
B['parsed_value'] = v; B.to_csv(out, index=False)
