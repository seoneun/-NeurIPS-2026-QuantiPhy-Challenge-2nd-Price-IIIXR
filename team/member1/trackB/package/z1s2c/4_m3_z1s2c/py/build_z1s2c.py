import sys, numpy as np, pandas as pd
S = sys.argv[1].rstrip("/")
rd = lambda p: pd.read_csv(p, float_precision="round_trip")
f2, fm = f"{S}/pred_t146_id1.csv", sys.argv[2]
Z = rd(f2); z = Z.set_index("id").parsed_value; m3 = rd(fm).set_index("id").parsed_value
REV = [249, 252, 891, 892, 893, 1837, 3104, 3108, 3109, 3202]
new = z.copy(); new[REV] = m3[REV].values
out = f"{S}/pred_t004_id1.csv"
O = Z.copy(); O["parsed_value"] = new.loc[O.id].values; O.to_csv(out, index=False, lineterminator="\r\n")
