import os
import pandas as pd
d=pd.read_csv(os.path.join(os.environ.get("QP_ROOT", "."), "analysis/v3/ag/rul143/wt/geo/out/kin2d_test_testS2.csv"))
pd.set_option('display.width',250)
print(d[[c for c in ["id","video","kind","value","mpp","how","scale_why","why","prior_dense","prior_range_px"] if c in d]].to_string(index=False,max_colwidth=40))
