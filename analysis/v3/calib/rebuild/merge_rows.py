#!/usr/bin/env python
import sys
import pandas as pd

base, rows, out = sys.argv[1:4]
o, T = pd.read_csv(base), pd.read_csv(rows)
assert T.id0.is_unique and T.new.dtype.kind == "f" and T.id0.between(0, len(o) - 1).all()
o.loc[T.id0.values, "parsed_value"] = T.new.values
assert o.parsed_value.notna().all() and (o.parsed_value > 0).all()
o.to_csv(out, index=False)
print('')
