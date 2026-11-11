import sys, io, json
import pandas as pd
rows = pd.read_csv(sys.argv[1], sep="\t")
P = {}
vals = []
for i, t in zip(rows.id0, rows.patch):
    if t not in P:
        P[t] = json.load(open(f"{sys.argv[2]}/ag_{t}_patch_test.json"))
    e = P[t].get(str(int(i)))
    if e is None:
        sys.exit('')
    vals.append(repr(float(e["value"])))
v = pd.read_csv(io.StringIO("value\n" + "\n".join(vals) + "\n")).value
with open(sys.argv[3], "w") as f:
    f.write("id0,value\n")
    for i, x in zip(rows.id0, v):
        f.write(f"{int(i)},{float(x)!r}\n")
print('')
