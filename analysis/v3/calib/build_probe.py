#!/usr/bin/env python
import os, sys, json
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
S = os.environ.get("QP_SUB", f"{R}/submissions"); ROWS = os.environ.get("QP_ROWS", f"{R}/analysis/v3/calib")


def path(n):
    return n if n.endswith(".csv") else f"{S}/{n}_id1.csv"


def load(p):
    return pd.read_csv(p, dtype={"parsed_value": str}, keep_default_na=False)



spec = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
bad = [l for l in spec if l[0] not in ("name", "base", "block", "io") or len(l) != (4 if l[0] == "block" else 2)]
if bad:
    sys.exit('')
kv = {l[0]: l[1] for l in spec if l[0] in ("name", "base", "io")}
name, base_p = kv["name"], path(kv["base"])
base = load(base_p)
bv = base.parsed_value.astype(float).values
cat = (base.inference_type.str[0] + base.video_type.str[1]).values
new, rows = {}, []
for l in spec:
    if l[0] != "block":
        continue
    blk, cats, src = l[1], l[2], l[3]
    kind, _, arg = src.partition(":")
    if kind == "json":
        vals = {int(k): repr(float(v["value"] if isinstance(v, dict) else v)) for k, v in json.load(open(f"{R}/{arg}")).items()}
        writer = os.path.basename(arg)
    elif kind == "rows":
        p, _, sub = arg.partition("#")
        T = pd.read_csv(f"{R}/{p}", dtype={"new": str})
        if sub:
            T = T[T.block.isin([sub, sub + "_copy"])]
        vals = dict(zip(T.id0.astype(int), T.new)); writer = os.path.basename(p) + (f"#{sub}" if sub else "")
    elif kind == "file":
        f = load(path(arg) if not arg.startswith("/") else arg)
        d = np.where(~np.isclose(f.parsed_value.astype(float).values, bv, rtol=1e-12))[0]
        vals = {int(i): f.parsed_value[i] for i in d}; writer = os.path.basename(arg)
    else:
        sys.exit('')
    for i, s in sorted(vals.items()):
        if cats != "*" and cat[i] not in cats.split("+"):
            continue
        assert i not in new
        if np.isclose(float(s), bv[i], rtol=1e-12):
            continue
        new[i] = s; rows.append((i, cat[i], blk, writer))
fio = kv.get("io") == "float"
o = pd.read_csv(base_p) if fio else base.copy()
for i, s in new.items():
    o.loc[i, "parsed_value"] = float(s) if fio else s
assert (o.parsed_value.astype(float) > 0).all()
T = pd.DataFrame(rows, columns=["id0", "cat", "block", "writer"]).sort_values("id0")
T["old"] = [base.parsed_value[i] for i in T.id0]; T["new"] = [new[i] for i in T.id0]
T["video"] = [base.video_id[i] for i in T.id0]; T["question"] = [base.question[i] for i in T.id0]
print(T.groupby(["cat", "block"]).size().to_string())
out = path(name)
o.to_csv(out, index=False); T.to_csv(f"{ROWS}/{name}_rows.csv", index=False)
