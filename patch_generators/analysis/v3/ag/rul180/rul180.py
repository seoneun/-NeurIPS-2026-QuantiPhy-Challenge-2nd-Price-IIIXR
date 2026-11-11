"""rul180: c86d3simfd (sim D3 focal/scale fix) restricted by the verifier's code filter:
keep a row only when geo is well above CaW (g / cs >= 1.5), i.e. drop agreeing rows and geo-below-CaW rows
(labelled analogs of those favour the gm hedge). Reads the c86d3simfd patch and the build detail; no labels."""
import json, os, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C86F_SUB", "sub_xt_Nzz29")
for split, det in (("test", "detail"), ("val", "valdetail")):
    P = json.load(open(f"{R}/cache/tmp/ag_c86d3simfd_patch_{split}.json"))
    d = pd.read_csv(f"{R}/submissions/{SUB}_{det}.csv", index_col=0)
    out = {}
    for k, v in P.items():
        r = d.loc[int(k)] if int(k) in d.index else None
        if r is None or not (pd.notna(r.g) and pd.notna(r.cs) and r.cs > 0): continue
        if r.g / r.cs >= 1.5:
            out[k] = dict(value=v["value"], why="rul180 (g/cs>=1.5): " + v["why"])
    json.dump(out, open(f"{R}/cache/tmp/ag_c86d3simfdf_patch_{split}.json", "w"), indent=1)
    print(split, len(P), "->", len(out), sorted(out))
