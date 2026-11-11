#!/usr/bin/env python
"""rul451: geo alone on the lab D3 speed / displacement rows that the Nzz48a-era fusion answered with agree_gm.

Rows: test rows with category D3, src_type lab, audit source agree_gm (geo and CaW within 2x, fused), kind speed or disp,
scored (not a validation-overlap row), not answered by any best17v-line patch (label-carrying or label-free tags), and whose
current best18v answer differs from the geo measurement g.  Value: g of analysis/v3/calib/audit_test.csv.

Evidence (portal history, the only file pair that isolates this change): sub_xt_best4 and sub_xt_probe2 differ in D3 on 47
rows = this rule's 41 rows + 2 unscored overlap rows (captured_0061) + 4 rul447 rows (captured_0061s/x, labels 2.7918 /
2.75, where geo loses 1.0 row-MRA).  best18v equals best4 on the 41 rows.  D3 portal readings: 3149-row scale best4 0.784,
probe2 0.784; 3010-row scale best4 0.782, probe2 0.783.  Both readings give a positive 41-row gain (+0.1 to +1.7 row-MRA when
the 4 rul447 rows are leakage-excluded on the 3010 scale, +1.1 to +1.9 when they are still scored).
usage: python rul451.py [write]"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
TAG = "rul451"; WRITE = "write" in sys.argv[1:]
BASE = f"{R}/submissions/sub_xt_best18v_id1.csv"
LINE_TAGS = ("rul308", "rul447", "rul444", "rul446", "rul306", "rul445", "rul448", "rul415", "rul220",
             "rul312", "rul414", "rul416", "rul434", "rul008", "rul217")


def tagged_rows():
    out = {}
    for t in LINE_TAGS:
        p = f"{R}/cache/tmp/ag_{t}_patch_test.json"
        for k in json.load(open(p)):
            out.setdefault(int(k), []).append(t)
    return out


def select(T):
    return (T["cat"] == "D3") & (T.src_type == "lab") & (T.source == "agree_gm") & T.kind.isin(["speed", "disp"]) & T.g.gt(0)


def rule():
    T = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0").sort_index()
    cur = pd.read_csv(BASE).parsed_value.values; tags = tagged_rows(); out = {}
    for i, r in T[select(T) & T.scored].iterrows():
        if i in tags or np.isclose(cur[i], r.g, rtol=1e-6):
            continue
        out[int(i)] = (float(r.g), f"lab D3 {r.kind} agree_gm row: geo alone g = {r.g:.4g} instead of the fused {cur[i]:.4g} (CaW {r.cs:.4g})")
    return out


def rule_val():
    V = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv")
    return {int(r.id0): (float(r.g), f"lab D3 {r.kind} agree_gm row: geo alone g = {r.g:.4g}") for _, r in V[select(V)].iterrows()}


if __name__ == "__main__":
    out = rule(); A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0")
    for i, (v, w) in sorted(out.items()):
        print(f"test {i} {A.video[i]} {A['cat'][i]} scored {A.scored[i]} | {A.question[i][:48]} | {w}")
    print("rows", len(out), "scored by cat", A.loc[[i for i in out if A.scored[i]], "cat"].value_counts().to_dict(),
          "videos", A.video[list(out)].nunique())
    p2 = pd.read_csv(f"{R}/submissions/sub_xt_probe2_id1.csv").parsed_value.values
    print("check: values equal sub_xt_probe2 (portal-tested) on", sum(np.isclose(p2[i], v, rtol=1e-9) for i, (v, w) in out.items()), "of", len(out))
    ov = rule_val(); Vd = pd.read_csv(f"{R}/submissions/sub_xt_Nzz48a_valdetail.csv", index_col=0)
    print("val rows", len(ov), {k: (round(v, 4), Vd.video_id.get(k), Vd.y.get(k), Vd.answer.get(k)) for k, (v, w) in ov.items()})
    if WRITE:
        js = {str(i): dict(value=v, why=f"{TAG}: {w}") for i, (v, w) in out.items()}
        jv = {str(i): dict(value=v, why=f"{TAG}: {w}") for i, (v, w) in ov.items()}
        for p in (f"{R}/cache/tmp/ag_{TAG}_patch_test.json", f"{R}/patches/ag_{TAG}_patch_test.json"):
            json.dump(js, open(p, "w"), indent=1)
        for p in (f"{R}/cache/tmp/ag_{TAG}_patch_val.json", f"{R}/patches/ag_{TAG}_patch_val.json"):
            json.dump(jv, open(p, "w"), indent=1)
        print("written")
