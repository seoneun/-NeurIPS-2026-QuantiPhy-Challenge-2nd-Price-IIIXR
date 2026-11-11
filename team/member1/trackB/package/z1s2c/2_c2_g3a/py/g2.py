import sys, os, json, math, numpy as np, pandas as pd
OD = sys.argv[1]; PD = f"{OD}/r36_audit"; os.makedirs(f"{OD}/patches", exist_ok=True)
BASE = {sp: pd.read_csv(f"{OD}/series/pred_t127_{sp}.csv", index_col=0)[sp] for sp in ("test", "val")}
P = {"rul098": "r36_D2_rul098", "stale2own": "r36_D2_stale2own", "A6_brkt": "r36_D3_A6_brkt", "STALE2_U": "r36_S2_STALE2_U",
     "rul441": "r36_D2_rul441", "hgt": "r36_D2_hgt", "rul244": "r36_D2_rul244", "rul016": "r36_D2_rul016",
     "second": "r36_D2_second", "rul315": "r36_D2_rul315"}
A_RULES = ["rul098", "stale2own", "A6_brkt", "STALE2_U"]
B_RULES = ["rul441", "hgt", "rul244", "rul016", "second", "rul315"]
COMBOS = {"g_AB": A_RULES + B_RULES, "g_A": A_RULES}
def chg(a, b): return a.index[np.abs(np.log(a / b.reindex(a.index))) > 1e-12]
def load(tags):
    out = {}; info = {}
    for sp in ("test", "val"):
        n = BASE[sp].copy(); seen = {}
        for t in tags:
            for k, v in json.load(open(f"{PD}/{P[t]}_{sp}.json")).items():
                i = int(k); b = float(v["before"]) if "before" in v else None
                seen[i] = t; n[i] = float(v["value"])
        out[sp] = n; info[sp] = seen
    return out, info
c9n = pd.read_csv(f"{OD}/pred_t127_id1.csv", float_precision="round_trip")
for nm, outname in (("g_AB", "pred_t129"), ("g_A", "pred_t130")):
    o, info = load(COMBOS[nm])
    for sp in ("test", "val"):
        d = {str(int(i)): dict(value=float(o[sp][i]), before=float(BASE[sp][i]), rule=info[sp][int(i)]) for i in chg(o[sp], BASE[sp])}
        json.dump(d, open(f"{OD}/patches/r36_BUILD_{nm}_{sp}.json", "w"), indent=0)
    p = json.load(open(f"{OD}/patches/r36_BUILD_{nm}_test.json")); s = c9n.copy()
    for k, v in p.items():
        r = s.index[s.id == int(k) + 1]
        s.loc[r[0], "parsed_value"] = v["value"]
    fn = f"{OD}/{outname}_id1.csv"; s.to_csv(fn, index=False)
