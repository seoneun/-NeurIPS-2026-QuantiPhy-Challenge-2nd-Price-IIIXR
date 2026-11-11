"""same-rule check for grpA tags: python cmp.py RUNDIR"""
import sys, os, json, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; RUN = sys.argv[1]
BASE = dict(rul435="sub_xt_Nzk", rul021="sub_xt_Nzo", rul045="sub_xt_Nzs", rul057="sub_xt_Nzw", rul436="sub_xt_Nze",
            rul114="sub_xt_Nzz15", rul123="sub_xt_Nzz17", rul139="sub_xt_Nzz21", rul193="sub_xt_Nzz27")
eq = lambda a, b: abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-12)
def stored(tag, sp):
    for p in (f"{R}/patches/ag_{tag}_patch_{sp}.json", f"{R}/cache/tmp/ag_{tag}_patch_{sp}.json"):
        if os.path.exists(p): return {int(k): float(v["value"]) for k, v in json.load(open(p)).items()}, p
    return None, None
G = {}
for sp in ("test", "val"):
    g0 = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0, usecols=lambda c: c in ("Unnamed: 0", "geo_value", "meas_mode"))
    g1 = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntffmvsrngsr_d_vw_raw.csv", index_col=0, usecols=lambda c: c in ("Unnamed: 0", "geo_value", "meas_mode"))
    g1 = g1.reindex(g0.index); v0, v1 = g0.geo_value.astype(float), g1.geo_value.astype(float)
    same = ((np.isclose(v0, v1, rtol=1e-9) | (v0.isna() & v1.isna())) & (g0.meas_mode.astype(str) == g1.meas_mode.astype(str)))
    G[sp] = (g0, g1, same)
cols = ["answer", "answer_source", "cs", "g"]
rows = []
for tag, bsub in BASE.items():
    for sp in ("test", "val"):
        suf = "detail" if sp == "test" else "valdetail"
        S, sp_path = stored(tag, sp)
        np_ = f"{RUN}/new/ag_{tag}_patch_{sp}.json"
        N = {int(k): float(v["value"]) for k, v in json.load(open(np_)).items()} if os.path.exists(np_) else None
        if S is None or N is None:
            rows.append(dict(tag=tag, sp=sp, note=f"stored {sp_path} new {os.path.exists(np_)}")); continue
        g0, g1, same = G[sp]
        both = sorted(set(S) & set(N)); unch = [i for i in both if bool(same.get(i, False))]
        r = dict(tag=tag, sp=sp, stored_n=len(S), new_n=len(N), both_n=len(both), both_eq=sum(eq(S[i], N[i]) for i in both),
                 unch_n=len(unch), unch_eq=sum(eq(S[i], N[i]) for i in unch),
                 only_stored=sorted(set(S) - set(N)), only_new=sorted(set(N) - set(S)))
        r["C2"] = r["unch_n"] >= 1 and r["unch_eq"] == r["unch_n"]; r["C3"] = r["both_n"] >= 1 and r["both_eq"] == r["both_n"]
        rows.append(r)
        # detail of differing / missing rows
        pb = f"{R}/submissions/{bsub}_{suf}.csv"; cb = f"{R}/submissions/sub_xt_clean4f_{suf}.csv"
        P = pd.read_csv(pb, index_col=0) if os.path.exists(pb) else None; C = pd.read_csv(cb, index_col=0)
        for i in sorted(set(S) | set(N)):
            if i in S and i in N and eq(S[i], N[i]): continue
            d = dict(tag=tag, sp=sp, id=i, stored=S.get(i), new=N.get(i), geo_same=bool(same.get(i, False)),
                     geo_rj=g0.geo_value.get(i), mode_rj=g0.meas_mode.get(i), geo_ff=g1.geo_value.get(i), mode_ff=g1.meas_mode.get(i))
            for c in cols:
                if P is not None and c in P.columns: d["prod_" + c] = P[c].get(i)
                if c in C.columns: d["cln_" + c] = C[c].get(i)
            print("DIFF", d)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(pd.DataFrame(rows).to_string())
