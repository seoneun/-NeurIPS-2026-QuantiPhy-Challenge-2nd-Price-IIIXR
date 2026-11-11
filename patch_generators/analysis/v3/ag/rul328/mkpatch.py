"""rul328: FINAL answers = the replayed fusion (run_fusion.py) on rule rows whose answer differs from xt_Nzl;
rows another patch owns are skipped."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul328/out"
BASE = "xt_Nzl"
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3")
import analyze_caw as A
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
for sp, suf in (("test", "_detail.csv"), ("val", "_valdetail.csv")):
    H = json.load(open(f"{R}/cache/tmp/ag_partattr_hits_{sp}.json"))
    b = pd.read_csv(f"{R}/submissions/{BASE}{suf}").set_index("Unnamed: 0")
    h = pd.read_csv(f"{OD}/ag_partattr_pa{suf}").set_index("Unnamed: 0").reindex(b.index)
    d = ~np.isclose(b.answer.values, h.answer.values, rtol=1e-6)
    ch = list(b.index[d])
    excl = set(b.index[b.answer_source.astype(str).str.startswith("patch")])
    keep = [i for i in ch if i not in excl and str(int(i)) in H]
    other = [i for i in ch if str(int(i)) not in H]
    print(f"\n[{sp}] rule hits {len(H)}; answer changes {len(ch)} (outside the rule rows: {other}); "
          f"patched-excluded {len([i for i in ch if i in excl])}; written {len(keep)}")
    P = {}
    for i in keep:
        m = H[str(int(i))]
        P[str(int(i))] = {"value": float(h.answer[i]),
                          "why": (f"rul328: '{m['part']} {m['dim']} of {m['owner']}' parsed as the whole '{m['target']}' -> "
                                  f"geo {b.g[i]:.4g} invalid (part_target); no-geo route {h.answer_source[i]} from CaW {h.cs[i]:.4g}; "
                                  f"was {b.answer_source[i]} {b.answer[i]:.4g}")}
    json.dump(P, open(f"{R}/cache/tmp/ag_partattr_patch_{sp}.json", "w"), indent=0)
    cols = [c for c in ("video_id", "cat", "question", "answer", "answer_source", "g", "cs", "mode") if c in b.columns]
    for i in ch:
        print(i, b.loc[i, cols].to_dict(), "->", round(h.answer[i], 4), h.answer_source[i])
    if sp == "val" and "y" in b.columns and len(ch):
        m0, m1 = A.row_mra(b.answer.values, b.y.values), A.row_mra(h.answer.values, h.y.values)
        print("val macro before %.4f after %.4f" % (A.macro(pd.Series(m0, index=b.index), b["cat"])[0],
                                                     A.macro(pd.Series(m1, index=b.index), b["cat"])[0]))
from lab_rows import labelled
L = labelled("sub_xt_Nzl")
PT = {sp: json.load(open(f"{R}/cache/tmp/ag_partattr_patch_{sp}.json")) for sp in ("test", "val")}
hit = [(r_.split, int(r_.Index), r_.y, r_.m) for r_ in L.itertuples() if str(int(r_.Index)) in PT[r_.split]]
print("\nlabelled rows (val + all-237 =", len(L), ") touched:", len(hit), hit)
for k, x in PT["test"].items():
    print(k, x)
