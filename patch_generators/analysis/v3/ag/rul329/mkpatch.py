"""rul329: diff the fusion run out/ag_partword_pw (re-measured geo raw) vs submissions/xt_Nzg (the same fusion, pipeline
raw) -> cache/tmp/ag_partword_patch_{split}.json (FINAL answers) + report. Rows already patched are skipped."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul329/out"
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3")
import analyze_caw as A
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 200)
M = {sp: (json.load(open(f"{R}/cache/tmp/ag_partword_meas_{sp}.json"))["res"] if os.path.exists(f"{R}/cache/tmp/ag_partword_meas_{sp}.json") else {})
     for sp in ("test", "val")}
for sp, suf in (("test", "_detail.csv"), ("val", "_valdetail.csv")):
    b = pd.read_csv(f"{R}/submissions/xt_Nzg{suf}").set_index("Unnamed: 0")
    s = pd.read_csv(f"{R}/submissions/sub_xt_Nzg{suf}").set_index("Unnamed: 0").reindex(b.index)
    h = pd.read_csv(f"{OD}/ag_partword_pw{suf}").set_index("Unnamed: 0").reindex(b.index)
    d = ~np.isclose(b.answer.values, h.answer.values, rtol=1e-6)
    ch = list(b.index[d])
    excl = set(s.index[s.answer_source.astype(str).str.startswith("patch")]) | set(b.index[b.answer_source.astype(str).str.startswith("patch")])
    keep = [i for i in ch if i not in excl and str(int(i)) in M[sp]]
    other = [i for i in ch if str(int(i)) not in M[sp]]
    P = {str(int(i)): {"value": float(h.answer[i]),
                       "why": (f"rul329: '{M[sp][str(int(i))]['x']}' track from the VLM-anchored v3 file; geo {h.g[i]:.4g} "
                               f"({M[sp][str(int(i))]['mode']}) -> {h.answer_source[i]} with CaW {h.cs[i]:.4g}; was {b.answer_source[i]} {s.answer[i]:.4g}")}
         for i in keep}
    json.dump(P, open(f"{R}/cache/tmp/ag_partword_patch_{sp}.json", "w"), indent=0)
    print(f"\n[{sp}] answer changes {len(ch)} (outside the re-measured rows: {other}), excluded (patched) {len([i for i in ch if i in excl])}, written {len(keep)}")
    if keep:
        T = pd.DataFrame({"vid": b.video_id[keep], "cat": b.cat[keep], "q": b.question[keep].str[:62], "g_old": b.g[keep], "g_new": h.g[keep].round(2),
                          "cs": b.cs[keep].round(2), "old(sub)": s.answer[keep].round(3), "src_old": s.answer_source[keep], "new": h.answer[keep].round(3),
                          "src_new": h.answer_source[keep]})
        print(T.to_string()); print("by cat:", T.cat.value_counts().to_dict())
    if sp == "val" and "y" in b.columns:
        m0, m1 = A.row_mra(s.answer.values, s.y.values), A.row_mra(h.answer.values, h.y.values)
        print("val macro before %.4f after %.4f" % (A.macro(pd.Series(m0, index=s.index), s.cat)[0], A.macro(pd.Series(m1, index=s.index), s.cat)[0]))
from lab_rows import labelled
L = labelled("sub_xt_Nzg")
PT = {sp: json.load(open(f"{R}/cache/tmp/ag_partword_patch_{sp}.json")) for sp in ("test", "val")}
hit = [(r_.split, int(r_.Index), r_.y, r_.m) for r_ in L.itertuples() if str(int(r_.Index)) in PT[r_.split]]
print("\nlabelled rows (val + all-237 =", len(L), ") touched:", len(hit), hit)
for i, x in PT["test"].items():
    print(i, x)
