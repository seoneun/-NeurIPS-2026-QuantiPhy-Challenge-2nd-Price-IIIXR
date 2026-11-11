"""rul024: diff the fusion run out/ag_c22rngparse_fix (re-measured geo raw) vs out/ag_c22rngparse_base (same fusion,
pipeline raw; must equal submissions/xt_Nzo) -> cache/tmp/ag_c22rngparse_patch_{split}.json (FINAL answers) + report.
Rows already patched (answer_source patch*) are skipped."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul024/out"
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3")
import analyze_caw as A
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 75); pd.set_option("display.max_rows", 200)
M = json.load(open(f"{R}/cache/tmp/ag_c22rngparse_meas.json"))
for sp, suf in (("test", "_detail.csv"), ("val", "_valdetail.csv")):
    x0 = pd.read_csv(f"{R}/submissions/xt_Nzo{suf}").set_index("Unnamed: 0")
    s = pd.read_csv(f"{R}/submissions/sub_xt_Nzo{suf}").set_index("Unnamed: 0").reindex(x0.index)
    b = pd.read_csv(f"{OD}/ag_c22rngparse_base{suf}").set_index("Unnamed: 0").reindex(x0.index)
    h = pd.read_csv(f"{OD}/ag_c22rngparse_fix{suf}").set_index("Unnamed: 0").reindex(x0.index)
    rep = np.isclose(x0.answer.values, b.answer.values, rtol=1e-9, equal_nan=True)
    print(f"[{sp}] base reproduces xt_Nzo: {int(rep.sum())}/{len(rep)}")
    d = ~np.isclose(b.answer.values, h.answer.values, rtol=1e-9, equal_nan=True)
    ch = list(b.index[d])
    excl = set(s.index[s.answer_source.astype(str).str.startswith("patch")]) | set(b.index[b.answer_source.astype(str).str.startswith("patch")])
    keep = [i for i in ch if i not in excl and str(int(i)) in M[sp]]
    other = [i for i in ch if str(int(i)) not in M[sp]]
    P = {}
    for i in keep:
        m = M[sp][str(int(i))]
        P[str(int(i))] = {"value": float(h.answer[i]),
                          "why": (f"rul024: time range without unit ('between {m['t1']:g}s and {m['t2']:g}' parsed as t1={m['t1']:g}, "
                                  f"t2={m['t2']:g}; target 'ball between ...' -> 'ball'); geo {m['mode']} rl{m['k']} disp over [{m['t1']:g}, {m['t2']:g}] "
                                  f"{h.g[i]:.4g} -> {h.answer_source[i]}; was {s.answer_source[i]} {s.answer[i]:.4g} (whole-clip window)")}
    json.dump(P, open(f"{R}/cache/tmp/ag_c22rngparse_patch_{sp}.json", "w"), indent=0)
    print(f"[{sp}] answer changes {len(ch)} (outside the re-measured rows: {other}), excluded (patched) {len([i for i in ch if i in excl])}, written {len(keep)}")
    if ch:
        T = pd.DataFrame({"vid": b.video_id[ch], "cat": b.cat[ch], "q": b.question[ch].str[:72], "g_new": h.g[ch].round(4),
                          "cs": b.cs[ch].round(3), "old(sub)": s.answer[ch].round(4), "src_old": s.answer_source[ch], "new": h.answer[ch].round(4),
                          "src_new": h.answer_source[ch]})
        print(T.to_string()); print("by cat:", T.cat.value_counts().to_dict())
    if sp == "test":
        v = b.video_id == "internet_0065"
        print(pd.DataFrame({"q": b.question[v].str[:72], "sub": s.answer[v].round(4), "fix": h.answer[v].round(4), "src": h.answer_source[v]}).to_string())
    if sp == "val" and "y" in b.columns:
        m0, m1 = A.row_mra(s.answer.values, s.y.values), A.row_mra(h.answer.values, h.y.values)
        print("val macro (fusion, before patches outside) sub %.4f fix-run %.4f" % (A.macro(pd.Series(m0, index=s.index), s.cat)[0], A.macro(pd.Series(m1, index=s.index), s.cat)[0]))
from lab_rows import labelled
L = labelled("sub_xt_Nzo")
PT = {sp: json.load(open(f"{R}/cache/tmp/ag_c22rngparse_patch_{sp}.json")) for sp in ("test", "val")}
hit = [(r_.split, int(r_.Index), r_.y, r_.m) for r_ in L.itertuples() if str(int(r_.Index)) in PT[r_.split]]
print("\nlabelled rows (val + all-237 =", len(L), ") touched:", len(hit), hit)
print("labelled rows of internet_0065:", L[L.video_id == "internet_0065"][["split", "y", "m"]].to_string() if (L.video_id == "internet_0065").any() else "none")
for i, x in PT["test"].items():
    print(i, x)
