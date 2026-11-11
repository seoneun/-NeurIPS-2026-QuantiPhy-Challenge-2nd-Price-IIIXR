"""rul044 step 5: FINAL answers from the fusion run out/ag_c30win_raw (submit_v3 unmodified, Nzr recipe, re-measured geo raw,
rul424 left out) -> cache/tmp/ag_c30win_patch_{test,val}.json, plus evaluation.
Checks: the raw run differs from the control run (same recipe, pipeline geo raw, rul424 left out: out/ag_c30win_nost2) ONLY on
accepted rows.  Evaluation: labelled rows (val + all-237) before / after; the time-window mechanism on labelled rows
(cache/tmp/ag_twin_mech.csv: windowed vs whole-clip geo on rows whose window the parser did read; labels evaluate only)."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul044/out"
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw")
import analyze_caw as A
from lab_rows import labelled
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
L = labelled("sub_xt_Nzr")
P = {}
for sp, suf in (("test", "_detail.csv"), ("val", "_valdetail.csv")):
    s = pd.read_csv(f"{R}/submissions/sub_xt_Nzr{suf}").set_index("Unnamed: 0")
    rec = json.load(open(f"{R}/cache/tmp/ag_c30win_rec_{sp}.json"))
    acc = {int(k) for k, v in rec.items() if v.get("accept")}
    pk = {}
    if acc:
        h = pd.read_csv(f"{OD}/ag_c30win_raw{suf}").set_index("Unnamed: 0").reindex(s.index)
        n0 = pd.read_csv(f"{OD}/ag_c30win_nost2{suf}").set_index("Unnamed: 0").reindex(s.index)
        d = ~np.isclose(n0.answer.values, h.answer.values, rtol=1e-9, equal_nan=True)
        ch = set(s.index[d])
        print(f"[{sp}] raw-run vs control-run answer changes: {sorted(ch)}; accepted rows: {sorted(acc)}; changes outside accepted: {sorted(ch - acc)}")
        assert not (ch - acc), "the re-measured raw changed rows it does not own"
        for i in sorted(acc):
            r = rec[str(i)]
            val = float(h.answer[i])
            if np.isclose(val, s.answer[i], rtol=1e-6):
                print(f"[{sp}] {i}: final answer unchanged ({val:.5g})"); continue
            why = (f"rul044: time window 'from {float(r['t1']):g} to {float(r['t2']):g}' without the seconds unit was not parsed (whole clip "
                   f"measured); re-parsed t1={float(r['t1']):g}, t2={float(r['t2']):g}; geo re-measured on the row's own track "
                   f"{r['base']:.4g} -> {r['new']:.4g} ({r['mode']}); fused {h.answer_source[i]} with CaW {h.cs[i]:.4g}; "
                   f"was {s.answer_source[i]} {s.answer[i]:.4g}")
            pk[str(i)] = dict(value=val, why=why)
    json.dump(pk, open(f"{R}/cache/tmp/ag_c30win_patch_{sp}.json", "w"), indent=1)
    P[sp] = pk
    print(f"[{sp}] patch rows: {len(pk)}")
    for i, x in pk.items():
        i = int(i)
        print(f"   {i} {s.video_id[i]} {s.cat[i]} | {str(s.question[i]).strip()} | {s.answer[i]:.4g} ({s.answer_source[i]}) -> {x['value']:.4g}")
    if sp == "test":
        cats = pd.Series([s.cat[int(i)] for i in pk]).value_counts().to_dict() if pk else {}
        print(f"[test] changed rows by category: {cats}")

# labelled evaluation (val + all-237)
print("\n== labelled evaluation (sub_xt_Nzr -> + rul044)")
for name, sel in (("val", L.split == "val"), ("all-237", slice(None))):
    LL = L[sel].copy()
    LL["id"] = LL.index
    LL = LL.reset_index(drop=True)
    new = LL.answer.copy()
    for k in LL.index:
        pk = P.get(LL.split[k], {})
        if str(int(LL.id[k])) in pk:
            new[k] = pk[str(int(LL.id[k]))]["value"]
    m1 = A.row_mra(new.values, LL.y.values)
    chg = LL.index[~np.isclose(new.values, LL.answer.values)]
    b, a = A.macro(pd.Series(LL.m.values), LL.cat)[0], A.macro(pd.Series(m1), LL.cat)[0]
    print(f"{name}: n={len(LL)} rows changed {len(chg)}  macro {b:.4f} -> {a:.4f}")
    for k in chg:
        print(f"   {LL.split[k]}:{LL.id[k]} y={LL.y[k]:.4g} {LL.answer[k]:.4g} -> {new[k]:.4g}  MRA {LL.m[k]:.3f} -> {m1[k]:.3f}")
test_lab = {int(i) for i in P.get("test", {})} & set(L[L.split == "test"].index)
print("labelled test rows touched:", sorted(test_lab))

# mechanism: windowed vs whole-clip geo on labelled rows whose window the parser read (twin_mech, labels evaluate only)
mp = f"{R}/cache/tmp/ag_twin_mech.csv"
if os.path.exists(mp):
    D = pd.read_csv(mp)
    B = D[D.g_win.notna() & D.g_whole.notna()].copy()
    print(f"\n== mechanism (ag_twin_mech.csv, {len(B)} labelled windowed rows measured both ways)")
    print(B[["split", "id", "video_id", "kind", "t1", "t2", "dur", "frac", "y", "g_win", "g_whole", "m_win", "m_whole"]].round(3).to_string(index=False))
    for lo, hi in ((0, 0.9), (0.6, 0.95), (0, 1.01)):
        g = B[(B.frac >= lo) & (B.frac < hi)]
        if len(g):
            print(f"  frac [{lo},{hi}): n={len(g)} MRA windowed {g.m_win.mean():.3f} whole-clip {g.m_whole.mean():.3f}  "
                  f"win better {(g.m_win > g.m_whole + 1e-9).sum()} worse {(g.m_win < g.m_whole - 1e-9).sum()}")
