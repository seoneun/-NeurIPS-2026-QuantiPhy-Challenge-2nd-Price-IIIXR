"""rul327: diff the fusion run out/ag_parse3_hb (pipeline fusion + hand bound) vs submissions/xt_Nzh (same fusion, no wrap)
-> cache/tmp/ag_parse3_patch_{split}.json (FINAL answers; rows already patched are skipped) + report + bound sensitivity."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul327/out"
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/analysis/v3/ag/rul327")
import analyze_caw as A
import kinematic as K
import handbound as H
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 200)
TAG = sys.argv[1] if len(sys.argv) > 1 else "ag_parse3_hb"
for sp, suf in (("test", "_detail.csv"), ("val", "_valdetail.csv")):
    b = pd.read_csv(f"{R}/submissions/xt_Nzh{suf}").set_index("Unnamed: 0")
    s = pd.read_csv(f"{R}/submissions/sub_xt_Nzh{suf}").set_index("Unnamed: 0").reindex(b.index)
    h = pd.read_csv(f"{OD}/{TAG}{suf}").set_index("Unnamed: 0").reindex(b.index)
    d = ~np.isclose(b.answer.values, h.answer.values, rtol=1e-9, equal_nan=True)
    ch = list(b.index[d])
    excl = set(s.index[s.answer_source.astype(str).str.startswith("patch")]) | set(b.index[b.answer_source.astype(str).str.startswith("patch")])
    keep = [i for i in ch if i not in excl]
    PR = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); PR[r["id"]] = r
    def why(i):
        r = PR[int(i)]; bd = H.hand_bound(r["question"], r.get("prior_text"), r["q"].get("kind"))
        k = H.kind_of(r["question"], r["q"].get("kind"))
        return (f"rul327 hand bound: {k} of a hand/finger in a close-up (prior {H.prior_len_m(r.get('prior_text'))} m <= 0.3) "
                f"must lie in {bd} m/s; CaW {b.cs[i]:.4g} outside, geo {b.g[i]:.4g} inside -> {h.answer_source[i]}; was {s.answer_source[i]} {s.answer[i]:.4g}")
    P = {str(int(i)): {"value": float(h.answer[i]), "why": why(i)} for i in keep}
    json.dump(P, open(f"{R}/cache/tmp/ag_parse3_patch_{sp}.json", "w"), indent=0)
    print(f"\n[{sp}] answer changes vs xt_Nzh: {len(ch)}; excluded (patched) {len([i for i in ch if i in excl])}; written {len(keep)}")
    if ch:
        cols = [c for c in ["video_id", "cat", "question", "g", "cs", "answer", "answer_source"] if c in b.columns]
        T = pd.DataFrame({"old": s.answer[ch], "src_old": s.answer_source[ch], "new": h.answer[ch], "src_new": h.answer_source[ch]})
        T = pd.concat([b.loc[ch, [c for c in ["video_id", "cat", "question", "g", "cs"] if c in b.columns]], T], axis=1)
        print(T.to_string())
        print("by cat:", T.cat.value_counts().to_dict() if "cat" in T else None)
    if sp == "val" and "y" in b.columns:
        m0, m1 = A.row_mra(s.answer.values, s.y.values), A.row_mra(h.answer.values, h.y.values)
        print("val macro before %.4f after %.4f" % (A.macro(pd.Series(m0, index=s.index), s.cat)[0], A.macro(pd.Series(m1, index=s.index), s.cat)[0]))
from lab_rows import labelled
L = labelled("sub_xt_Nzh")
PT = {sp: json.load(open(f"{R}/cache/tmp/ag_parse3_patch_{sp}.json")) for sp in ("test", "val")}
hit = [(r_.split, int(r_.Index), r_.y, r_.m) for r_ in L.itertuples() if str(int(r_.Index)) in PT[r_.split]]
print("\nlabelled rows (val + all-237 =", len(L), ") touched:", len(hit), hit)
# which labelled rows does the hand rule even recognise?
P = {}
for sp in ("test", "val"):
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); P[(sp, r["id"])] = r
nl = [(sp_, i) for (sp_, i) in [(r_.split, int(r_.Index)) for r_ in L.itertuples()] if H.hand_target(P[(sp_, i)]["question"])]
print("labelled rows with a hand/finger speed target:", nl)
# sensitivity: rows the bound would change at other thresholds (offline replica of the phys_bounds step on the final table)
print("\nhand-target rows in test and the bound decision:")
for sp, suf in (("test", "_detail.csv"),):
    s = pd.read_csv(f"{R}/submissions/sub_xt_Nzh{suf}").set_index("Unnamed: 0")
    for (sp_, i), r in sorted(P.items()):
        if sp_ != sp or not H.hand_target(r["question"]):
            continue
        f = K.q_si(r["question"]); k = H.kind_of(r["question"], r["q"].get("kind"))
        print(f"  {i} kind {r['q'].get('kind')}->{k} prior_len {H.prior_len_m(r.get('prior_text'))} bound {H.hand_bound(r['question'], r.get('prior_text'), r['q'].get('kind'))} "
              f"| ans {s.answer[i]:.4g} {s.answer_source[i]} g {s.g[i]:.4g} cs {s.cs[i]:.4g} (SI x{f}) | norm: {H.norm(r['question'])[:70]}")
    for ahi in (1.0, 1.25, 1.5, 2.0, 2.2, 2.5):
        H.AVG_HI = ahi; H.INST_HI = ahi + 1.0
        fire = []
        for (sp_, i), r in P.items():
            if sp_ != sp: continue
            bd = H.hand_bound(r["question"], r.get("prior_text"), r["q"].get("kind"))
            if bd is None: continue
            f = K.q_si(r["question"]); inside = lambda x: np.isfinite(x) and x > 0 and bd[0] <= x * f <= bd[1]
            if inside(s.answer[i]): continue
            c = [x for x in (s.g[i], s.cs[i]) if inside(x)]
            if len(c) == 1: fire.append((i, round(float(s.answer[i]), 4), round(float(c[0]), 4)))
        print(f"  avg hi {ahi} / inst hi {ahi + 1.0}: fires on {fire}")
