#!/usr/bin/env python
"""ag rul424 patch: rows whose final answer differs between sub_xt_Nzc and sub_ag_stale2_Nzn (the same recipe on the
narrowed geo rj5sQ3n from narrow.py), excluding rows whose old or new answer_source is lab_pool* / depth_bound* / patch*.
OWN_ONLY=1 (default 0; outputs tagged ag_stale2_own) also requires the row's own raw geo to have changed (narrow.py take); cross-row propagations are listed.
writes cache/tmp/ag_stale2_patch_{test,val}.json and the patched submissions sub_ag_stale2_P (+ pl_ag_stale2_P for PL)."""
import json, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
BAD = ("lab_pool", "depth_bound", "patch")
OWN_ONLY = os.environ.get("OWN_ONLY", "0") == "1"
TAG = "ag_stale2_own" if OWN_ONLY else "ag_stale2"
st = lambda s: s.astype(str).str.replace("mo_guard:", "", regex=False)
for split, kind in (("test", "detail"), ("val", "valdetail")):
    a = pd.read_csv(f"{R}/submissions/sub_xt_Nzc_{kind}.csv").set_index("Unnamed: 0")
    r = pd.read_csv(f"{R}/submissions/sub_ag_stale2_Nzr_{kind}.csv").set_index("Unnamed: 0").reindex(a.index)
    b = pd.read_csv(f"{R}/submissions/sub_ag_stale2_Nzn_{kind}.csv").set_index("Unnamed: 0").reindex(a.index)
    print(f"[{split}] reproduction Nzr vs Nzc: answers differing {int((~np.isclose(a.answer, r.answer, rtol=1e-9)).sum())}, sources differing {int((a.answer_source.astype(str) != r.answer_source.astype(str)).sum())}")
    rw = pd.read_csv(f"{R}/cache/tmp/ag_stale2_rows_{split}.csv").set_index("Unnamed: 0")
    own = set(rw.index[rw["take"].astype(bool)])
    ch = ~np.isclose(a.answer, b.answer, rtol=1e-9)
    bad = st(a.answer_source).str.startswith(BAD) | st(b.answer_source).str.startswith(BAD)
    isown = pd.Series(a.index.isin(list(own)), index=a.index)
    sel = ch & ~bad & (isown | (not OWN_ONLY))
    Z = pd.DataFrame(dict(vid=a.video_id, cat=a["cat"], q=a.get("question", a.video_id).astype(str).str[:60], old=a.answer, src_old=a.answer_source, new=b.answer,
                          src_new=b.answer_source, g_old=a.get("g"), g_new=b.get("g"), cs=b.get("cs"), mode_new=b.get("mode"), own=isown, bad=bad))[ch]
    print(f"[{split}] answers changed {int(ch.sum())}: own {int((ch & isown).sum())}, propagated {int((ch & ~isown).sum())}, excluded lab/depth_bound/patch {int((ch & bad).sum())}; patch rows {int(sel.sum())} by cat {a['cat'][sel].value_counts().to_dict()}")
    print(Z.round(4).to_string())
    P = {str(int(i)): {"value": float(b.answer[i]), "why": f"xground flag on measured tracks ({'own row' if isown[i] else 'cross-row'}): {a.answer_source[i]} -> {b.answer_source[i]}"} for i in a.index[sel]}
    json.dump(P, open(f"{R}/cache/tmp/{TAG}_patch_{split}.json", "w"), indent=0)
    o = a.copy(); o.loc[sel, "answer"] = b.answer[sel]; o.loc[sel, "answer_source"] = "patch:rul424"
    if kind == "valdetail":
        sys.path.insert(0, f"{R}/caw"); import analyze_caw as A
        o["mra"] = A.row_mra(o.answer.values, o.y.values)
    o.to_csv(f"{R}/submissions/sub_{TAG}_P_{kind}.csv")
    if split == "test":
        sub = pd.read_csv(f"{R}/submissions/sub_xt_Nzc_id1.csv"); sub["parsed_value"] = o.answer.values
        sub.to_csv(f"{R}/submissions/sub_{TAG}_P_id1.csv", index=False)
        s0 = sub.rename(columns={"id": "Unnamed: 0"}); s0["Unnamed: 0"] = s0["Unnamed: 0"] - 1; s0.to_csv(f"{R}/submissions/sub_{TAG}_P.csv", index=False)
        # PL twin: pl_xt_Nzc with the same rows taken from the PL-recipe rebuild on the narrowed geo
        pa = pd.read_csv(f"{R}/submissions/pl_xt_Nzc_detail.csv").set_index("Unnamed: 0")
        pb = pd.read_csv(f"{R}/submissions/pl_ag_stale2_Nzn_detail.csv").set_index("Unnamed: 0").reindex(pa.index)
        po = pa.copy(); po.loc[sel, "answer"] = pb.answer[sel]
        po.to_csv(f"{R}/submissions/pl_{TAG}_P_detail.csv")
        pv = pd.read_csv(f"{R}/submissions/pl_xt_Nzc_valdetail.csv"); pv.to_csv(f"{R}/submissions/pl_{TAG}_P_valdetail.csv", index=False)
        ps = pd.read_csv(f"{R}/submissions/pl_xt_Nzc_id1.csv"); ps["parsed_value"] = po.answer.values
        ps.to_csv(f"{R}/submissions/pl_{TAG}_P_id1.csv", index=False)
        p0 = ps.rename(columns={"id": "Unnamed: 0"}); p0["Unnamed: 0"] = p0["Unnamed: 0"] - 1; p0.to_csv(f"{R}/submissions/pl_{TAG}_P.csv", index=False)
        print(f"[test] PL twin rows differing from pl_xt_Nzc: {int((~np.isclose(pa.answer, po.answer, rtol=1e-9)).sum())}")
