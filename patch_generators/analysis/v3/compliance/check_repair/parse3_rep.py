"""parse3_rep.py BASE: per-question version of analysis/v3/ag/rul327 (run_fusion.py ag_parse3_hb, mkpatch.py).
Rule (unchanged): kinematic.class_bound is wrapped with the hand / finger close-up speed bound of
analysis/v3/ag/rul327/handbound.py (own question and own prior text only); the fusion's --phys_bounds / --phys_bounds_final
steps then route an answer outside the bound to the candidate (geo or CaW) inside it.
Changed vs production: the fusion is the clean strict fusion (grpC_strict.py; strict_build.sh keeps --phys_bounds extra3
and --phys_bounds_final) run through the launcher parse3_pre_hbpy.py, on the unchanged clean geo tables, instead of the
non-strict xt_Nzh replay; the base is BASE (sub_xt_clean4f) instead of xt_Nzh.
Patch = rows with a hand bound whose strict answer changes vs BASE -> cache/tmp/ag_parse3_patch_{test,val}.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
sys.path.insert(0, f"{R}/analysis/v3/ag/rul327")
import handbound as H  # noqa: E402

out = G.run("rul327", lambda sp, u, raw: [], python=f"python {R}/analysis/v3/compliance/check_repair/parse3_pre_hbpy.py")
for sp, (n, b, bs, _) in out.items():
    PR = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l); PR[r["id"]] = r
    rows = [i for i in b.index if int(i) in PR and H.hand_bound(PR[int(i)]["question"], PR[int(i)].get("prior_text"),
                                                                  PR[int(i)]["q"].get("kind")) is not None]
    keep = G.gate(n, b, bs, rows)
    P = {}
    for i in keep:
        r = PR[int(i)]
        bd = H.hand_bound(r["question"], r.get("prior_text"), r["q"].get("kind"))
        k = H.kind_of(r["question"], r["q"].get("kind"))
        P[i] = {"value": float(n.answer[i]),
                "why": (f"rul327 hand bound: {k} of a hand/finger in a close-up (prior {H.prior_len_m(r.get('prior_text'))} m <= 0.3) "
                        f"must lie in {bd} m/s; strict fusion CaW {b.cs[i]:.4g}, geo {b.g[i]:.4g} -> {n.answer_source[i]}; "
                        f"was {b.answer_source[i]} {b.answer[i]:.4g}")}
        print(sp, i, P[i])
    print(f"[rul327 {sp}] rows with a hand bound {len(rows)} {rows[:20]}, written {len(keep)}")
    G.write_patch("rul327", sp, P)
