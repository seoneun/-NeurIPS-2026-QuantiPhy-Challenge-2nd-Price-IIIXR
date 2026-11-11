"""partattr_rep.py BASE: per-question version of analysis/v3/ag/rul328 (run_fusion.py raw + fuse, mkpatch.py).
Rule (unchanged, analysis/v3/ag/rul328/rule.py imported read-only): the row's own question matches
'what is the <part> <length|width|height|diameter> of <owner> in' with a part word and the row's own parse target is the
owner -> the row's geo measures the whole owner and is set to the pipeline's no-geo state (run_fusion.py raw: geo / meas
columns NaN, meas_valid False, meas_why part_target, answer_source vlm, parsed_value = vlm_value), here in the main clean
geo table. Changed vs production: the fusion is the clean strict fusion (grpC_strict.py) instead of the non-strict
xt_Nzl replay (var_norm, twin pooling, 65 patches); the base is BASE (sub_xt_clean4f) instead of xt_Nzl.
Patch = rows whose strict answer changes vs BASE -> cache/tmp/ag_partattr_patch_{test,val}.json.
Value = the main strict fusion answer (production run_fusion.py raw changed the main geo raw only)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
sys.path.insert(0, f"{R}/analysis/v3/ag/rul328")
import rule  # noqa: E402

COLS = ("geo_value", "meas_value", "meas_mode", "meas_f", "meas_mpp", "meas_grav", "meas_gap", "meas_owlvote", "meas_nseg",
        "meas_colswap", "merge_src")
HITS = {}


def edit(sp, u, raw):
    P = rule.parse(sp)
    hits = {int(i): m for i in raw.index for m in [rule.match(raw.question[i], P.get(int(i)))] if m}
    HITS[sp] = hits
    ch = []
    for i in hits:
        if np.isnan(float(raw.geo_value.get(i, np.nan))) and not bool(raw.meas_valid.get(i)):
            continue
        print(sp, u or "m", i, raw.video_id[i], "geo", raw.geo_value[i], raw.meas_mode[i], "->", "part_target")
        for c in COLS:
            if c in raw.columns:
                raw.loc[i, c] = np.nan
        raw.loc[i, "meas_valid"] = False
        raw.loc[i, "meas_why"] = "part_target"
        raw.loc[i, "answer_source"] = "vlm"
        raw.loc[i, "parsed_value"] = raw.vlm_value[i]
        ch.append(i)
    return ch


out = G.run("rul328", edit)
for sp, (n, b, bs, rows) in out.items():
    keep = G.gate(n, b, bs, rows)
    P = {}
    for i in keep:
        m = HITS[sp][i]
        P[i] = {"value": float(n.answer[i]),
                "why": (f"rul328: '{m['part']} {m['dim']} of {m['owner']}' parsed as the whole '{m['target']}' -> geo {b.g[i]:.4g} "
                        f"invalid (part_target); strict no-geo route {n.answer_source[i]} from CaW {n.cs[i]:.4g}; "
                        f"was {b.answer_source[i]} {b.answer[i]:.4g}")}
        print(sp, i, P[i])
    print(f"[rul328 {sp}] rule hits {len(HITS[sp])}, geo rows invalidated {len(rows)}, written {len(keep)}")
    G.write_patch("rul328", sp, P)
