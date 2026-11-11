#!/usr/bin/env python
"""make_c4.py W TAG:RUN [TAG:RUN ...]: C4 patches of the repaired CHECK tags (notes/check_repair.md).
For each tag T, the patch = W/RUN/new/ag_T_patch_<split>.json of the rerun_clean.sh run W/RUN that decided T (RUN = rep for
the repaired builders of clean_builders_rep.tsv; main / fix for tags whose clean4 re-run is used as is), rows with a finite
value > 0 -> cache/tmp/ag_<T>C4_patch_{test,val}.json. The rc of T's last builder line in W/RUN/rerun.log must be 0;
otherwise T gets an empty patch and a WARN line (as build_clean4.sh step 2 does for C3). Writes W/ctags4.tsv.
No stored production value is read."""
import sys, os, json, math
W, specs = sys.argv[1], [a.split(":") for a in sys.argv[2:]]
rc = {}
for run in sorted({r for _, r in specs}):
    for l in open(f"{W}/{run}/rerun.log"):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 3 and f[2].startswith("rc="): rc[(run, f[0])] = f[2][3:]       # last line of the tag wins
st = ["tag\trun\trc_last\ttest_rows\tval_rows"]
for t, run in specs:
    r = rc.get((run, t), "missing"); n = {}
    for sp in ("test", "val"):
        p = f"{W}/{run}/new/ag_{t}_patch_{sp}.json"; o = {}
        if r == "0" and os.path.exists(p):
            for k, e in json.load(open(p)).items():
                v = e.get("value")
                if v is None or not math.isfinite(float(v)) or float(v) <= 0: continue
                o[str(int(k))] = dict(value=float(v), why=f"{t}C4: {t} repaired builder re-run on the clean base ({run}) | {str(e.get('why', ''))[:200]}")
        json.dump(o, open(f"cache/tmp/ag_{t}C4_patch_{sp}.json", "w"), indent=0); n[sp] = len(o)
    if r != "0" or n["test"] == 0: print(f"WARN C4 tag {t}: run {run} rc {r}, test rows {n['test']}")
    st.append(f"{t}\t{run}\t{r}\t{n['test']}\t{n['val']}")
open(f"{W}/ctags4.tsv", "w").write("\n".join(st) + "\n")
print(f"C4 patches: {len(specs)} tags, {sum(int(s.split(chr(9))[3]) for s in st[1:])} test rows")
