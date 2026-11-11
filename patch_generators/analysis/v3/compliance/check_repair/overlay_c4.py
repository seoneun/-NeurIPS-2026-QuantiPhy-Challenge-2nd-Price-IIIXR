#!/usr/bin/env python
"""overlay_c4.py NAME C4TAG [C4TAG ...]: submissions/sub_xt_<NAME>_id1.csv = clean4 + the C4 patches, with the precedence that
build_clean4.sh would give them if the C4 tags were inserted into its apply list at the P_NZZ48A position of the original tag
(notes/check_repair.md 7). Inputs: submissions/sub_xt_clean4_id1.csv, submissions/sub_xt_clean4f_id1.csv (fusion),
cache/tmp/strictfill2/clean4/{order.txt,writers.csv} (clean4's apply list and the tags holding each test row),
cache/tmp/ag_<tag>C4_patch_test.json.
Per test row held by >= 1 C4 tag (C4 tags in P_NZZ48A order, the later one wins among them):
  - a clean4 tag later in the apply list holds the row (a fusion-time C3/S2 tag with a later P_NZZ48A position, or any
    post-fusion tag) -> the clean4 final value stays (that tag wins in the build);
  - else the C4 value. In the build, strict_final and labmoP skip rows whose patched value differs from the fusion answer
    (strict_final.py:17, labmoP.py:48-50), so a C4 value != fusion answer survives both. A C4 value equal to the fusion
    answer keeps the clean4 final value (flagged 'c4_eq_fusion').
Writes OUT/<NAME>_rows.csv (scratch dir from env OUT, default cache/tmp/strictfill2/check_repair) with the decision per row."""
import sys, os, json, subprocess
import numpy as np, pandas as pd
nm, c4 = sys.argv[1], sys.argv[2:]
OUT = os.environ.get("OUT", "cache/tmp/strictfill2/check_repair")
W = "cache/tmp/strictfill2/clean4"
pnz = subprocess.run(["bash", "-c", 'source scripts/common.sh > /dev/null 2>&1; source scripts/fusion_recipe.sh > /dev/null 2>&1; echo "$P_NZZ48A"'],
                     capture_output=True, text=True).stdout.strip()
pos = {t: i for i, t in enumerate(x for x in pnz.split(",") if x)}
order = open(f"{W}/order.txt").read().split()
post0 = order.index("cutlb3dP")                       # first post-fusion tag of the clean4 apply list
def opos(t):                                          # P_NZZ48A position of a clean4 fusion-time tag (C3 / S2 suffix)
    for suf in ("C3", "S2"):
        if t.endswith(suf) and t[:-len(suf)] in pos: return pos[t[:-len(suf)]]
    raise KeyError(t)
fus = set(order[:post0]); post = set(order[post0:])
c4 = sorted(c4, key=lambda t: pos[t[:-2]])
fin = pd.read_csv("submissions/sub_xt_clean4_id1.csv"); f = pd.read_csv("submissions/sub_xt_clean4f_id1.csv").parsed_value.values
wr = pd.read_csv(f"{W}/writers.csv").set_index("row")
o = fin.copy(); rows = []
for t in c4:
    for k, e in json.load(open(f"cache/tmp/ag_{t}_patch_test.json")).items():
        rows.append((int(k), t, float(e["value"])))
R = pd.DataFrame(rows, columns=["row", "tag", "c4"])
dec = []
for i, g in R.groupby("row", sort=True):
    t, v = g.iloc[-1].tag, g.iloc[-1].c4                # later C4 tag in P_NZZ48A order wins
    held = str(wr.loc[i, "tags"]) if isinstance(wr.loc[i, "tags"], str) else ""
    later = [x for x in held.split() if x in post or (x in fus and opos(x) > pos[t[:-2]])]
    c4fus = bool(np.isclose(v, f[i], rtol=1e-9))
    use = not later and not c4fus
    if use: o.loc[i, "parsed_value"] = v
    dec.append(dict(row=i, c4_tags=" ".join(g.tag), winner_c4=t, c4=v, clean4=fin.parsed_value[i], fusion=f[i],
                    clean4_writer=wr.loc[i, "final_writer"], later_tags=" ".join(later), c4_eq_fusion=c4fus, used=use,
                    changed=use and not np.isclose(v, fin.parsed_value[i], rtol=1e-9)))
D = pd.DataFrame(dec); os.makedirs(OUT, exist_ok=True); D.to_csv(f"{OUT}/{nm}_rows.csv", index=False)
assert o.parsed_value.notna().all() and (o.parsed_value > 0).all()
o.to_csv(f"submissions/sub_xt_{nm}_id1.csv", index=False)
print(f"{nm}: C4 rows {len(D)}, used {int(D.used.sum())}, changed vs clean4 {int(D.changed.sum())}, "
      f"blocked by a later clean4 tag {int((D.later_tags != '').sum())}, c4 == fusion {int(D.c4_eq_fusion.sum())}")
