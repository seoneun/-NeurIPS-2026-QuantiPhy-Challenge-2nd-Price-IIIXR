import sys, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3"); sys.path.insert(0, R + "/analysis/v3/ag/rul204")
import analyze_caw as A
from lab_rows import labelled
import rul204 as RU
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
L = labelled(RU.SUB)
for sp in ("val", "test"):
    P = json.load(open(f"{R}/cache/tmp/ag_c95gamsim_s2sim_patch_{sp}.json"))
    Ls = L[L.split == sp]
    hit = [int(k) for k in P if int(k) in Ls.index]
    print(sp, "patched", len(P), "labelled changed", len(hit))
    for k in hit:
        r = Ls.loc[k]; nv = P[str(k)]["value"]
        print(k, r.video_id, r.y, r.answer, nv, r.m, A.row_mra(np.array([nv]), np.array([r.y]))[0])
# looser diagnostics: all sim S2 size groups with labelled rows; factor, iqr, hedge vs geo MRA on labelled hedge rows w/ 2d_ratio
rows = []
for sp in ("val", "test"):
    D = RU.load(sp)
    Ls = L[L.split == sp]
    for vid, z in D[D.video_id.astype(str).str.startswith("simulation") & (D.cat == "S2")].groupby("video_id"):
        st = RU.group_stats(z)
        lz = Ls[Ls.video_id == vid]
        if st is None or not len(lz): continue
        lz = lz.copy(); lz["mm"] = D.mm.reindex(lz.index)
        h = lz[lz.answer_source.isin(RU.HEDGE) & (lz.mm == "2d_ratio")]
        if not len(h): continue
        lr = np.log(h.g / h.cs)
        same = h[(np.sign(lr) == np.sign(st[0])) & (abs(lr - st[0]) <= np.log(1.35)) & (abs(lr) >= np.log(1.2))]
        rows.append(dict(sp=sp, vid=vid, pk=z.pk.iloc[0], fac=np.exp(st[0]), iqr=st[1], n=st[2], nh=len(h), hm=h.m.mean(), gm=h.mg.mean(),
                         ns=len(same), hms=same.m.mean() if len(same) else np.nan, gms=same.mg.mean() if len(same) else np.nan))
T = pd.DataFrame(rows)
print(T.round(3).to_string())
for lab, s in [("all", T), ("fac>=1.4|<=1/1.4", T[abs(np.log(T.fac)) >= np.log(1.4)]), ("fac1.2+", T[abs(np.log(T.fac)) >= np.log(1.2)]), ("size", T[T.pk == "size"])]:
    if len(s):
        w = s.ns
        print(lab, "groups", len(s), "rows(same-sign)", int(w.sum()), "hedge", round((s.hms * w).sum() / max(w.sum(), 1), 3), "geo", round((s.gms * w).sum() / max(w.sum(), 1), 3))
