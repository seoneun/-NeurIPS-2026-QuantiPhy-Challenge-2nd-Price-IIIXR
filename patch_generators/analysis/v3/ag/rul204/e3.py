import sys
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3"); sys.path.insert(0, R + "/analysis/v3/ag/rul204")
import analyze_caw as A
from lab_rows import labelled
import rul204 as RU
L = labelled(RU.SUB)
out = []
for sp in ("val", "test"):
    D = RU.load(sp); Ls = L[L.split == sp].copy()
    Ls["mm"] = D.mm.reindex(Ls.index); Ls["pk"] = D.pk.reindex(Ls.index); Ls["qk"] = D.qkind.reindex(Ls.index)
    out.append(Ls)
Z = pd.concat(out)
Z = Z[Z.video_id.astype(str).str.startswith("simulation") & Z.answer_source.isin(RU.HEDGE) & (Z.mm == "2d_ratio") & (Z.qk != "accel")]
Z["lr"] = np.log(Z.g / Z.cs)
for cat in ("S2", "D2", "S3", "D3"):
    for pk in ("size", "speed"):
        s = Z[(Z.cat == cat) & (Z.pk == pk)]
        for lo in (1.2, 1.4):
            t = s[abs(s.lr) >= np.log(lo)]
            if len(t): print(cat, pk, ">=%.1f" % lo, "n", len(t), "hedge %.3f geo %.3f caw %.3f" % (t.m.mean(), t.mg.mean(), t.mc.mean()))
s = Z[(Z.cat == "S2") & (abs(Z.lr) >= np.log(1.2))]
print(s[["split", "video_id", "pk", "y", "answer", "g", "cs", "m", "mg", "mc"]].round(3).to_string())
