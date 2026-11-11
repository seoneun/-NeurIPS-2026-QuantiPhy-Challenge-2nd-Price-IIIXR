"""latgeo2 refinement options relative to best21v (= best18v + rul345 on its 84 rows)."""
import sys, json, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
W2 = f"{R}/analysis/v3/ag/rul346"; rng = np.random.default_rng(2)
B = pd.read_pickle(f"{W2}/blk.pkl")
def report(name, ids, new, cur):
    new, cur = np.asarray(new, float), np.asarray(cur, float)
    r = (FC.row_mra(new, new) - FC.row_mra(cur, new)).sum(); w = (FC.row_mra(new, cur) - FC.row_mra(cur, cur)).sum()
    out = f"{name}: rows {len(ids)}; bound if right {r:+.2f}, if wrong {w:+.2f} S3 row-MRA"
    for sg in (0.25,):
        gr = np.mean([(FC.row_mra(new, y) - FC.row_mra(cur, y)).sum() for y in new * np.exp(sg * rng.standard_normal((3000, len(new))))])
        gw = np.mean([(FC.row_mra(new, y) - FC.row_mra(cur, y)).sum() for y in cur * np.exp(sg * rng.standard_normal((3000, len(new))))])
        out += f"; sigma {sg}: if right {gr:+.2f}, if wrong {gw:+.2f}"
    print(out)
# A: copies median within block groups whose spread > 5 %
A = B[(B.spread > 1.05) & ~np.isclose(B.grp_med, B.new, rtol=0.005)]
print(A[["video", "q", "new", "grp_med", "spread"]].round(2).to_string())
report("A copies median (spread > 5 %)", A.index, A.grp_med, A.new)
# A2: all block rows set to the group median (one answer per question)
A2 = B[~np.isclose(B.grp_med, B.new, rtol=1e-9)]
report("A2 copies median (all groups)", A2.index, A2.grp_med, A2.new)
# B: 0033 rows back to best18v (depth_info not the shown camera's)
Bq = B[B.rec == "captured_0033"]
report("B 0033 revert to best18v", Bq.index, Bq.old, Bq.new)
Bq2 = Bq[Bq.q.str.contains("0.5s")]
report("B2 0033 0.5 s rows: copies median 307.7 (only 2554 changes)", Bq2.index[:1], [Bq2.grp_med.iat[0]], [Bq2.new.iat[0]])
# C: the 12 strict extension rows (geo-led patch values without the factor) -> rule geo x F
E = pd.read_csv(f"{W2}/ext_rows.csv")
report("C extension (12 rows)", E.id0, E.new, E.b21)
for rec, sub in E.groupby(E.video.str[:13]): report(f"   C {rec}", sub.id0, sub.new, sub.b21)
