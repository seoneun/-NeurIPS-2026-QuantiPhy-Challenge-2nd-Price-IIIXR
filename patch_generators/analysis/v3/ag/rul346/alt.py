"""latgeo2: expected rul345 gain (S3 row-MRA, 84 rows, block value vs best18v value) when the truth is centred on
different values with lognormal noise sigma (3000 draws). Output alt.out. Used to read the portal +2.9."""
import sys, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
B = pd.read_pickle(f"{R}/analysis/v3/ag/rul346/blk.pkl"); rng = np.random.default_rng(3)
def exp_gain(center, sg, n=3000):
    ys = center * np.exp(sg * rng.standard_normal((n, len(B))))
    g = np.array([(FC.row_mra(B.new.values, y) - FC.row_mra(B.old.values, y)).sum() for y in ys]); return g.mean(), g.std()
for name, cen in (("truth ~ g x F (block right)", B.new.values), ("truth ~ g alone (no factor)", B.g.values), ("truth ~ old (fusion)", B.old.values),
                  ("truth ~ g x F^1.5 (factor 1.3^1.5=1.48 equiv.)", B.g.values * B.F.values ** 1.5), ("truth ~ sqrt(gF * old)", np.sqrt(B.new.values * B.old.values))):
    for sg in (0.2, 0.25, 0.3):
        m, s = exp_gain(cen, sg); print(f"{name:48s} sigma {sg}: expected block gain {m:+.1f} (sd {s:.1f})")
lat = B.F >= 1.15
for sub, nm in ((lat, "F>=1.15"), (~lat, "F<1.15")):
    for cen_name, cen in (("gF", B.new.values), ("g", B.g.values), ("old", B.old.values)):
        y = cen[sub.values] * np.exp(0.25 * rng.standard_normal((3000, sub.sum())))
        g = np.array([(FC.row_mra(B.new.values[sub.values], yy) - FC.row_mra(B.old.values[sub.values], yy)).sum() for yy in y])
        print(f"   subset {nm}: truth ~ {cen_name}: expected {g.mean():+.1f}")
