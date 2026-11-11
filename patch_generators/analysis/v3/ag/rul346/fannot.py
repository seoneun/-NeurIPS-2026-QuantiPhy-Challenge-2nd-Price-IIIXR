import sys, re, json, pickle, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
W2 = f"{R}/analysis/v3/ag/rul346"
T = pd.read_pickle(f"{W2}/tab.pkl")
t = pd.read_parquet(f"{R}/data/test/test_dataset.parquet"); v = pd.read_csv(f"{R}/data/val/validation_dataset.csv")
pri = {**{("test", i): p for i, p in enumerate(t.prior)}, **{("val", i): p for i, p in enumerate(v.ground_truth_prior)}}
T["prior"] = [str(pri.get((s, i), "")) for s, i in zip(T.split, T.id0)]
T["pk"] = np.where(T.prior.str.contains("veloc|speed|m/s|cm/s", case=False), "vel", np.where(T.prior.str.contains("accel", case=False), "acc",
            np.where(T.prior.str.contains("distance|displacement", case=False), "dist", "size")))
T["rec"] = T.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True)
c = T[T["mode"].astype(str).str.startswith("3d_calib") & ~T["mode"].astype(str).str.contains("rejected")]
g = c.groupby(["rec", "pk"]).agg(n=("id0", "size"), cats=("cat", lambda s: ",".join(sorted(set(s)))), fW=("fW", "median"), fWmin=("fW", "min"), fWmax=("fW", "max"))
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print("calibrated focal f/W per recording and prior kind (spec 0.596; spec/1.3 = 0.458)"); print(g.round(3).to_string())
dyn = g.reset_index(); dyn = dyn[dyn.pk != "size"]
print("motion-prior recordings: n", len(dyn), "median f/W %.3f, IQR %.3f-%.3f" % (dyn.fW.median(), dyn.fW.quantile(.25), dyn.fW.quantile(.75)))
sz = g.reset_index(); sz = sz[sz.pk == "size"]
print("size-prior recordings: n", len(sz), "median f/W %.3f, IQR %.3f-%.3f" % (sz.fW.median(), sz.fW.quantile(.25), sz.fW.quantile(.75)))
