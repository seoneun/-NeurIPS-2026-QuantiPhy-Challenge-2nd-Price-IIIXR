import sys, json, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
T = pd.read_pickle(f"{R}/analysis/v3/ag/rul346/tab.pkl"); T = T[T.split == "test"].set_index("id0")
B = pd.read_csv(f"{R}/analysis/v3/ag/r2_lab_main/r2_lab_latgeo_rows.csv").set_index("id0")
b21 = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv").parsed_value.values
B["b21"] = b21[B.index]; B["F"] = T.loc[B.index, "F"]; B["g"] = T.loc[B.index, "g"]; B["lat"] = T.loc[B.index, "lat"]
B["q"] = T.loc[B.index, "q"]; B["kind"] = T.loc[B.index, "kind"]
print("best21v == new on all block rows:", np.allclose(B.b21, B.new), " rows", len(B))
B["rec"] = B.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True)
B["qk"] = B.q.str.lower().str.replace(r"\s+", " ", regex=True)
B["right"] = FC.row_mra(B.new.values, B.new.values) - FC.row_mra(B.old.values, B.new.values)
B["wrong"] = FC.row_mra(B.new.values, B.old.values) - FC.row_mra(B.old.values, B.old.values)
# copies median: same recording, same question text
B["grp_med"] = B.groupby(["rec", "qk"]).new.transform("median"); B["ncopy"] = B.groupby(["rec", "qk"]).new.transform("size")
B["spread"] = B.groupby(["rec", "qk"]).new.transform(lambda s: s.max() / s.min())
s = B.groupby("rec").agg(n=("new", "size"), right=("right", "sum"), wrong=("wrong", "sum"), F=("F", "median"), lat=("lat", "median"))
print(s.round(2).to_string()); print("total right %.1f wrong %.1f" % (B.right.sum(), B.wrong.sum()))
pd.set_option("display.width", 250)
print(B[["video", "kind", "q", "old", "g", "F", "lat", "new", "grp_med", "spread", "right", "wrong"]].round(3).to_string())
B.to_pickle(f"{R}/analysis/v3/ag/rul346/blk.pkl")
