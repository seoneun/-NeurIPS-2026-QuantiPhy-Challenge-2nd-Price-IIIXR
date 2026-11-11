import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, f"{R}/analysis/v3")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
raw = pd.read_csv(f"{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
for i in (1221, 1135):
    rw = raw.loc[i]; MV.TRACK_VER = str(rw.meas_tracks)
    byv, vjs = MV.load_split_tracks("test", "e")
    rec = [x for x in byv[rw.video_id] if x["id"] == i][0]; vj = vjs[rw.video_id]
    V = MV.Video(vj, "e"); tr = V.get(rec["target"])
    W, H = vj["W"], vj["H"]; f = float(rw.meas_f) * W
    e, js = tr.ext_px("length", "orange car")
    print(i, rw.video_id, "W,H", W, H, "f", f, "nfr", len(tr.st), "okc", len(tr.okc), "oke", len(tr.oke))
    for j in range(len(tr.st)):
        s = tr.st[j]
        if not s: continue
        tag = ("E" if j in list(js) else "") + ("C" if j in list(tr.okc) else "")
        print(f"  j{j} t{tr.t_all[j]:.2f} box {[round(x) for x in s['box']]} bord {s.get('border')} {tag}")
