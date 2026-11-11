import sys, json
import numpy as np, pandas as pd
DET, CUR, GEO, QM, QV, PARSE, OUT = sys.argv[1:8]
BAND = np.log(2.0)
det = pd.read_csv(DET, index_col=0)
G = pd.read_csv(GEO, index_col=0)
Q = {"m": pd.read_csv(QM).set_index("id"), "v": pd.read_csv(QV).set_index("id")}
P = {r["id"]: r for r in map(json.loads, open(PARSE))}
cur = pd.read_csv(CUR).parsed_value.values
out = {}
for pos, (i, r) in enumerate(det.iterrows()):
    i = int(i); x = P.get(i) or {}
    if x.get("cat") != "S2" or not str(r.video_id).startswith("internet") or (x.get("q") or {}).get("kind") != "size":
        continue
    if str(r.answer_source) != "dis_size_gm":
        continue
    a, g, cs = float(cur[pos]), float(r.g), float(r.cs)
    if not np.isclose(a, float(r.answer), rtol=1e-9):
        continue
    gg = float(G.geo_value.get(i, np.nan))
    if not (np.isfinite(gg) and np.isclose(gg, g, rtol=1e-6) and cs > 0):
        continue
    ch = next((c for c in "mv" if np.isclose(g, Q[c].value.get(i, np.nan), rtol=1e-6)), None)
    if ch is None:
        continue
    o = "v" if ch == "m" else "m"
    g2, ok2 = float(Q[o].value.get(i, np.nan)), bool(Q[o].ok.get(i, False))
    if not (ok2 and np.isfinite(g2) and g2 > 0):
        continue
    if not (abs(np.log(g / cs)) > BAND and abs(np.log(g2 / cs)) <= BAND):
        continue
    out[str(i)] = dict(value=g2)
json.dump(out, open(OUT, "w"), indent=0)
