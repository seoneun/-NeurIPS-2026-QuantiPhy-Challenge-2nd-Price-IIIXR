import json, glob, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
conf, low = [], []
for p in glob.glob(f"{R}/geo/out/tracks_test_v2/*.json"):
    j = json.load(open(p))
    for q in j["queries"]:
        if q.get("anchor") is None:
            continue
        (conf if q["best_gd"] >= 0.4 else low).append((q["best_gd"], q["best_owl"]))
conf, low = np.array(conf), np.array(low)
T = float(np.percentile(conf[:, 1], 10))
print("confident queries", len(conf), "OWL support quantiles (5,10,25,50):", np.round(np.percentile(conf[:, 1], [5, 10, 25, 50]), 3).tolist())
print("queries with G-DINO < 0.4:", len(low), "share of those with 0.15<=gd<0.3 and owl>=T:",
      round(float(((low[:, 0] >= 0.15) & (low[:, 0] < 0.3) & (low[:, 1] >= T)).mean()), 3) if len(low) else None)
json.dump(dict(T_owl=T, n_conf=int(len(conf)), rule="10th percentile of OWLv2 support on queries with G-DINO >= 0.4 (test tracks)"),
          open(f"{R}/geo/out/owl_threshold_v2.json", "w"))
print("T_owl", round(T, 3))
