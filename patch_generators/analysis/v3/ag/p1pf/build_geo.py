"""P1 part 4: carry the re-measured P1 rows into the production geo builds, the way geo/personf_merge.py merges a person-focal
build into its reference.

For each row of a P1 video: the production final geo value must equal the reference re-measurement of the detector-track run
(m) or the VLM-track run (v) (checked to 1e-6; rows whose final geo came from elsewhere are left alone). The row then takes
that run's person-focal value and mode (3d_calib -> 3d_personf). personf_merge's two refinements are applied:
  (a) a 3d_calib row whose geo moves < 5% keeps its production value and strong mode;
  (c) a speed of the person whose height set the focal (target depth key = a person key) keeps its production value.
usage: python analysis/v3/ag/p1pf/build_geo.py <out_suffix> [copy]
  writes preds/{split}_geo_v4ntrj5sQ2<sfx>srngsr_d_vw(_raw).csv and the u build preds/{split}_geo_v4ntrj5sQ2<sfx>usrngsr_d_vw(_raw).csv;
  'copy' writes unchanged copies (to check that the renamed build reproduces the production submission)."""
import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/tools")
from qp_common import OUT_COLS  # noqa: E402
sfx = sys.argv[1]
copy = len(sys.argv) > 2 and sys.argv[2] == "copy"
PER = re.compile(r"person|people|\bman\b|woman|girl|boy|human|player|child|kid|lady|guy", re.I)
log = []
for split in ("val", "test"):
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        x = json.loads(ln); P[int(x["id"])] = x
    for u in ("", "u"):
        src = f"{R}/preds/{split}_geo_v4ntrj5sQ2{u}srngsr_d_vw_raw.csv"
        A = pd.read_csv(src).set_index("Unnamed: 0")
        o = A.copy()
        if split == "test" and not copy:
            M = {}
            for run in ("m", "v"):
                for md in ("ref", "new"):
                    M[run, md] = pd.read_csv(f"{R}/cache/tmp/ag_p1pf_meas_test_{run}{u}_{md}.csv").set_index("id")
            ids = M["m", "ref"].index
            for i in ids:
                g0 = float(A.geo_value.get(i, np.nan))
                hit = None
                for run in ("m", "v"):
                    vr = float(M[run, "ref"].value.get(i, np.nan))
                    if np.isfinite(g0) and np.isfinite(vr) and abs(vr / g0 - 1) < 1e-6:
                        hit = run; break
                if hit is None:
                    if np.isfinite(g0):
                        log.append((split, u, i, A.video_id[i], "unmatched: final geo not from a re-measured run", g0, np.nan, A.meas_mode[i]));
                    continue
                vn = float(M[hit, "new"].value.get(i, np.nan)); mn = str(M[hit, "new"]["mode"].get(i, ""))
                if not np.isfinite(vn) or abs(vn / g0 - 1) < 1e-9:
                    continue
                mode0 = str(A.meas_mode[i])
                q = (P.get(int(i)) or {}).get("q") or {}
                if mode0.replace("vlm_", "").startswith("3d_calib") and A.answer_source[i] == "meas" and 0.95 <= vn / g0 <= 1.05:
                    log.append((split, u, i, A.video_id[i], "(a) moved < 5%: keep", g0, vn, mode0)); continue
                if q.get("kind") in ("speed", "avgspeed", "hvel", "path", "disp") and any(PER.search(k) for k in ((P.get(int(i)) or {}).get("target_dkeys") or [])):
                    log.append((split, u, i, A.video_id[i], "(c) person's own motion: keep", g0, vn, mode0)); continue
                pre = "vlm_" if mode0.startswith("vlm_") else ""
                o.loc[i, "geo_value"] = vn
                o.loc[i, "meas_value"] = vn if "meas_value" in o else np.nan
                o.loc[i, "meas_mode"] = pre + mn
                o.loc[i, "meas_f"] = float(M[hit, "new"].f.get(i, np.nan)) if "f" in M[hit, "new"] else o.loc[i, "meas_f"]
                log.append((split, u, i, A.video_id[i], f"{hit}: {mode0} -> {pre + mn}", g0, vn, mode0))
            o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
        tag = f"geo_v4ntrj5sQ2{sfx}{u}srngsr_d_vw"
        o.to_csv(f"{R}/preds/{split}_{tag}_raw.csv")
        o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_{tag}.csv", index=False)
        ch = int((~np.isclose(pd.to_numeric(o.geo_value, errors="coerce"), pd.to_numeric(A.geo_value, errors="coerce"), rtol=1e-9, equal_nan=True)).sum())
        print(f"{split} {tag}: geo changed on {ch} rows")
L = pd.DataFrame(log, columns=["split", "u", "id", "vid", "how", "g0", "gnew", "mode0"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
if len(L):
    print(L[L.u == ""].to_string())
    print("u build differs from main on", int((L[L.u == "u"].set_index("id").gnew.reindex(L[L.u == ""].id).values != L[L.u == ""].gnew.values).sum()), "rows")
L.to_csv(f"{R}/cache/tmp/ag_p1pf_buildlog.csv", index=False)
