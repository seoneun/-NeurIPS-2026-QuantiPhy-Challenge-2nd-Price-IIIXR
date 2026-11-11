"""rul144: speed at time t inside a rul016-fitted flight (coverage gap of the accepted rul016).

Gravity prior, 2D, q.kind == speed with q.t set, question not matched by rul016's phrasing branches.  The target track's
airborne segment is fitted exactly as rul016 (air_fit: cy quadratic, cx linear, takeoff = vertex).  With the same gates
(r2 > 0.995, vres < 0.15, g/a within 1.3x of geo's meas_mpp) and te <= t <= tl, the speed at t is
    v = hypot(|vx| * mpp, g * (t - te)),  mpp = g / a   (g = the INPUT gravity prior value).
Final answer through rul016's fusion emulation (agree within 2x -> gm etc.; strong-mode disagreement -> this value;
weak-mode disagreement / non-reproducible fusion -> no change)."""
import json, sys, math, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/c21s3")
sys.path.insert(0, R + "/analysis/v3/ag/rul016")
from trk import load, series
from c21air_final import air_fit, fuse, PH

DET = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz24"


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")
    det = pd.read_csv(f"{R}/submissions/{DET}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out, rep = {}, []
    for i, r in raw.iterrows():
        rid = int(r["Unnamed: 0"]); q = str(r.question)
        p = P.get(rid) or {}; pq = p.get("q") or {}; pri = (p.get("priors") or [{}])[0]
        if not pri.get("gravity") or "3d" in str(r.meas_mode): continue
        if pq.get("kind") != "speed" or pq.get("t") is None: continue
        if PH.search(q): rep.append((rid, "c21air_phrase")); continue
        g = float(pri.get("value_si") or 9.8)
        t = float(pq["t"])
        d, j = load(r.video_id, split)
        if j is None: rep.append((rid, "notrack")); continue
        S = series(j); tgt = (p.get("target") or {}).get("phrase") or pq.get("target")
        key = next((k for k in S if k.split("|")[0] == tgt), None)
        if key is None: rep.append((rid, "notarget", tgt)); continue
        T, X, Y, W, H, B, o = S[key]
        f = air_fit(T, X, Y)
        if f is None: rep.append((rid, r.video_id, "nofit")); continue
        mpp = g / f["a"]; gm = float(r.meas_mpp) if pd.notna(r.meas_mpp) else np.nan
        ok = f["r2"] > 0.995 and f["vres"] < 0.15 and (np.isnan(gm) or 1 / 1.3 < mpp / gm < 1.3)
        inwin = f["te"] <= t <= f["tl"]
        vxs = abs(f["vx"]) * mpp; vy = g * (t - f["te"])
        v = math.hypot(vxs, vy) * {"cm/s": 100.0, "mm/s": 1000.0}.get(str(pq.get("unit") or "m/s"), 1.0)
        extra = ("csv_fps", p.get("fps"), "trk_fps", j["fps"], "g", g, "t", t, "te", round(f["te"], 3), "tl", round(f["tl"], 3),
                 "n", f["n"], "r2", round(f["r2"], 5), "vres", round(f["vres"], 3), "mpp", round(mpp, 5), "geo_mpp", round(gm, 5),
                 "vx", round(vxs, 4), "vy", round(vy, 4), "ok", ok, "inwin", inwin)
        if abs(float(j["fps"]) - float(p.get("fps") or j["fps"])) > 1e-6: rep.append((rid, "fps_mismatch") + extra); continue
        if not ok or not inwin: rep.append((rid, r.video_id, "gate_fail", round(v, 4)) + extra); continue
        dr = det.loc[rid]
        cs, mode, cat = float(dr.cs), str(dr["mode"]), str(dr["cat"])
        w3 = bool(dr.w3) if ("w3" in dr and dr.w3 == dr.w3) else False
        old_g = float(dr.g) if dr.g == dr.g else np.nan
        chk, chk_s = fuse(old_g, cs, mode, w3, cat, "speed") if old_g == old_g else (None, "nogeo")
        same = chk is not None and abs(chk - float(dr.answer)) < 1e-6 * max(1, abs(dr.answer))
        fin, src = fuse(float(v), cs, mode, w3, cat, "speed")
        rep.append((rid, r.video_id, cat, q, "old", round(float(dr.answer), 4), dr.answer_source, "old_g", round(old_g, 4),
                    "cs", round(cs, 4), "mode", mode, "new_v", round(v, 4), "final", None if fin is None else round(fin, 4), src,
                    "fusion_reproduces_old", same, chk_s) + extra)
        if fin is None: continue
        out[str(rid)] = {"value": float(fin), "why": f"rul144: in flight te={f['te']:.3f}<=t={t}<=tl={f['tl']:.3f}; "
                         f"hypot(|vx| g/a={vxs:.3f}, g(t-te)={vy:.3f})={v:.3f}; fusion {src} with CaW {cs:.3g}; old fusion reproduced={same}"}
    json.dump(out, open(f"{R}/cache/tmp/ag_c79airspd_patch_{split}.json", "w"), indent=1)
    print(split, "patched", len(out))
    for x in rep: print("  ", x)


if __name__ == "__main__":
    for s in ("test", "val"): run(s)
