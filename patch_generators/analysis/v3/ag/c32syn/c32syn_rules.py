"""c32 synthesis builders (label-free; labels only evaluate).
rul046: internet video, a parsed prior of kind 'accel' (gravity or stated), detail kind in {speed, disp, avgspeed},
  raw meas_mode == '2d_ratio', meas_why empty (the accel fit succeeded; excludes prior_no_freefall fallback rulers),
  answer_source in {agree_gm, agree_wgm}, geo finite -> answer = geo.
rul047: c32ngprop (analysis/v3/ag/c32i/c32rules.py) with the speed estimator replaced by the derivative of the
  free-fall fit itself: v = mpp * hypot(vx_lin, a_px (t - t_vertex)) when t_vertex < t <= last fitted frame; disp / gap
  rows keep c32ngprop's values. The +-1 frame central difference is biased by the clip's 60->50 pulldown.
"""
import sys, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/c32i")
KIN = {"speed", "disp", "avgspeed"}


def P(split):
    d = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); d[int(j["id"])] = j
    return d


def ikin(split):
    det = pd.read_csv(f"{R}/submissions/sub_xt_Nzt_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    PP = P(split); out = {}
    for i, r in det.iterrows():
        j = PP.get(int(i))
        if j is None or j.get("video_source") != "internet":
            continue
        if not any(str(p.get("kind")) == "accel" for p in (j.get("priors") or [])):
            continue
        kind = r["kind"] if "kind" in det.columns else (j.get("q") or {}).get("kind")
        if kind not in KIN or r.answer_source not in ("agree_gm", "agree_wgm"):
            continue
        if str(raw.loc[i, "meas_mode"]) != "2d_ratio" or pd.notna(raw.loc[i, "meas_why"]):
            continue
        g = float(r.g)
        if not np.isfinite(g) or g <= 0:
            continue
        out[str(i)] = {"value": g, "why": f"rul046: internet accel-prior {kind}, clean 2d_ratio fit; drop {r.answer_source} hedge with CaW {r.cs:.4g} (was {r.answer:.4g}) -> geo {g:.4g}"}
    return out


def ngpropf(split):
    import c32rules as C
    a, _ = C.run(split)
    PP = P(split)
    for k in list(a):
        j = PP[int(k)]; q = j["q"]
        if q.get("kind") != "speed" or q.get("t") is None:
            continue
        A = C.load_track(split, j["video_id"], q.get("target"))
        f = C.ffit(A)
        t = A[:, 0]; cx = (A[:, 1] + A[:, 3]) / 2
        tq = float(q["t"])
        if not (f[2] < tq <= t[-1]):
            continue
        mpp = 9.8 / f[0]
        seg = t >= f[2]
        vx = np.polyfit(t[seg], cx[seg], 1)[0] if seg.sum() >= 3 else 0.0
        v = float(np.hypot(vx, f[0] * (tq - f[2])) * mpp) * C.UNIT.get(str(q.get("unit")).lower(), 1.0)
        a[k] = {"value": v, "why": f"rul047: free-fall fit derivative a={f[0]:.0f}px/s^2 vertex {f[2]:.3f}s mpp {mpp:.5f} -> {v:.4g} (c32ngprop cd-mix {a[k]['value']:.4g})"}
    for k in a:
        a[k]["why"] = a[k]["why"].replace("c32ngprop:", "rul047:")
    return a


if __name__ == "__main__":
    for split in ("test", "val"):
        o1 = ikin(split); o2 = ngpropf(split)
        o1 = {k: v for k, v in o1.items() if k not in o2}   # ngpropf owns the fallback-ruled rows
        json.dump(o1, open(f"{R}/cache/tmp/ag_c32ikin_patch_{split}.json", "w"), indent=0)
        json.dump(o2, open(f"{R}/cache/tmp/ag_c32ngpropf_patch_{split}.json", "w"), indent=0)
        print(split, "ikin", len(o1), "ngpropf", len(o2))
        for k, v in {**o1, **o2}.items():
            print(" ", k, round(v["value"], 4), v["why"][:200])
