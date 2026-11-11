#!/usr/bin/env python
"""rul019: radial lower bound on a lab ball's speed from rest under the stated acceleration.
Rows (computed from parse / tracks / prior text only):
  * captured_* D3 speed rows, unit m/s / cm/s / mm/s;
  * exactly one non-gravity accel prior a whose object is the target phrase (rul438 same_obj);
  * the target has exactly one own depth key K (key name == target phrase) with >= 2 timed entries.
Onset: the target's e-track (tracks v2sc -> v2 -> v2repro), rul438 fit_onset start window: >= 8 rest frames, sqrt(d) linear over
  frames i_m..i_m+4, k0 = -b/alpha (taken even when fit_onset later fails, e.g. short_fit); t_on = k0 / fps (dataset csv fps).
Keys: d_r = latest own key with t <= t_on - 1/fps (at rest);
  (i)  speed at t:            d_t = own key with |t_k - t| <= 1/fps;
  (ii) when='end' + 'slope':  d_t = slope end key (slope_* with near/end/tower/far) in the motion direction sign(d_last - d_r).
Bound: from rest under constant a, v(t)^2 = 2 a s(t) with s the path length >= |d_t - d_r| (radial change <= path): v >= v_lb = sqrt(2 a |d_t - d_r|).
Gate: latest own timed key (t_L, d_L): sqrt(2 a |d_L - d_r|) / (a (t_L - t_on)) in [0.8, 1.25] (the keys and the onset tell one story).
Fire: answer < 0.85 v_lb -> 1.05 v_lb (question unit).
Writes cache/tmp/ag_c21vlb_patch_{test,val}.json and cache/tmp/ag_c21vlb_rows.csv.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul019/rule.py [sub]"""
import sys, json, os, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3/ag/rul438"); sys.path.insert(0, f"{R}/analysis/v3")
from onset import load_track, centres
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzn"
FUSE = os.environ.get("C21VLB_FUSE", "0") == "1"
UNIT = {"m/s": 1.0, "cm/s": 100.0, "mm/s": 1000.0}
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 500)


def norm(p):
    return " ".join(str(p or "").lower().replace("_", " ").replace("the ", " ").split())


def same_obj(a, b):
    a, b = norm(a), norm(b)
    return bool(a and b and (a == b or a.endswith(" " + b) or b.endswith(" " + a)))


def start_window(arr, min_rest=8):
    """rul438 fit_onset up to the start-window fit; returns (k0, alpha, i_m, status)."""
    if len(arr) < min_rest + 5:
        return None, "short"
    k = arr[:, 0]; c = arr[:, 1:3]
    if np.any(np.diff(k) != 1):
        return None, "gaps"
    base = np.median(c[:5], axis=0)
    d = np.hypot(*(c - base).T)
    noise = max(1.0, float(np.percentile(d[:min_rest], 90)))
    thr = max(3.0, 3.0 * noise)
    i_m = None
    for i in range(min_rest, len(d) - 4):
        w_ = d[i:i + 4]
        if np.all(w_ > thr) and np.all(np.diff(w_) > -1.0) and np.all(d[:i] <= thr):
            i_m = i; break
    if i_m is None:
        return None, "no_onset"
    x = k[i_m:i_m + 5]; y = np.sqrt(d[i_m:i_m + 5])
    A_ = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
    if coef[0] <= 0:
        return None, "neg_slope"
    bad = bool(np.max(np.abs(coef[0] * x + coef[1] - y)) > max(0.3, 0.06 * y.max()))
    return dict(k0=float(-coef[1] / coef[0]), alpha=float(coef[0]), i_m=int(i_m), k_m=int(k[i_m]), bad_start=bad), "ok"


def main():
    rows = []
    for split, suf in (("test", "detail"), ("val", "valdetail")):
        det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
        vinfo = {}
        p_vi = f"{R}/data/{split}_video_info.csv"
        if os.path.exists(p_vi):
            vi = pd.read_csv(p_vi)
            vinfo = {str(r.vid): float(r.fps_meta) for r in vi.itertuples()}
        out = {}
        for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
            j = json.loads(l); i = int(j["id"]); vid = str(j["video_id"])
            if not vid.startswith("captured_") or j.get("cat") != "D3" or i not in det.index:
                continue
            q = j["q"]; unit = str(q.get("unit") or "").strip()
            if q.get("kind") != "speed" or unit not in UNIT:
                continue
            pr = [p for p in (j.get("priors") or []) if p.get("kind") == "accel" and not p.get("gravity") and p.get("value_si")]
            if len(pr) != 1:
                continue
            tgt = (j.get("target") or {}).get("phrase") or q.get("target")
            pob = (j.get("prior_obj") or {}).get("phrase") or pr[0].get("obj")
            if not same_obj(tgt, pob):
                continue
            a = float(pr[0]["value_si"])
            dep = j.get("depth") or []
            own = {}
            for e in dep:
                if same_obj(e.get("key"), tgt) and e.get("t") is not None and e.get("d") is not None:
                    own.setdefault(e["key"], []).append((float(e["t"]), float(e["d"])))
            own2 = {kk: sorted(v) for kk, v in own.items() if len(v) >= 2}
            r = det.loc[i]
            rec = dict(split=split, id=i, video_id=vid, q=str(j["question"])[:62], old=float(r["answer"]), src=str(r["answer_source"]),
                       g=r.get("g"), cs=r.get("cs"), a=a, unit=unit)
            if "y" in det.columns:
                rec["y"] = r["y"]
            rows.append(rec)
            if len(own2) != 1:
                rec["why"] = f"own keys {list(own)}"; continue
            K, ent = next(iter(own2.items()))
            fps = float(j.get("fps") or vinfo.get(vid) or 0)
            if vinfo.get(vid) and abs(vinfo[vid] - fps) > 1e-6:
                rec["fps_mismatch"] = vinfo[vid]
            tr, tv = load_track(split, vid, ("v2sc", "v2", "v2repro"))
            if tr is None or not fps:
                rec["why"] = "no track/fps"; continue
            ek = [e.get("obj") for e in tr.get("queries", []) if norm(str(e.get("key", "")).split("|")[0]) == norm(tgt)]
            ek = [o for o in ek if o is not None]
            if not ek:
                rec["why"] = f"{tv} no e-track"; continue
            sw, st = start_window(centres(tr, ek[0]))
            if sw is None:
                rec["why"] = f"{tv} e{ek[0]} {st}"; continue
            t_on = sw["k0"] / fps; dt = 1.0 / fps
            rec.update(tv=tv, k0=round(sw["k0"], 2), t_on=round(t_on, 3), bad_start=sw["bad_start"])
            rest = [(t, d) for t, d in ent if t <= t_on - dt]
            if not rest:
                rec["why"] = "no rest key"; continue
            t_r, d_r = rest[-1]
            t_L, d_L = ent[-1]
            if t_L <= t_on + dt:
                rec["why"] = "last key before onset"; continue
            ratio = np.sqrt(2 * a * abs(d_L - d_r)) / (a * (t_L - t_on))
            rec["gate"] = round(float(ratio), 3)
            d_t = None
            if q.get("t") is not None:
                tq = float(q["t"])
                near = [(abs(t - tq), d) for t, d in ent if abs(t - tq) <= dt + 1e-9]
                if near:
                    d_t = min(near)[1]; rec["branch"] = "i"
            elif q.get("when") == "end" and "slope" in str(j["question"]).lower():
                sgn = np.sign(d_L - d_r)
                ends = [(e["key"], float(e["d"])) for e in dep if e.get("d") is not None and str(e.get("key", "")).lower().startswith("slope")
                        and re.search(r"near|end|tower|far", str(e["key"]).lower()) and np.sign(float(e["d"]) - d_r) == sgn and sgn != 0]
                if ends:
                    kk, d_t = max(ends, key=lambda x: abs(x[1] - d_r)); rec["branch"] = f"ii:{kk}"
                    rec["n_ends"] = len(ends)
            if d_t is None:
                rec["why"] = "no d_t key"; continue
            v_lb = float(np.sqrt(2 * a * abs(d_t - d_r)))
            rec.update(d_r=d_r, d_t=d_t, v_lb=round(v_lb * UNIT[unit], 3), at=(round(a * (float(q["t"]) - t_on) * UNIT[unit], 3) if q.get("t") is not None else np.nan))
            if not (0.8 <= ratio <= 1.25):
                rec["why"] = f"gate {ratio:.3f}"; continue
            if rec["old"] >= 0.85 * v_lb * UNIT[unit]:
                rec["why"] = "above bound"; continue
            new = 1.05 * v_lb * UNIT[unit]
            why = f"rul019: v >= sqrt(2a|d_t-d_r|) = {v_lb * UNIT[unit]:.1f} {unit} (a {a}, d {d_r}->{d_t}, t_on {t_on:.3f}, gate {ratio:.2f})"
            cs = r.get("cs")
            if FUSE and cs is not None and np.isfinite(cs) and cs > 0 and max(cs / new, new / cs) <= 2.0:
                new = float(np.sqrt(new * cs)); why += f" | fused gm with CaW {cs:.1f}"
            rec["new"] = new; rec["why"] = "FIRE"
            out[str(i)] = {"value": float(new), "why": why}
        tag = "rul019" + ("F" if FUSE else "")
        json.dump(out, open(f"{R}/cache/tmp/ag_{tag}_patch_{split}.json", "w"), indent=1)
        print(f"[{split}] patch rows {len(out)} -> cache/tmp/ag_{tag}_patch_{split}.json")
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/cache/tmp/ag_c21vlb_rows.csv", index=False)
    print(df.drop(columns=["q"]).to_string())
    print(df[["id", "video_id", "q"]].to_string())


if __name__ == "__main__":
    main()
