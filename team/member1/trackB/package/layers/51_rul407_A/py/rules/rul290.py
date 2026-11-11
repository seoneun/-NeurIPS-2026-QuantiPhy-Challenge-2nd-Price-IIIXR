#!/usr/bin/env python
import json, os, sys, math, importlib.util
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
R = os.path.abspath(os.environ["RA_Q"]); WO = os.path.abspath(os.environ["RA_OUT"])
BASE = sys.argv[1]; GT = sys.argv[2]
TAG = "rul290"
_av = sys.argv; sys.argv = [_av[0], BASE]
_sp = importlib.util.spec_from_file_location("rul289_orig", f"{R}/analysis/v3/ag/rul289/rul289.py")
F = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(F); sys.argv = _av
MV = F.MV


def own_shift(q, vj):
    if not vj or not vj.get("fidx"):
        return 0.0
    dur = (max(vj["fidx"]) + 1) / float(vj["fps"])
    ts = [float(q[k]) for k in ("t", "t1", "t2") if q.get(k) is not None]
    if not ts or max(ts) <= dur + 0.5 or min(ts) - dur < 0 or max(ts) - dur > dur:
        return 0.0
    return dur


def run(split):
    det = pd.read_csv(f"{R}/submissions/{BASE}_detail.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv", index_col=0)
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    rows, vjs, vids = [], {}, {}
    sel = [r for r in recs if r["q"].get("kind") == "accel" and r["id"] in det.index and r["id"] in raw.index
           and str(raw.video_source[r["id"]]).lower().startswith(("simulation", "segment"))]
    for r in sel:
        i = r["id"]; D = det.loc[i]; q = r["q"]
        pr = r["priors"][0] if r["priors"] else None
        rec = dict(id=i, split=split, vid=r["video_id"], cat=D["cat"], unit=q.get("unit"), asrc=D.answer_source, mode=D.get("mode", ""),
                   ans=float(D.answer), g=float(D.g), cs=float(D.cs), geo_raw=float(raw.geo_value[i]) if pd.notna(raw.geo_value[i]) else np.nan,
                   fps=float(raw.fps[i]))
        rows.append(rec)
        if str(D.answer_source).startswith("patch"):
            rec["why"] = "already_patched"; continue
        if q.get("unit") not in F.SI:
            rec["why"] = "unit"; continue
        gval, gwhy = F.g_from_prior(pr, raw.ground_truth_prior[i], q.get("target"))
        rec["gsrc"] = gwhy
        if gval is None:
            rec["why"] = gwhy; continue
        rec["g_si"] = gval
        if r["video_id"] not in vjs:
            vjs[r["video_id"]] = F.load_vj(split, r["video_id"])
        vj, tdir = vjs[r["video_id"]]
        if vj is None:
            rec["why"] = "no_tracks"; continue
        rec["tdir"] = tdir
        if r["video_id"] not in vids:
            vids[r["video_id"]] = MV.Video(vj, "e")
        tr = vids[r["video_id"]].get(r["target"])
        if tr is None:
            rec["why"] = "no_target_track"; continue
        fps = float(vj["fps"]); rec["fps_trk"] = fps
        if abs(fps - rec["fps"]) > 1e-6:
            rec["why"] = "fps_mismatch"; continue
        sh = own_shift(q, vj); rec["tshift"] = sh
        if q.get("t1") is not None and q.get("t2") is not None:
            t_lo, t_hi = float(q["t1"]) - sh, float(q["t2"]) - sh
        elif q.get("t") is not None:
            t_lo = t_hi = float(q["t"]) - sh
        else:
            rec["why"] = "no_time"; continue
        Fi = np.array(vj["fidx"], float)
        jj = [int(j) for j in tr.okc]
        if len(jj) < F.MIN_N:
            rec["why"] = "short_track"; continue
        fr = Fi[jj]; box = np.array([tr.st[j]["box"] for j in jj], float)
        cx = 0.5 * (box[:, 0] + box[:, 2]); bot = box[:, 3]
        a_lo, a_hi = t_lo * fps, t_hi * fps
        rec["a_lo"], rec["a_hi"] = a_lo, a_hi
        S, why = F.segment(fr, cx, bot, a_lo, a_hi)
        if S is None:
            rec["why"] = why; continue
        mem = S["mem"]
        rec.update(S0=fr[mem[0]], S1=fr[mem[-1]], nS=len(mem))
        if len(mem) < 4:
            rec["why"] = "fail:nS"; continue
        x = fr[mem]; yb = bot[mem]; xc = cx[mem]
        cy, r2, keep = F.rfit(x, yb); ccx, _, _ = F.rfit(x, xc)
        a_y, a_x = 2 * cy[0], 2 * ccx[0]
        v0 = float(np.polyval(np.polyder(cy), x[0]))
        rec.update(a_px=a_y, a_x=a_x, r2=r2, v0=v0)
        geo_si = rec["geo_raw"] * F.SI[q["unit"]] if np.isfinite(rec["geo_raw"]) else rec["g"] * F.SI[q["unit"]]
        rec["geo_g"] = geo_si / gval if np.isfinite(geo_si) else np.nan
        fails = []
        if len(mem) < F.MIN_N: fails.append("nS")
        if not (r2 >= F.R2_MIN): fails.append("r2")
        if not (a_y > 0): fails.append("ay")
        if not (abs(a_x) <= F.AX_FRAC * abs(a_y)): fails.append("ax")
        if not (v0 <= F.V0_MAX): fails.append("v0")
        if not (a_hi <= fr[mem[-1]] + F.END_MARGIN): fails.append("end")
        if not (np.isfinite(rec["geo_g"]) and rec["geo_g"] > 0 and abs(math.log(rec["geo_g"])) <= math.log(F.GEO_TOL)): fails.append("geo")
        rec["why"] = "fire" if not fails else "fail:" + ",".join(fails)
        if fails:
            continue
        val = gval / F.SI[q["unit"]]
        c = rec["cs"]
        if np.isfinite(c) and c > 0 and abs(math.log(val / c)) <= math.log(2.0):
            fin, fz = math.sqrt(val * c), "agree_gm"
        elif str(rec["mode"]) in F.STRONG or not (np.isfinite(c) and c > 0):
            fin, fz = val, "geo"
        else:
            fin, fz = np.nan, "dis_nonstrong"
        rec["new"], rec["fz"] = fin, fz
    X = pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()
    patch = {}
    for i in (X.index[X.why == "fire"] if len(X) else []):
        if not np.isfinite(X.new[i]) or abs(X.new[i] / X.ans[i] - 1) < 0.005:
            continue
        patch[str(int(i))] = dict(value=float(X.new[i]),
                                  why=f"{TAG}: vertical free fall of the own target a_px={X.a_px[i]:.2f}px/f^2 (S f{X.S0[i]:.0f}-f{X.S1[i]:.0f}, "
                                      f"R2 {X.r2[i]:.4f}, own tshift {X.tshift[i]:.2f}s), own geo/g={X.geo_g[i]:.3f} -> g={X.g_si[i]:.2f} m/s^2 "
                                      f"({X.fz[i]}); was {X.ans[i]:.4g} ({X.asrc[i]})")
    return X, patch


if __name__ == "__main__":
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 40)
    for split in ("test",):
        X, patch = run(split)
        cols = [c for c in ["vid", "cat", "asrc", "mode", "ans", "g", "cs", "tdir", "tshift", "a_lo", "a_hi", "S0", "S1", "nS", "a_px", "r2", "v0",
                            "geo_g", "why", "new"] if c in X]
        print('')
        if len(X):
            print(X[X.why.astype(str).str.startswith(("fire", "fail"))][cols].round(3).to_string())
            print("why counts:", X.why.value_counts().to_dict())
        json.dump(patch, open(f"{WO}/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        for k, v in patch.items():
            print(k, round(v["value"], 4), v["why"])
