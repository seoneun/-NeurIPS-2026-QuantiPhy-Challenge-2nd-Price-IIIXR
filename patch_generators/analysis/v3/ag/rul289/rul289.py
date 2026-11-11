"""rul289: vertical free fall from rest in simulations -> the asked acceleration is g.

Rows: kind == accel, video_source simulation / segmentation, not already patch:*.  The asked window ([t1, t2], or t) lies
inside a free fall of the target that its own pixel track shows: from rest, straight down, no contact yet.  The acceleration
of a body in free fall is g, which needs no scale, no depth and no focal (only the clip's timebase, and the sim's own
convention: the dataset states "acceleration of the box before 0.6 s = 9.8 m/s^2" for falling sim objects, e.g. sim_0191,
0203, 0207, 0257-0262).

g: a gravity value stated in the prior (gravity flag / 'gravit' text, or an accel prior of the SAME object in 9.0-10.5) is used
if it is in 9.0-10.5 m/s^2; any other stated gravity -> skip; otherwise g = 9.8.

Free-fall segment S (box per frame from the target track: v2sc -> v2 -> v1 tracks; box centre-x and box bottom):
  the maximal run of consecutive tracked frames containing the asked frames such that centre-x stays within +-3 px of its value
  at S start, the box bottom is non-decreasing, and the per-frame drop does not fall by more than 40 % (once >= 3 px/frame);
  one isolated outlier frame is skipped (a frame that breaks the run while the next frame continues it, e.g. 0183 f6).
  The first frame that breaks the run is the contact frame.  Leading rest frames (>= 3 at the rest level) are trimmed to the
  last one.
Gates: |S| >= 6; robust quadratic fit (3-MAD clip) of box bottom on S: R^2 >= 0.99, a_y > 0, |a_x| <= 0.35 a_y (centre-x
quadratic); speed at S start <= 2 px/frame (released from rest); the asked end time's nearest frame is in S
(t_end * fps <= last S frame + 0.5); geo confirms |log(geo_SI / g)| <= log(1.3).
Final: g in the question's unit; pipeline fusion with CaW (agree within 2x -> geometric mean; disagreement -> the physics value
in strong modes, unchanged otherwise).
fps = the track json fps (= CSV fps, checked), never the container fps.

usage: tools/run_limited.sh --io python analysis/v3/ag/rul289/rul289.py [SUB]   (default sub_xt_Nzl)
 -> cache/tmp/ag_ffacc_patch_{test,val}.json, cache/tmp/ag_ffacc_diag_{split}.csv
"""
import json, os, sys, math
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/caw")
import measure_v3 as MV

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzl"
RAWN = os.environ.get("FFACC_RAW", "geo_v4ntrj5sQ2srngsr_d_vw")
SI = {"m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001}
STRONG = ("2d_ratio", "3d_calib")
CX_TOL, DROP_FRAC, DROP_MIN = 3.0, 0.6, 3.0
MIN_N, R2_MIN, AX_FRAC, V0_MAX = 6, 0.99, 0.35, 2.0
GEO_TOL = float(os.environ.get("FFACC_GEO_TOL", "1.3"))
END_MARGIN = float(os.environ.get("FFACC_END_MARGIN", "0.5"))   # t_end*fps <= last S frame + END_MARGIN
OUT = os.environ.get("FFACC_OUT", "ag_ffacc")


def load_vj(split, v):
    for d in (f"tracks_{split}_v2sc", f"tracks_{split}_v2", f"tracks_{split}"):
        p = f"{R}/geo/out/{d}/{v}.json"
        if os.path.exists(p):
            return json.load(open(p)), d
    return None, None


def walk(fr, cx, bot, s):
    """free-fall run from tracked index s: (members, contact index or None, outlier index or None)"""
    ref = cx[s]; prev = s; dprev = None; out = None; mem = [s]; n = len(fr)

    def ok(p, k, dp):
        if fr[k] - fr[p] <= 0:
            return False, None
        d = (bot[k] - bot[p]) / (fr[k] - fr[p])
        if abs(cx[k] - ref) > CX_TOL or bot[k] < bot[p]:
            return False, d
        if dp is not None and dp >= DROP_MIN and d < DROP_FRAC * dp:
            return False, d
        return True, d

    k = s + 1
    while k < n:
        good, d = ok(prev, k, dprev)
        if good:
            mem.append(k); prev = k; dprev = d; k += 1; continue
        if out is None and k + 1 < n:
            good2, d2 = ok(prev, k + 1, dprev)
            if good2:
                out = k; mem.append(k + 1); prev = k + 1; dprev = d2; k += 2; continue
        return mem, k, out
    return mem, None, out


def rfit(x, y, deg=2):
    """quadratic fit with one 3-MAD clip -> (coef, r2, kept mask)"""
    c = np.polyfit(x, y, deg)
    r = y - np.polyval(c, x)
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) <= 3 * max(mad, 0.5)
    if keep.sum() >= deg + 3 and keep.sum() < len(x):
        c = np.polyfit(x[keep], y[keep], deg)
    else:
        keep = np.ones(len(x), bool)
    rr = y[keep] - np.polyval(c, x[keep])
    sst = float(np.sum((y[keep] - y[keep].mean()) ** 2))
    r2 = 1 - float(np.sum(rr ** 2)) / max(sst, 1e-9)
    return c, r2, keep


def segment(fr, cx, bot, a_lo, a_hi):
    """maximal free-fall run containing tracked frames a_lo..a_hi (fidx units)"""
    n = len(fr)
    js = [j for j in range(n) if a_lo - 0.5 <= fr[j] <= a_hi + 0.5]
    if not js:
        return None, "no_frames_in_window"
    j0, j1 = js[0], js[-1]
    best = None
    for s in range(j0, -1, -1):
        mem, contact, out = walk(fr, cx, bot, s)
        if mem[-1] >= j1 and all((j in mem) or j == out for j in range(j0, j1 + 1)):
            best = (s, mem, contact, out)
        else:
            break
    if best is None:
        return None, "window_not_free"
    s, mem, contact, out = best
    # trim a leading rest period (>= 3 frames at the rest level) to its last frame
    rest = [j for j in mem if j <= j0 and bot[j] - bot[s] <= 1.0]
    if len(rest) >= 3:
        mem = [j for j in mem if j >= rest[-1]]
    return dict(mem=mem, contact=contact, out=out), "ok"


def g_from_prior(pr, prior_text, target):
    pt = str(prior_text).lower()
    if pr and (pr.get("gravity") or "gravit" in pt):
        v = float(pr.get("value_si") or 0)
        return (v, "prior_g") if 9.0 <= v <= 10.5 else (None, f"other_gravity({v})")
    if pr and pr.get("kind") == "accel" and 9.0 <= float(pr.get("value_si") or 0) <= 10.5 \
            and str(pr.get("obj", "")).strip().lower() == str(target or "").strip().lower():
        return float(pr["value_si"]), "prior_fall"
    if "gravit" in pt:
        return None, "gravity_unparsed"
    return 9.8, "default"


def tshift_map(recs, vjs):
    byv = {}
    for r in recs:
        byv.setdefault(r["video_id"], []).append(r)
    out = {}
    for v, rs in byv.items():
        vj = vjs.get(v)
        if not vj or not vj.get("fidx"):
            continue
        dur = (max(vj["fidx"]) + 1) / float(vj["fps"])
        ts = [float(r["q"][k]) for r in rs for k in ("t", "t1", "t2") if r["q"].get(k) is not None]
        if not ts or max(ts) <= dur + 0.5 or min(ts) - dur < 0 or max(ts) - dur > dur:
            continue
        out[v] = dur
    return out


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWN}_raw.csv")
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    rows, vjs, vids = [], {}, {}
    sel = [r for r in recs if r["q"].get("kind") == "accel" and r["id"] in det.index and r["id"] in raw.index
           and str(raw.video_source[r["id"]]).lower().startswith(("simulation", "segment"))]
    for r in sel:
        if r["video_id"] not in vjs:
            vjs[r["video_id"]] = load_vj(split, r["video_id"])
    shift = tshift_map([r for r in recs if r["video_id"] in vjs], {k: v[0] for k, v in vjs.items() if v[0]})
    for r in sel:
        i = r["id"]; D = det.loc[i]; q = r["q"]
        pr = r["priors"][0] if r["priors"] else None
        rec = dict(id=i, split=split, vid=r["video_id"], cat=D["cat"], q=r["question"][:70], unit=q.get("unit"),
                   asrc=D.answer_source, mode=D.get("mode", ""), ans=float(D.answer), g=float(D.g), cs=float(D.cs),
                   geo_raw=float(raw.geo_value[i]) if pd.notna(raw.geo_value[i]) else np.nan, y=D.get("y", np.nan),
                   fps=float(raw.fps[i]), prior=str(raw.ground_truth_prior[i])[:60])
        rows.append(rec)
        if str(D.answer_source).startswith("patch"):
            rec["why"] = "already_patched"; continue
        if q.get("unit") not in SI:
            rec["why"] = "unit"; continue
        gval, gwhy = g_from_prior(pr, raw.ground_truth_prior[i], q.get("target"))
        rec["gsrc"] = gwhy
        if gval is None:
            rec["why"] = gwhy; continue
        rec["g_si"] = gval
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
        sh = shift.get(r["video_id"], 0.0)
        if q.get("t1") is not None and q.get("t2") is not None:
            t_lo, t_hi = float(q["t1"]) - sh, float(q["t2"]) - sh
        elif q.get("t") is not None:
            t_lo = t_hi = float(q["t"]) - sh
        else:
            rec["why"] = "no_time"; continue
        F = np.array(vj["fidx"], float)
        jj = [int(j) for j in tr.okc]                  # usable position frames (pipeline's area filter)
        if len(jj) < MIN_N:
            rec["why"] = "short_track"; continue
        fr = F[jj]; box = np.array([tr.st[j]["box"] for j in jj], float)
        cx = 0.5 * (box[:, 0] + box[:, 2]); bot = box[:, 3]
        a_lo, a_hi = t_lo * fps, t_hi * fps
        rec["a_lo"], rec["a_hi"] = a_lo, a_hi
        S, why = segment(fr, cx, bot, a_lo, a_hi)
        if S is None:
            rec["why"] = why; continue
        mem = S["mem"]
        rec.update(S0=fr[mem[0]], S1=fr[mem[-1]], nS=len(mem), contact=(fr[S["contact"]] if S["contact"] is not None else np.nan),
                   outlier=(fr[S["out"]] if S["out"] is not None else np.nan))
        if len(mem) < 4:
            rec["why"] = "fail:nS"; continue
        x = fr[mem]; yb = bot[mem]; xc = cx[mem]
        cy, r2, keep = rfit(x, yb); ccx, _, _ = rfit(x, xc)
        a_y, a_x = 2 * cy[0], 2 * ccx[0]
        v0 = float(np.polyval(np.polyder(cy), x[0]))
        rec.update(a_px=a_y, a_x=a_x, r2=r2, v0=v0, nclip=int((~keep).sum()))
        geo_si = rec["geo_raw"] * SI[q["unit"]] if np.isfinite(rec["geo_raw"]) else rec["g"] * SI[q["unit"]]
        rec["geo_g"] = geo_si / gval if np.isfinite(geo_si) else np.nan
        fails = []
        if len(mem) < MIN_N: fails.append("nS")
        if not (r2 >= R2_MIN): fails.append("r2")
        if not (a_y > 0): fails.append("ay")
        if not (abs(a_x) <= AX_FRAC * abs(a_y)): fails.append("ax")
        if not (v0 <= V0_MAX): fails.append("v0")
        if not (a_hi <= fr[mem[-1]] + END_MARGIN): fails.append("end")
        if not (np.isfinite(rec["geo_g"]) and rec["geo_g"] > 0 and abs(math.log(rec["geo_g"])) <= math.log(GEO_TOL)): fails.append("geo")
        rec["why"] = "fire" if not fails else "fail:" + ",".join(fails)
        if fails:
            continue
        val = gval / SI[q["unit"]]
        c = rec["cs"]
        if np.isfinite(c) and c > 0 and abs(math.log(val / c)) <= math.log(2.0):
            fin, fz = math.sqrt(val * c), "agree_gm"
        elif str(rec["mode"]) in STRONG or not (np.isfinite(c) and c > 0):
            fin, fz = val, "geo"
        else:
            fin, fz = np.nan, "dis_nonstrong"
        rec["new"], rec["fz"] = fin, fz
    X = pd.DataFrame(rows, columns=["id", "split", "vid", "cat", "q", "unit", "asrc", "mode", "ans", "g", "cs", "geo_raw", "y", "fps", "prior"]
                     if not rows else None).set_index("id")
    if "why" not in X:
        X["why"] = ""
    for c_ in ("new", "fz", "a_px", "S0", "S1", "r2", "geo_g", "g_si"):
        if c_ not in X:
            X[c_] = np.nan
    patch = {}
    for i in X.index[X.why == "fire"]:
        if not np.isfinite(X.new[i]) or abs(X.new[i] / X.ans[i] - 1) < 0.005:      # 445: 9.806 -> 9.8 is a no-op
            continue
        patch[str(int(i))] = dict(value=float(X.new[i]),
                                  why=f"rul289: vertical free fall a_px={X.a_px[i]:.2f}px/f^2 (S f{X.S0[i]:.0f}-f{X.S1[i]:.0f}, R2 {X.r2[i]:.4f}), "
                                      f"geo/g={X.geo_g[i]:.3f} -> g={X.g_si[i]:.2f} m/s^2 ({X.fz[i]})")
    return X, patch


if __name__ == "__main__":
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 40)
    for split in ("test", "val"):
        X, patch = run(split)
        X.to_csv(f"{R}/cache/tmp/{OUT}_diag_{split}.csv")
        cols = [c for c in ["vid", "cat", "asrc", "ans", "g", "cs", "y", "tdir", "a_lo", "a_hi", "S0", "S1", "nS", "contact", "outlier",
                            "a_px", "a_x", "r2", "v0", "geo_g", "why", "new"] if c in X]
        print(f"===== {split}: {len(X)} sim accel rows; fire {int((X.why == 'fire').sum())}; patch {len(patch)}")
        print(X[cols].round(3).to_string())
        json.dump(patch, open(f"{R}/cache/tmp/{OUT}_patch_{split}.json", "w"), indent=1)
        for k, v in patch.items():
            print(k, round(v["value"], 4), v["why"])
