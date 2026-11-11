"""rul116 part A diagnostics: dist2 rows (query time t) in geo mode 2d_ratio whose target / target2 box (the tracks
measure_v3 uses, incl. the second_fix instance) touches the image border (2 px) at t on exactly one side of that track's
motion axis. Rigid length L = median box extent on that axis over the track's fully uncut frames (>= 8, CV <= 3%).
Rebuilt centre on the axis = free edge -/+ L/2; perpendicular coordinate = box centre. px' = distance of the two rebuilt
box centres. writes cache/tmp/ag_c66d2edge_diagA_{split}.csv"""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
os.environ.setdefault("QP_SECOND_FIX", "1")
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 40)
SPLIT = sys.argv[1]
TOL, NUNC, CVMAX = 2.0, 8, 0.03


def frame_at(tr, t):
    ta = np.asarray(tr.t_all, float)
    dt = float(np.median(np.diff(ta))) if len(ta) > 1 else 1.0
    j = int(np.argmin(np.abs(ta - t)))
    if abs(ta[j] - t) > 1.5 * dt or not tr.st[j] or tr.st[j].get("box") is None:
        return None
    return j


def cutv(b, W, H):
    return (b[0] <= TOL, b[2] >= W - 1 - TOL, b[1] <= TOL, b[3] >= H - 1 - TOL)    # L R T B


def info(tr, j, W, H):
    """rebuilt centre of track tr at frame j; dict"""
    b = np.array(tr.st[j]["box"], float)
    c_box = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    c_m = np.array(tr.st[j]["c"], float)
    okc = [k for k in range(len(tr.st)) if tr.st[k] and tr.st[k].get("box") is not None]
    C = np.array([tr.st[k]["c"] for k in okc], float)
    ax = 0 if (C[:, 0].max() - C[:, 0].min()) >= (C[:, 1].max() - C[:, 1].min()) else 1
    cv = cutv(b, W, H)
    lo, hi = (cv[0], cv[1]) if ax == 0 else (cv[2], cv[3])
    unc = [k for k in okc if not any(cutv(tr.st[k]["box"], W, H))]
    ext = np.array([tr.st[k]["box"][ax + 2] - tr.st[k]["box"][ax] for k in unc], float)
    L = float(np.median(ext)) if len(ext) else np.nan
    cvL = float(np.std(ext) / np.mean(ext)) if len(ext) >= 2 else np.nan
    d = dict(ax="xy"[ax], cut_any=any(cv), side=("lo" if lo and not hi else "hi" if hi and not lo else "both" if lo and hi else "none"),
             perp_cut=bool(cv[2] or cv[3]) if ax == 0 else bool(cv[0] or cv[1]), n_unc=len(unc), L=L, cvL=cvL,
             ext_t=float(b[ax + 2] - b[ax]), box=[round(x) for x in b])
    c_new = c_box.copy()
    d["fire"] = d["side"] in ("lo", "hi") and len(unc) >= NUNC and np.isfinite(cvL) and cvL <= CVMAX
    if d["side"] == "lo":
        c_new[ax] = b[ax + 2] - L / 2
    elif d["side"] == "hi":
        c_new[ax] = b[ax] + L / 2
    d.update(c_box=c_box, c_m=c_m, c_new=c_new)
    return d


raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
MV.TRACK_VER = "v2sc"
byv, vjs = MV.load_split_tracks(SPLIT, "e")
MV.TRACK_VER = "v3ntmv"
_, vjs3 = MV.load_split_tracks(SPLIT, "e")
rows = []
for v, recs in byv.items():
    cand = [r for r in recs if r["q"]["kind"] == "dist2" and r["q"].get("t") is not None]
    if not cand:
        continue
    MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
    VV = {}
    for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
        if v in src_:
            VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
    for r in cand:
        if r["id"] not in raw.index:
            continue
        rr = raw.loc[r["id"]]
        d = dict(id=r["id"], vid=v, cat=r["cat"], qt=r["q"]["t"], mode=str(rr.meas_mode), mtr=rr.meas_tracks, grav=str(rr.meas_grav),
                 mpp=rr.meas_mpp, meas=rr.meas_value, geo=rr.geo_value, minq=bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", r["question"].lower())))
        mt_ = rr.meas_tracks if rr.meas_tracks in VV else "v2"
        if mt_ not in VV:
            d["note"] = "no_track_file"; rows.append(d); continue
        vj, V = VV[mt_]
        W, H = float(vj["W"]), float(vj["H"])
        tg = V.get(r["target"])
        tg2 = V.get(r["target2"]) if r.get("target2") else None
        if d["grav"] == "second_fix" and tg is not None:
            tg2 = MV.second_instance(V, vj, tg)
        if tg is None or tg2 is None:
            d["note"] = "no_track"; rows.append(d); continue
        d.update(tk1=tg.q.get("key"), tk2=tg2.q.get("key"), o1=tg.q.get("obj"), o2=tg2.q.get("obj"))
        t = float(r["q"]["t"])
        j1, j2 = frame_at(tg, t), frame_at(tg2, t)
        if j1 is None or j2 is None:
            d["note"] = "miss_at_t"; rows.append(d); continue
        try:
            pv = MV.M1.dist2_px(tg, tg2, t, r["q"].get("when"))
        except Exception as e:
            pv = None
        d["px_pipe"] = pv
        a, b = info(tg, j1, W, H), info(tg2, j2, W, H)
        for k_, x in (("1", a), ("2", b)):
            for kk in ("ax", "side", "perp_cut", "n_unc", "L", "cvL", "ext_t", "fire", "box"):
                d[kk + k_] = x[kk]
        d["px_box"] = float(np.linalg.norm(a["c_box"] - b["c_box"]))
        d["px_c"] = float(np.linalg.norm(a["c_m"] - b["c_m"]))
        ca = a["c_new"] if a["fire"] else a["c_box"]
        cb = b["c_new"] if b["fire"] else b["c_box"]
        d["px_new"] = float(np.linalg.norm(ca - cb))
        d["trig"] = (a["side"] in ("lo", "hi")) or (b["side"] in ("lo", "hi"))
        d["fire"] = bool(a["fire"] or b["fire"]) and not any(x["side"] in ("lo", "hi") and not x["fire"] for x in (a, b))
        rows.append(d)
D = pd.DataFrame(rows).set_index("id")
D.to_csv(f"{R}/cache/tmp/ag_c66d2edge_diagA_{SPLIT}.csv")
X = D[D["mode"].astype(str).str.startswith("2d_ratio") & D.get("trig", pd.Series(False, index=D.index)).fillna(False).astype(bool)]
print(SPLIT, "dist2 rows", len(D), "2d_ratio with a one-side axis cut", len(X), "fire", int(X.fire.sum()))
cols = ["vid", "qt", "grav", "tk1", "tk2", "side1", "ax1", "n_unc1", "L1", "cvL1", "ext_t1", "fire1", "side2", "ax2", "n_unc2", "L2", "cvL2", "ext_t2", "fire2",
        "px_pipe", "px_c", "px_box", "px_new", "mpp", "geo", "fire"]
print(X[cols].round(3).to_string())
Y = D[D["mode"].astype(str).str.contains("2d_ratio") & ~D["mode"].astype(str).str.startswith("2d_ratio") & D.get("trig", pd.Series(False, index=D.index)).fillna(False).astype(bool)]
print("\nother 2d modes with a trigger (not in scope):"); print(Y[cols + ["mode"]].round(3).to_string())
