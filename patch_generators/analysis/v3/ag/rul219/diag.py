"""rul219 diagnostics: whole-clip displacement / path rows (no time window) in 2D videos.
For the target track (the track file measure_v3 used for the row: meas_tracks v1 -> v3ntmv, v2 -> v2sc): the visible span,
whether the first / last visible boxes are cut by the image border (within 2 px) or the track ends before the clip does,
the longest uncut run and its px speed (median 0.25 s chord speed), constancy (robust CV of chords) and straightness."""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 60)
SPLIT = sys.argv[1]
TOL, CHORD = 2.0, 0.25


def chord_speeds(t, P, h=CHORD):
    out = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + h - 1e-9)
        if j >= len(t):
            break
        out.append(float(np.linalg.norm(P[j] - P[i]) / (t[j] - t[i])))
    return np.array(out)


def rcv(x):
    if len(x) < 3:
        return np.nan
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)) / m) if m > 0 else np.nan


def runs(cut, t):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out, s = [], None
    for k in range(len(cut)):
        brk = k > 0 and (t[k] - t[k - 1]) > 3 * dt
        if cut[k] or brk:
            if s is not None:
                out.append((s, k)); s = None
        if not cut[k] and s is None:
            s = k
    if s is not None:
        out.append((s, len(cut)))
    return out


def side(b, W, H):
    s = ""
    if b[0] <= TOL: s += "L"
    if b[2] >= W - 1 - TOL: s += "R"
    if b[1] <= TOL: s += "T"
    if b[3] >= H - 1 - TOL: s += "B"
    return s


FF = json.load(open(f"{R}/geo/out/file_fps_{SPLIT}.json"))
raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
MV.TRACK_VER = "v2sc"
byv, vjs = MV.load_split_tracks(SPLIT, "e")
MV.TRACK_VER = "v3ntmv"
_, vjs3 = MV.load_split_tracks(SPLIT, "e")
rows = []
for v, recs in byv.items():
    cand = [r for r in recs if (r.get("q") or {}).get("kind") in ("disp", "path")
            and all((r.get("q") or {}).get(k) is None for k in ("t", "t1", "t2", "when"))]
    if not cand:
        continue
    MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
    VV = {}
    for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
        if v in src_:
            VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
    for r in cand:
        pr = (r.get("priors") or [{}])[0]
        e = FF.get(v) or {}
        n = e.get("n480") or e.get("n")
        fps = float(r.get("fps") or e.get("csv"))
        d = dict(id=r["id"], vid=v, cat=r["cat"], vt=r["video_type"], vsrc=r.get("video_source"), qk=r["q"]["kind"],
                 question=r["question"][:80], praw=pr.get("raw"), pkind=pr.get("kind"), pt=pr.get("t"), pt1=pr.get("t1"), vsi=pr.get("value_si"),
                 pobj=(r.get("prior_obj") or {}).get("phrase"), tgt=(r.get("target") or {}).get("phrase"),
                 n=n, fps=fps, T=(n / fps if n else np.nan))
        d["same"] = (d["pobj"] or "").strip().lower() == (d["tgt"] or "").strip().lower() and bool(d["tgt"])
        rr = raw.loc[r["id"]] if r["id"] in raw.index else None
        if rr is None:
            d["note"] = "no_raw"; rows.append(d); continue
        d.update(mode=str(rr.meas_mode), mtr=rr.meas_tracks, mpp=rr.meas_mpp, geo=rr.geo_value, why=rr.meas_why)
        mt_ = rr.meas_tracks if rr.meas_tracks in VV else "v2"
        if mt_ not in VV:
            d["note"] = "no_track_file"; rows.append(d); continue
        vj, V = VV[mt_]
        W, H = float(vj["W"]), float(vj["H"])
        d["vjfps"] = vj.get("fps"); d["nfidx"] = len(vj["fidx"]); d["fmax"] = max(vj["fidx"])
        tr = V.get(r["target"]) if (r.get("target") or {}).get("phrase") else None
        if tr is None or tr.n < 3:
            d["note"] = "no_target_track"; rows.append(d); continue
        ptr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
        d["same_tr"] = bool(ptr is not None and (ptr is tr or (tr.q.get("obj") is not None and ptr.q.get("obj") == tr.q.get("obj"))))
        t, P = np.asarray(tr.t, float), np.asarray(tr.uv, float)
        boxes = [tr.st[j]["box"] for j in tr.okc]
        cut = np.array([side(b, W, H) != "" for b in boxes])
        d.update(trn=tr.n, t_first=round(float(t[0]), 3), t_last=round(float(t[-1]), 3), t_end=round(float(vj["fidx"][-1]) / float(vj["fps"]), 3),
                 s_first=side(boxes[0], W, H), s_last=side(boxes[-1], W, H), clip=round(float(cut.mean()), 3),
                 size=round(tr.size_px, 1), W=W, H=H)
        d["px_disp"] = float(np.linalg.norm(P[-1] - P[0]))
        rs = runs(cut, t)
        if rs:
            i, j = max(rs, key=lambda x: (t[x[1] - 1] - t[x[0]], x[1] - x[0]))
            tt, PP = t[i:j], P[i:j]
            d.update(run_n=j - i, run_t0=round(float(tt[0]), 3), run_t1=round(float(tt[-1]), 3), run_disp=round(float(np.linalg.norm(PP[-1] - PP[0])), 1))
            cs_ = chord_speeds(tt, PP)
            d["chord_run"] = float(np.median(cs_)) if len(cs_) else np.nan
            d["cv_run"] = rcv(cs_)
            if len(tt) >= 4:
                cx, cy = np.polyfit(tt, PP[:, 0], 1), np.polyfit(tt, PP[:, 1], 1)
                res = np.hypot(PP[:, 0] - np.polyval(cx, tt), PP[:, 1] - np.polyval(cy, tt))
                d["lin_v"] = float(np.hypot(cx[0], cy[0])); d["lin_vx"] = float(cx[0]); d["lin_vy"] = float(cy[0])
                d["lin_res"] = float(np.sqrt(np.mean(res ** 2)))
                # first vs second half of the run (speed constancy)
                h = len(tt) // 2
                if h >= 3 and len(tt) - h >= 3:
                    v1 = np.hypot(np.polyfit(tt[:h], PP[:h, 0], 1)[0], np.polyfit(tt[:h], PP[:h, 1], 1)[0])
                    v2 = np.hypot(np.polyfit(tt[h:], PP[h:, 0], 1)[0], np.polyfit(tt[h:], PP[h:, 1], 1)[0])
                    d["half_ratio"] = float(v2 / v1) if v1 > 0 else np.nan
        cs_f = chord_speeds(t, P)
        d["chord_full"] = float(np.median(cs_f)) if len(cs_f) else np.nan
        rows.append(d)
D = pd.DataFrame(rows).set_index("id")
D.to_csv(f"{R}/cache/tmp/ag_clipdisp_diag_{SPLIT}.csv")
print(SPLIT, "rows", len(D), D.qk.value_counts().to_dict(), "2D", int((D.vt.astype(str).str[1] == "2").sum()))
