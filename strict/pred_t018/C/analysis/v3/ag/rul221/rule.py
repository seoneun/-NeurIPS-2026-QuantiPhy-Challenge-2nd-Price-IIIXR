import os, sys, json, math
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
TAG = "rul221"
SUB = sys.argv[1]
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
TOL = 2.0
MODES = {"3d_calib", "3d_personf"}
SRCS = {"rul263_dimgm", "rul263_caw", "agree_gm", "agree_wgm"}
EXTRA = 1.0
RMIN, RMAX = 1.1, 2.0
BAND = math.log(2.0)
DIMGM = {"width": 2.6, "length": 1.6}


def fuse_agree(r, gn):
    s, a, g, c = str(r.answer_source), float(r.answer), float(r.g), float(r.cs)
    if not (np.isfinite(c) and c > 0):
        return None, "no caw"
    if abs(math.log(gn / c)) > BAND:
        return gn, "left 2x band, strong -> geo"
    lg = math.log(g / c)
    w = math.log(a / c) / lg if abs(lg) > 1e-6 else 0.5
    if not (0.3 <= w <= 0.9):
        w = 0.5
    return a * (gn / g) ** w, f"{s} w{w:.2f}"


def fuse(r, gn, dim, src_vid):
    s, a, g = str(r.answer_source), float(r.answer), float(r.g)
    if s in ("agree_gm", "agree_wgm"):
        return fuse_agree(r, gn)
    if not src_vid.startswith("simulation") or dim not in DIMGM:
        return None, 't152'
    rmax = DIMGM[dim]
    if s == "rul263_dimgm":
        v = a * a / g
        if abs(math.log10(gn / v)) > math.log10(rmax):
            return v, 't153'
        return math.sqrt(v * gn), f"rul263_dimgm gm(caw {v:.3f}, geo)"
    if s == "rul263_caw":
        v = a
        if abs(math.log10(gn / v)) <= math.log10(rmax) and abs(math.log10(g / v)) > math.log10(rmax):
            return math.sqrt(v * gn), 't154'
        return None, "rul263_caw unchanged"
    return None, s


def timed_keys(rec, keys):
    return sorted([(d["t"], d["d"]) for d in rec["depth"] if d["key"] in keys and d["t"] is not None])


def main():
    split = "test"
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv", index_col=0)
    if "cs" not in det:
        det["cs"] = det["c"]
    cand = det[(raw.meas_kind.reindex(det.index).astype(str) == "size") & det.cat.isin(["D3", "S3"]) & det.answer_source.astype(str).isin(SRCS)]
    cand = cand[raw.meas_mode.reindex(cand.index).astype(str).isin(MODES)]
    gr = raw.geo_value.reindex(cand.index).astype(float)
    cand = cand[np.abs(cand.g / gr - 1) <= 0.01]
    P, TV, VC = {}, {}, {}
    for i, r in cand.iterrows():
        rw = raw.loc[i]; ver = str(rw.meas_tracks)
        if ver not in TV:
            MV.TRACK_VER = ver
            TV[ver] = MV.load_split_tracks(split, "e")
        byv, vjs = TV[ver]; v = rw.video_id
        recs = [x for x in byv.get(v, []) if x["id"] == i] if v in vjs else []
        if not recs:
            continue
        rec = recs[0]; vj = vjs[v]; W, H = float(vj["W"]), float(vj["H"])
        if (ver, v) not in VC:
            try:
                VC[(ver, v)] = MV.Video(vj, "e")
            except Exception:
                VC[(ver, v)] = None
        V = VC[(ver, v)]
        q = rec.get("q") or {}; dim = q.get("dim")
        tr = V.get(rec["target"]) if (V is not None and rec.get("target")) else None
        if tr is None or dim not in ("length", "width", "height"):
            continue
        keys = rec.get("target_dkeys") or []
        tk = timed_keys(rec, keys); ksrc = "target"
        if len(tk) < 2 and not keys:
            keys = rec.get("prior_dkeys") or []; tk = timed_keys(rec, keys); ksrc = "prior"
        if len(tk) < 2:
            continue
        tk = tk[:2] if len(set(x[0] for x in tk)) >= 2 else tk
        (t0, d0), (t1, d1) = tk[0], tk[-1]
        if t1 - t0 < 0.05:
            continue
        Dlin = lambda t: d0 + (d1 - d0) * (np.asarray(t, float) - t0) / (t1 - t0)
        f = float(rw.meas_f) * W
        cx, cy = W / 2.0, H / 2.0
        e, js = tr.ext_px(dim, (rec["target"] or {}).get("phrase"))
        js = np.asarray(js)
        if len(js) < 3:
            continue
        ax = "y" if dim == "height" else "x"
        bj = np.array([tr.st[j]["box"] for j in js], float)
        ej = np.asarray(e, float)
        match = float(np.median(np.abs(ej - ((bj[:, 3] - bj[:, 1]) if ax == "y" else (bj[:, 2] - bj[:, 0]))) / np.maximum(ej, 1)))
        ok = np.array([j for j in tr.okc if tr.st[j]], int)
        B = np.array([tr.st[j]["box"] for j in ok], float)
        bw, bh = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
        cutx = (B[:, 0] <= TOL) | (B[:, 2] >= W - TOL); cuty = (B[:, 1] <= TOL) | (B[:, 3] >= H - TOL)
        ea = bw if ax == "x" else bh
        cut = cutx if ax == "x" else cuty
        tk_ = tr.t_all[ok].astype(float)
        uv = np.array([tr.st[j]["c"] for j in ok], float)
        nr = np.sqrt(1 + ((uv[:, 0] - cx) ** 2 + (uv[:, 1] - cy) ** 2) / f ** 2)
        Dp = Dlin(tk_)
        sz = ea * Dp / nr / f
        kf = [int(np.argmin(np.abs(tk_ - tt))) for tt in (t0, t1)]
        kfd = [float(abs(tk_[k] - tt)) for k, tt in zip(kf, (t0, t1))]
        kcut = "".join(("C" if cut[k] else "u") if d_ <= 0.1 else "-" for k, d_ in zip(kf, kfd))
        if match > 0.08:
            continue
        if "C" not in kcut:
            continue
        kc = [k for k, d_ in zip(kf, kfd) if d_ <= 0.1 and cut[k]][-1]
        win = (tk_ >= t0 - EXTRA) & (tk_ <= t1 + EXTRA) & ~cut & (Dp > 0)
        if not win.any():
            continue
        k = int(np.argmax(np.where(win, ea, -1)))
        gn, A_r = float(sz[k]), float(sz[k] / sz[kc])
        if A_r < RMIN:
            continue
        if gn / r.g > RMAX or gn / r.g < 1 / RMAX or abs(gn / r.g - 1) < 0.05:
            continue
        val, why = fuse(r, gn, dim, v)
        if val is None:
            continue
        P[str(i)] = {"value": float(val), "why": f"{TAG}[A]: {ax}-axis box cut at depth-key frame t={tk_[kc]:.2f}s; "
                                                 f"g {r.g:.3f}->{gn:.3f} ({ksrc} keys); {why}"}
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print('')


if __name__ == "__main__":
    main()
