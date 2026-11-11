"""rul221: D3/S3 3D size rows whose target box is cut by the image border along the measured axis at the depth-key
frames. The extent is re-read at the frame (within the depth keys' span +-1 s) with the largest extent that is NOT cut on that
axis; the depth there is the linear inter/extrapolation of the target's two timed depth keys (prior keys if the target has
none). g_new = ext_px * D / (|ray| f) with the raw's own focal; re-fused as the row's current fusion (d3size_dimgm: gm with the
scaled CaW answer and its 1.6/2.6 ratio gate; agree_*: tedge fuse). Inputs only; labels evaluate only.
Writes cache/tmp/ag_d02d3edgekey_patch_{test,val}.json and ..._diag.csv."""
import os, sys, json, math
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, R + "/geo")
import measure_v3 as MV
import measure as M1
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
TAG = "rul221"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz36"
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
TOL = 2.0
MODES = {"3d_calib", "3d_personf"}
SRCS = {"d3size_dimgm", "d3size_caw", "agree_gm", "agree_wgm"}
EXTRA = 1.0          # s beyond the depth keys
RMIN, RMAX = 1.1, 2.0
VAR = os.environ.get("D02_VAR", "A")
BAND = math.log(2.0)
DIMGM = {"width": 2.6, "length": 1.6}     # simulation:DIM:0:RATIO (sub_v30 command)
SCALE = {"height": 1.05}                   # d3size_scale simulation 1.1 (-height), 1.05 (+height)


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
        return None, f"{s}: no dimgm for this dim/source"
    rmax = DIMGM[dim]
    if s == "d3size_dimgm":
        v = a * a / g                       # the scaled CaW value the gm was taken with
        if abs(math.log10(gn / v)) > math.log10(rmax):
            return v, f"dimgm gate fails at new geo -> CaW {v:.3f}"
        return math.sqrt(v * gn), f"d3size_dimgm gm(caw {v:.3f}, geo)"
    if s == "d3size_caw":
        v = a
        if abs(math.log10(gn / v)) <= math.log10(rmax) and abs(math.log10(g / v)) > math.log10(rmax):
            return math.sqrt(v * gn), "d3size_caw entered dimgm gate -> gm"
        return None, "d3size_caw unchanged"
    return None, s


def timed_keys(rec, keys):
    return sorted([(d["t"], d["d"]) for d in rec["depth"] if d["key"] in keys and d["t"] is not None])


def main():
    import analyze_caw as A
    from lab_rows import labelled
    diag, patches = [], {}
    for split in ("test", "val"):
        raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv", index_col=0)
        det = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv" if split == "test" else f"{R}/submissions/{SUB}_valdetail.csv", index_col=0)
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
            row = dict(split=split, id=i, vid=v, q=str(rw.question)[:60], src=r.answer_source, mode=rw.meas_mode, g=r.g, cs=r.cs, ans=r.answer)
            recs = [x for x in byv.get(v, []) if x["id"] == i] if v in vjs else []
            if not recs:
                row["why"] = "no rec"; diag.append(row); continue
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
                row["why"] = "no track/dim"; diag.append(row); continue
            keys = rec.get("target_dkeys") or []
            tk = timed_keys(rec, keys); ksrc = "target"
            if len(tk) < 2 and not keys:
                keys = rec.get("prior_dkeys") or []; tk = timed_keys(rec, keys); ksrc = "prior"
            if len(tk) < 2:
                row["why"] = "no 2 timed keys"; diag.append(row); continue
            tk = tk[:2] if len(set(x[0] for x in tk)) >= 2 else tk
            (t0, d0), (t1, d1) = tk[0], tk[-1]
            if t1 - t0 < 0.05:
                row["why"] = "keys same t"; diag.append(row); continue
            Dlin = lambda t: d0 + (d1 - d0) * (np.asarray(t, float) - t0) / (t1 - t0)
            f = float(rw.meas_f) * W
            cx, cy = W / 2.0, H / 2.0
            e, js = tr.ext_px(dim, (rec["target"] or {}).get("phrase"))
            js = np.asarray(js)
            if len(js) < 3:
                row["why"] = "few frames"; diag.append(row); continue
            ax = "y" if dim == "height" else "x"
            bj = np.array([tr.st[j]["box"] for j in js], float)
            ej = np.asarray(e, float)
            match = float(np.median(np.abs(ej - ((bj[:, 3] - bj[:, 1]) if ax == "y" else (bj[:, 2] - bj[:, 0]))) / np.maximum(ej, 1)))
            # every tracked frame (okc), boxes -> extents on the measured axis (a) and the other axis (o)
            ok = np.array([j for j in tr.okc if tr.st[j]], int)
            B = np.array([tr.st[j]["box"] for j in ok], float)
            bw, bh = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
            cutx = (B[:, 0] <= TOL) | (B[:, 2] >= W - TOL); cuty = (B[:, 1] <= TOL) | (B[:, 3] >= H - TOL)
            ea, eo = (bw, bh) if ax == "x" else (bh, bw)
            cut, cuto = (cutx, cuty) if ax == "x" else (cuty, cutx)
            eo_s = np.array([np.median(eo[max(0, k - 2):k + 3]) for k in range(len(eo))])
            tk_ = tr.t_all[ok].astype(float)
            uv = np.array([tr.st[j]["c"] for j in ok], float)
            nr = np.sqrt(1 + ((uv[:, 0] - cx) ** 2 + (uv[:, 1] - cy) ** 2) / f ** 2)
            Dp = Dlin(tk_)
            sz = ea * Dp / nr / f
            used = np.isin(ok, js)
            g_rep = float(np.median(sz[used])) if used.any() else np.nan
            kf = [int(np.argmin(np.abs(tk_ - tt))) for tt in (t0, t1)]
            kfd = [float(abs(tk_[k] - tt)) for k, tt in zip(kf, (t0, t1))]
            kcut = "".join(("C" if cut[k] else "u") if d_ <= 0.1 else "-" for k, d_ in zip(kf, kfd))
            dt_used = float(np.median(np.maximum(0, np.maximum(t0 - tk_[used], tk_[used] - t1)))) if used.any() else np.nan
            row.update(ksrc=ksrc, t0=t0, t1=t1, ax=ax, match=round(match, 3), n=len(ok), ncut=int(cut.sum()), kcut=kcut, g_rep=g_rep,
                       dt_used=dt_used)
            if match > 0.08:
                row["why"] = "axis extent != box side"; diag.append(row); continue
            if "C" not in kcut:
                row["why"] = "key frames uncut"; diag.append(row); continue
            kc = [k for k, d_ in zip(kf, kfd) if d_ <= 0.1 and cut[k]][-1]
            # variant A (audit spec): frame with the largest uncut extent within keys +-1 s, depth = linear key inter/extrapolation
            win = (tk_ >= t0 - EXTRA) & (tk_ <= t1 + EXTRA) & ~cut & (Dp > 0)
            if win.any():
                k = int(np.argmax(np.where(win, ea, -1)))
                row.update(A_t=float(tk_[k]), A_e=float(ea[k]), A_ecut=float(ea[kc]), A_gn=float(sz[k]), A_r=float(sz[k] / sz[kc]))
            # variant B: depth carried from the cut key frame by the other-axis apparent size (pinhole Z ~ 1/size)
            Zk = float(Dp[kc] / nr[kc])
            selB = used & ~cut & ~cuto
            if not cuto[kc] and selB.any():
                Zb = Zk * eo_s[kc] / eo_s[selB]
                szB = ea[selB] * Zb / f
                row.update(B_Zk=Zk, B_Zused=float(np.median(Zb)), B_Dlin_used=float(np.median((Dp / nr)[selB])), B_gn=float(np.median(szB)))
            gn = row.get(VAR + "_gn")
            if gn is None:
                row["why"] = "no new geo"; diag.append(row); continue
            if VAR == "A" and row["A_r"] < RMIN:
                row["why"] = "r<1.1"; diag.append(row); continue
            if VAR == "B" and not (dt_used > 0.3):
                row["why"] = "used frames inside key span"; diag.append(row); continue
            if gn / r.g > RMAX or gn / r.g < 1 / RMAX or abs(gn / r.g - 1) < 0.05:
                row["why"] = "gn/g out of range"; diag.append(row); continue
            val, why = fuse(r, gn, dim, v)
            if val is None:
                row["why"] = "fuse skip: " + why; diag.append(row); continue
            row["new_ans"] = val; row["why"] = "ok"
            P[str(i)] = {"value": float(val), "why": f"{TAG}[{VAR}]: {ax}-axis box cut at depth-key frame t={tk_[kc]:.2f}s; "
                                                     f"g {r.g:.3f}->{gn:.3f} ({ksrc} keys); {why}"}
            diag.append(row)
        json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
        patches[split] = P
        print(split, "candidates", len(cand), "patched", len(P))
    D = pd.DataFrame(diag)
    D.to_csv(f"{R}/cache/tmp/ag_{TAG}_diag.csv", index=False)
    pd.set_option('display.width', 300); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)
    print(D.why.value_counts().to_string())
    cols = [c for c in ["split", "id", "vid", "q", "src", "ksrc", "t0", "t1", "n", "ncut", "kcut", "dt_used", "g", "g_rep", "A_t", "A_r", "A_gn", "B_Zk", "B_Zused", "B_Dlin_used", "B_gn", "cs", "ans", "new_ans", "why"] if c in D]
    sel = D.ncut.fillna(0) > 0 if "ncut" in D else D.why.notna()
    print(D[sel][cols].round(3).to_string())
    for split, P in patches.items():
        for k, x in P.items():
            print(split, k, x)
    L = labelled(SUB)
    for split, P in patches.items():
        ids = [int(k) for k in P]
        hit = L[(L.split == split) & L.index.isin(ids)]
        print("labelled hits", split, len(hit))
        for i, lr in hit.iterrows():
            print(split, i, lr.video_id, "y", lr.y, "old m", round(lr.m, 3), "new m", A.row_mra(np.array([P[str(i)]["value"]]), np.array([lr.y])))
    # labelled rows near the gate (diagnostics): what would geo-at-uncut give?
    if "kcut" in D:
        for _, d in D[D.kcut.astype(str).str.contains("C")].iterrows():
            hit = L[(L.split == d.split) & (L.index == d.id)]
            if len(hit):
                y = float(hit.y.iloc[0]); print("lab-diag", d.split, d.id, d.vid, d.why, "y", y, "g", round(d.g, 3), "A", round(d.get("A_gn", np.nan), 3), "B", round(d.get("B_gn", np.nan), 3), "ans", round(d.ans, 3))


if __name__ == "__main__":
    main()
