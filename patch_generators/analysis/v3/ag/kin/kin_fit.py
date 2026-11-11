"""kin rows: does the prior object start from rest at t=0? Quadratic fits of the prior object's track (2D px; 3D for D3)."""
import sys, os, json, re
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/analysis/v3")
import numpy as np, pandas as pd
import measure_v3 as MV, measure as M1
import kinematic as K
MV.FIX = set("abcdef")
K.BEFORE_OK = True
split = sys.argv[1] if len(sys.argv) > 1 else "test"
os.chdir(R)
raw = pd.read_csv(f"preds/{split}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv", index_col=0)
det = pd.read_csv(f"submissions/sub_xt_Ngd_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
P_ = {}
for l in open(f"geo/out/parse_{split}_v2.jsonl"):
    r = json.loads(l); P_[r["id"]] = r


def load_vj(v):
    for p in (f"geo/out/tracks_{split}_v2sc/{v}.json", f"geo/out/tracks_{split}_v2/{v}.json", f"geo/out/tracks_{split}/{v}.json"):
        if os.path.exists(p):
            return json.load(open(p)), p.split("/")[2]
    return None, None


def get_track(rec, vj):
    for cfg in ("e", "d"):
        try:
            V = MV.Video(vj, cfg)
        except Exception:
            continue
        for ent in (rec.get("prior_obj"), rec.get("target")):
            tr = V.get(ent)
            if tr is not None and tr.n >= 6:
                return tr, V
    return None, None


def qfit(t, P):
    """robust per-coordinate quadratic in absolute time: returns C (3 x k), rms residual, R2, n kept"""
    m = np.ones(len(t), bool)
    for _ in range(3):
        A = np.stack([np.ones(m.sum()), t[m], t[m] ** 2], 1)
        C, *_ = np.linalg.lstsq(A, P[m], rcond=None)
        res = P - np.stack([np.ones(len(t)), t, t ** 2], 1) @ C
        rn = np.linalg.norm(res, axis=1)
        s = 1.4826 * np.median(rn[m]) + 1e-9
        m2 = rn <= max(3 * s, 1e-6)
        if m2.sum() < 6 or (m2 == m).all():
            break
        m = m2
    res = P[m] - np.stack([np.ones(m.sum()), t[m], t[m] ** 2], 1) @ C
    ss_t = float(((P[m] - P[m].mean(0)) ** 2).sum())
    r2 = 1 - float((res ** 2).sum()) / ss_t if ss_t > 0 else 0.0
    return C, float(np.sqrt((res ** 2).sum(1).mean())), r2, int(m.sum())


def metrics(C, tq):
    a = 2 * C[2]; v0 = C[1]; vq = v0 + a * tq
    an = float(np.linalg.norm(a))
    if an <= 0:
        return None
    ts = -float(v0 @ a) / an ** 2                       # time of minimum speed (start from rest if ~0)
    return dict(a_px=an, v0=float(np.linalg.norm(v0)), vq=float(np.linalg.norm(vq)), r=float(np.linalg.norm(vq)) / (an * tq),
                z=float(np.linalg.norm(v0)) / (an * tq), ts=ts)


def run(ids, need_kin=True):
    rows = []
    for i in ids:
        pr, q = str(raw.ground_truth_prior[i]), str(raw.question[i])
        if need_kin:
            x, h = K.propagate(pr, q)
            if not np.isfinite(x):
                x, h = K.propagate_gravity(pr, q)
            if h not in ("a*t", "g*t"):
                continue
        else:
            x, h = np.nan, "-"
        rec = P_[i]; vid = raw.video_id[i]; fps = float(raw.fps[i])
        tk, tq, _ = K.times(q)
        if tq is None or tq <= 0:
            continue
        a_pr = rec["priors"][0]["value_si"] if rec.get("priors") else np.nan
        t2p = rec["priors"][0].get("t2") if (rec.get("priors") and need_kin) else None
        vj, src = load_vj(vid)
        out = dict(id=i, vid=vid, cat=det.cat[i], how=h, tq=tq, a=a_pr, kin=x, ans=det.answer[i], g=det.g[i], mode=(det["mode"][i] if "mode" in det else raw.meas_mode[i]), fps=fps, src=src)
        if "y" in det:
            out["y"] = det.y[i]
        if vj is None:
            rows.append(out); continue
        if need_kin:
            tr, V = get_track(rec, vj)
        else:
            tr, V = None, None
            for cfg in ("e", "d"):
                try:
                    V = MV.Video(vj, cfg)
                except Exception:
                    continue
                tr = V.get(rec.get("target"))
                if tr is not None and tr.n >= 6:
                    break
        if tr is None or tr.n < 6:
            out["why"] = "no_track"; rows.append(out); continue
        t, P = tr.t, tr.uv
        out.update(n=len(t), t0=float(t.min()), t1=float(t.max()), cam=round(V.cam_mag / V.W, 3))
        # camera check from the scene itself: another tracked object that stays put (phase correlation on rendered
        # videos follows the moving object: simulation_0208 reads 286 px of 'camera motion' with the table fixed to 2 px)
        nst = 0
        for o in {id(x): x for x in V.tr.values() if x is not None}.values():
            if o is tr or o.n < max(6, 0.5 * tr.n):
                continue
            if o.n == tr.n and np.allclose(o.uv, P, atol=1.0):      # same SAM2 object under another phrase (mug / mug cup)
                continue
            dev = np.linalg.norm(o.uv - np.median(o.uv, 0), axis=1)
            if np.percentile(dev, 90) < max(3.0, 0.005 * V.W):
                nst += 1
        out["static_ref"] = nst
        sg = tr.segs()
        out["segs"] = ";".join(f"{t[a_]:.2f}-{t[b_-1]:.2f}{'R' if rs else 'M'}" for (a_, b_), rs in zip(sg["segs"], sg["rest"]))
        bd = np.array([(tr.st[j] or {}).get("border", 9) for j in tr.okc])
        out["clip0"] = float(np.mean(bd[t <= tq] != 0)) if (t <= tq).any() else np.nan
        # geo's local estimate at tq (px/s) and the clip-average speed (px/s)
        vl = MV.vel_at_c(t, P, min(max(tq, t.min()), t.max()), sg)
        out["v_loc"] = float(np.linalg.norm(vl)) if vl is not None else np.nan
        out["v_avg"] = MV.avg_speed_c(t, P, t.min(), t.max(), sg)
        wins = {}
        k = MV.seg_index(sg, t, tq)
        if k is not None:
            a_, b_ = sg["segs"][k]; wins["seg"] = (a_, b_)
        hi = tq + 0.4 if t2p is None else min(tq + 0.4, t2p)
        m = np.where(t <= hi + 1e-9)[0]
        if len(m) >= 6:
            wins["early"] = (m[0], m[-1] + 1)
        hi2 = t.max() if t2p is None else min(t.max(), t2p)
        m = np.where(t <= hi2 + 1e-9)[0]
        if len(m) >= 6:
            wins["all"] = (m[0], m[-1] + 1)
        for wn, (a_, b_) in wins.items():
            if b_ - a_ < 6:
                continue
            C, rms, r2, nk = qfit(t[a_:b_], P[a_:b_])
            mt = metrics(C, tq)
            if mt is None:
                continue
            for kk, vv in mt.items():
                out[f"{wn}_{kk}"] = vv
            out[f"{wn}_r2"] = r2; out[f"{wn}_tw"] = f"{t[a_]:.2f}-{t[b_-1]:.2f}"
        if det.cat[i] in ("D3", "S3") and rec.get("depth") and need_kin:
            f = float(raw.meas_f[i]) * V.W if np.isfinite(raw.meas_f[i]) else np.nan
            keys = rec.get("prior_dkeys") or rec.get("target_dkeys") or []
            m = np.where(t <= (t.max() if t2p is None else min(t.max(), t2p)) + 1e-9)[0]
            for dn, Dfn in (("lin", M1.depth_fn(rec["depth"], keys)), ("trk", M1.track_depth_fn(rec["depth"], keys, tr))):
                if Dfn is None or not np.isfinite(f):
                    continue
                P3, _ = M1.to3d(P, Dfn(t), f, V.W / 2, V.H / 2)
                C, rms, r2, nk = qfit(t[m], P3[m])
                mt = metrics(C, tq)
                if mt:
                    out[f"3{dn}_a"] = mt["a_px"]; out[f"3{dn}_r"] = mt["r"]; out[f"3{dn}_z"] = mt["z"]; out[f"3{dn}_ts"] = mt["ts"]
            s = np.sqrt(np.array([tr.st[j]["area"] for j in tr.okc], float))
            s = np.array([np.median(s[max(0, ii - 2):ii + 3]) for ii in range(len(s))])
            timed = [(e["t"], e["d"]) for e in rec["depth"] if e["key"] in keys and e["t"] is not None]
            if timed and np.isfinite(f):
                Ks = [dd * np.interp(tt_, t, s) for tt_, dd in timed]
                out["sz_kcv"] = float(np.std(Ks) / np.mean(Ks))
                Zd = float(np.mean(Ks)) / s
                P3, _ = M1.to3d(P, Zd, f, V.W / 2, V.H / 2)
                C, rms, r2, nk = qfit(t[m], P3[m])
                mt = metrics(C, tq)
                if mt:
                    out["3sz_a"] = mt["a_px"]; out["3sz_r"] = mt["r"]; out["3sz_z"] = mt["z"]; out["3sz_ts"] = mt["ts"]
        rows.append(out)
    return pd.DataFrame(rows).set_index("id")


if __name__ == "__main__":
    pd.set_option("display.width", 340); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 60)
    Z = run(det.index[det.answer_source == "kin"])
    Z.to_csv(f"cache/tmp/ag_kin_fit_{split}.csv")
    cols = ["vid", "cat", "how", "tq", "a", "ans", "g", "y", "n", "t0", "t1", "cam", "clip0", "segs",
            "seg_r", "seg_z", "seg_ts", "seg_r2", "early_r", "early_z", "early_ts", "early_r2", "all_r", "all_z", "all_ts", "all_r2"]
    print(Z[[c for c in cols if c in Z]].round(3).to_string())
    c3 = [c for c in Z.columns if c.startswith("3") or c == "sz_kcv"]
    if c3:
        print(Z.loc[Z.cat == "D3", ["vid", "tq", "a", "ans", "g"] + c3].round(3).to_string())
