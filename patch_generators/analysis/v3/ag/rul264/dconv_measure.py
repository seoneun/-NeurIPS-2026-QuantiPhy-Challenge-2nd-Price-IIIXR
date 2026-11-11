"""rul264: for every dist2 row (val + test) compute the centre-to-centre and surface-gap distances from the tracks, in the
scale the geo measurement uses. Writes R/cache/tmp/ag_dconv_rows_{split}.csv"""
import sys, os, json, re
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
import numpy as np, pandas as pd
import measure_v3 as MV
MV.FIX = set("df"); MV.TRACK_VER = "v2sc"

MINRE = r"\b(?:minimum|smallest|closest|shortest|nearest)\b"


def wording(q):
    q = q.lower()
    if re.search(MINRE, q): return "min"
    if re.search(r"\bgap\b", q): return "gap"
    if re.search(r"how far apart|far apart", q): return "apart"
    if re.search(r"\bspacing\b|\bseparation\b|\bseparat", q): return "spacing"
    if re.search(r"distance between", q): return "between"
    if re.search(r"distance (?:from|of)\b", q): return "from_to"
    if re.search(r"how far", q): return "howfar"
    return "other"


def pair_stats(tr1, tr2, t, when):
    fr1, fr2 = tr1.q.get("frames"), tr2.q.get("frames")
    T = tr1.t_all
    both = [j for j in range(len(T)) if fr1[j].get("st") and fr2[j].get("st")]
    if not both:
        return None
    if t is not None or when is not None:
        if when == "start":
            t = max(tr1.t.min(), tr2.t.min())
        elif when == "end":
            t = min(tr1.t.max(), tr2.t.max())
        js = [min(both, key=lambda j: abs(T[j] - t))]
    else:
        js = both[::max(1, len(both) // 25)]
    cs, gp, gb, gd = [], [], [], []
    for j in js:
        s1, s2 = fr1[j]["st"], fr2[j]["st"]
        c1, c2 = np.array(s1["c"], float), np.array(s2["c"], float)
        c = float(np.linalg.norm(c1 - c2))
        if fr1[j].get("poly") and fr2[j].get("poly"):
            g = MV._poly_dist(fr1[j]["poly"], fr2[j]["poly"])[0]
        else:
            g = np.nan
        b1, b2 = s1["box"], s2["box"]
        dx = max(0, b2[0] - b1[2], b1[0] - b2[2]); dy = max(0, b2[1] - b1[3], b1[1] - b2[3])
        bg = float(np.hypot(dx, dy))
        u = (c2 - c1) / max(c, 1e-6)
        h1 = (b1[2] - b1[0]) / 2 * abs(u[0]) + (b1[3] - b1[1]) / 2 * abs(u[1])
        h2 = (b2[2] - b2[0]) / 2 * abs(u[0]) + (b2[3] - b2[1]) / 2 * abs(u[1])
        cs.append(c); gp.append(g); gb.append(bg); gd.append(max(0.0, c - h1 - h2))
        sz1 = float(np.hypot(b1[2] - b1[0], b1[3] - b1[1])); sz2 = float(np.hypot(b2[2] - b2[0], b2[3] - b2[1]))
    gp = np.array(gp, float)
    return dict(c_px=float(np.median(cs)), gpoly_px=float(np.nanmedian(gp)) if np.isfinite(gp).any() else np.nan,
                gbox_px=float(np.median(gb)), gdir_px=float(np.median(gd)), nfr=len(js), diag1=sz1, diag2=sz2)


def run(split):
    byv, vjs = MV.load_split_tracks(split, "d")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/sub_xt_Ngd_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    out = []
    for v, rs in byv.items():
        rs2 = [r for r in rs if r["q"]["kind"] == "dist2" or (r["id"] in raw.index and raw.loc[r["id"], "meas_kind"] == "dist2")]
        if not rs2:
            continue
        V = MV.Video(vjs[v], "d") if v in vjs else None
        for r in rs2:
            q = r["q"]
            d = dict(id=r["id"], split=split, video_id=v, src=r["video_source"], vtype=r["video_type"], cat=r["cat"],
                     question=r["question"], prior=r.get("prior_text"), pkind=q["kind"], wording=wording(r["question"]),
                     t=q.get("t"), when=q.get("when"), tgt=(r.get("target") or {}).get("phrase"), tgt2=(r.get("target2") or {}).get("phrase"))
            if V is not None and r.get("target"):
                a, b = V.get(r["target"]), (V.get(r["target2"]) if r.get("target2") else None)
                d["same_obj"] = bool(a is not None and b is not None and a.q.get("obj") is not None and a.q.get("obj") == b.q.get("obj"))
                if a is not None and b is not None and a.n and b.n:
                    try:
                        st = pair_stats(a, b, q.get("t"), q.get("when"))
                    except Exception as e:
                        st = dict(err=str(e)[:80])
                    if st:
                        d.update(st)
            if r["id"] in raw.index:
                x = raw.loc[r["id"]]
                for k in ("meas_value", "geo_value", "vlm_value", "meas_gap", "meas_mode", "meas_mpp", "merge_src", "meas_valid", "meas_kind", "meas_why"):
                    d[k] = x.get(k)
            if r["id"] in det.index:
                x = det.loc[r["id"]]
                for k in ("answer", "answer_source", "g", "cs", "mode"):
                    d["d_" + k] = x.get(k)
                if "y" in det.columns:
                    d["y_val"] = x["y"]
            out.append(d)
    df = pd.DataFrame(out)
    df.to_csv(f"{R}/cache/tmp/ag_dconv_rows_{split}.csv", index=False)
    print(split, len(df), df.pkind.value_counts().to_dict(), df.src.value_counts().to_dict())


if __name__ == "__main__":
    for s in sys.argv[1:] or ["val", "test"]:
        run(s)
