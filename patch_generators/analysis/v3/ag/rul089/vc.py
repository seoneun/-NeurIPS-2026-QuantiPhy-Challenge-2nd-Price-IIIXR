"""c51scan helper: re-run the pipeline's measure_video (geo/measure_v3.py, read-only import) with the env of the rj5sQ
chain on a chosen track set (v2sc = detector chain, v3ntmv = VLM chain), and load the VLM-chain row table."""
import os, sys, json, copy, glob
ENV = dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", QP_TSHIFT="1", QP_FPERSON="1", QP_FPERSON_MINN="2",
           QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1")
for k, v in ENV.items():
    os.environ.setdefault(k, v)
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo")
import numpy as np, pandas as pd
import measure_v3 as MV
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"


def setup(split, vids, tracks="v3ntmv"):
    MV.TRACK_VER = tracks
    MV.FIX = set("abcdef")
    MV.MOGE_OPTS = {"fbrej"}
    for p in glob.glob(f"{R}/geo/out/moge_{split}/*.npz"):
        v = os.path.basename(p)[:-4]
        if v not in vids or v in MV.MOGE:
            continue
        z = np.load(p)
        MV.MOGE[v] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
    fn_ = f"{R}/geo/out/scene_ruler_{split}.json"
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
    fp_ = f"{R}/geo/out/person_focal_{split}.json"
    if os.path.exists(fp_):
        MV.FPERSON.update(json.load(open(fp_)))
    MV.M1.SCALE_DEPTH = True
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))["T_owl"]
    return json.load(open(f"{R}/geo/out/focal_medians_test.json"))["f_over_W"]


def measure(rs, vj, fmed, cfg="e"):
    allf = float(np.median(list(fmed.values())))
    fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * vj["W"] for r in rs}
    o, V = MV.measure_video(copy.deepcopy(rs), vj, fdd, cfg)
    out = {}
    for x, r in zip(o, rs):
        u = r["q"]["unit"]
        out[r["id"]] = dict(value=(x["val_si"] / MV.UNIT_SI[u]) if x.get("ok") else float("nan"), mode=x.get("mode"),
                            f=x.get("f"), mpp=x.get("mpp"), why=x.get("why"))
    return out, V


def vlm_rows(split):
    """VLM-chain rows of the final raw (merge_src v / xg_v) with the VLM measurement's own mpp / f / why"""
    r = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv").set_index("Unnamed: 0")
    v = pd.read_csv(f"{R}/preds/{split}_geo_v3v_rntrj5sQ_raw.csv").set_index("Unnamed: 0").reindex(r.index)
    vl = r.merge_src.astype(str).isin(["v", "xg_v"])
    X = pd.DataFrame(index=r.index[vl])
    X["vid"] = r.video_id[vl]; X["ms"] = r.merge_src[vl]; X["g"] = pd.to_numeric(r.geo_value[vl], errors="coerce")
    X["mode"] = r.meas_mode[vl]; X["fps"] = r.fps[vl]
    for c in ("meas_mpp", "meas_f", "meas_why", "meas_tracks", "meas_mode"):
        X[c + "_v"] = v[c][vl]
        X[c + "_r"] = r[c][vl]
    return X


def load_vj(split, vid, tracks):
    """the track file measure_v3 --tracks <tracks> reads for this video (its own dir, else the v2 / v1 fallbacks)"""
    for p in (f"{R}/geo/out/tracks_{split}_{tracks}/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json",
              f"{R}/geo/out/tracks_{split}/{vid}.json"):
        if os.path.exists(p):
            return json.load(open(p)), p.split("/")[-2]
    return None, None
