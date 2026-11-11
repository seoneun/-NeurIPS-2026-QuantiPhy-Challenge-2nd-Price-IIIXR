import os, sys, json, copy, glob
ENV = dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", QP_TSHIFT="1", QP_FPERSON="1", QP_FPERSON_MINN="2")
for k, v in ENV.items():
    os.environ.setdefault(k, v)
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, f"{R}/geo")
import numpy as np
import measure_v3 as MV


def setup(split, tracks="v2sc", vids=None):
    MV.TRACK_VER = tracks
    MV.FIX = set("abcdef")
    MV.MOGE_OPTS = {"fbrej"}
    for p in glob.glob(f"{R}/geo/out/moge_{split}/*.npz"):
        v = os.path.basename(p)[:-4]
        if vids and v not in vids:
            continue
        z = np.load(p)
        MV.MOGE[v] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))
    fn_ = f"{R}/geo/out/scene_ruler_{split}.json"
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
    fp_ = f"{R}/geo/out/person_focal_{split}.json"
    if os.environ.get("QP_FPERSON") == "1" and os.path.exists(fp_):
        MV.FPERSON.update(json.load(open(fp_)))
    MV.M1.SCALE_DEPTH = True
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))["T_owl"]
    fmed = {'simulation': 1.3136267014192877, 'segmentation': 1.7604539374494734, 'lab': 0.6216217221323498}
    return fmed


def measure(rs, vj, fmed, cfg="e"):
    allf = float(np.median(list(fmed.values())))
    W = vj["W"]
    fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * W for r in rs}
    o, V = MV.measure_video(copy.deepcopy(rs), vj, fdd, cfg)
    out = {}
    for x, r in zip(o, rs):
        u = r["q"]["unit"]
        out[r["id"]] = dict(value=(x["val_si"] / MV.UNIT_SI[u]) if x.get("ok") else float("nan"), mode=x.get("mode"), f=x.get("f"), why=x.get("why"))
    return out, V


if __name__ == "__main__":
    import pandas as pd
    split = sys.argv[1]; vids = sys.argv[2].split(",")
    fmed = setup(split, vids=set(vids))
    byv, vjs = MV.load_split_tracks(split, "e")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    for v in vids:
        o, _ = measure(byv[v], vjs[v], fmed)
        for i, x in o.items():
            print(v, i, "new %.4g %s %s | raw %s %s" % (x["value"], x["mode"], x["why"], raw.geo_value.get(i), raw.meas_mode.get(i)))
