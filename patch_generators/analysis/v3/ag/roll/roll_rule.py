"""A2 person axis / rolled videos (rule tag: roll).

(1) Rolled video: a person-class track (phrase not swimmer / lying / diver) whose box is wide (median w/h > 1.5 over >= 10
    frames, >= 80% of frames wide), large enough to read its shape (long side >= 10% of the short frame side), that moves
    (travel >= 0.5 x its long side) and moves ACROSS its long axis (image-y range > 1.5 x image-x range: a walker's body
    axis is perpendicular to the walk; a top-down walker's stride box is elongated ALONG the walk and is rejected).
    -> the video is rolled 90 degrees: every "height" extent is read on the image x axis (box width), so the person ruler
       and target heights use the upright vertical; the person focal is computed from the rotated box too.
(2) Lying prior: a "height" prior of a swimmer / lying / diving person is the body length = the mask's major axis
    (rotated-rect long side), not the image-vertical box extent.
Geo is re-measured with geo/measure_v3.py's own functions (same env / args as the rj5sQ chain), once as-is and once with
the rule; the stored geo value is scaled by rule/base (so chain post-processing is kept), then the fusion is recomputed.
Writes cache/tmp/ag_roll_patch_{test,val}.json and cache/tmp/ag_roll_rows.csv."""
import os, sys, json, re, math
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
ENV = dict(QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1", QP_TSHIFT="1",
           QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", QP_FPERSON="1", QP_FPERSON_MINN="2")
os.environ.update(ENV)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/geo")
import measure_v3 as MV  # noqa: E402
assert MV.__file__.startswith(f"{R}/geo/"), MV.__file__

PER = re.compile(r"\b(person|people|man|men|woman|women|human|pedestrian|player|boy|girl|child|kid|lady|guy|walker|runner|"
                 r"jogger|athlete|skater|dancer|worker|swimmer|diver|boatman)\b", re.I)
LIE = re.compile(r"\bswim|\blying\b|\blies\b|\bdiver\b|\bdiving\b|\bcrawl|\bprone\b|\bfloating\b|\bsleep", re.I)
ROLL = set()          # rolled video ids (filled by detect_roll)
STATE = dict(on=False, vid=None)
_orig_ext = MV.Track.ext_px


def ext_px(self, dim, phrase):
    if STATE["on"] and dim == "height":
        js = self.oke
        if len(js) == 0:
            return np.array([]), js
        if STATE["vid"] in ROLL:                 # rolled 90 deg: the upright vertical is the image x axis
            box = np.array([self.st[j]["box"] for j in js], float)
            return box[:, 2] - box[:, 0], js
        if LIE.search(phrase or ""):             # a lying / swimming body: its "height" is the body length (major axis)
            rr = np.array([self.st[j]["rr"][:2] for j in js], float)
            return rr.max(1), js
    return _orig_ext(self, dim, phrase)


MV.Track.ext_px = ext_px


def setup(tracks):
    MV.TRACK_VER = tracks; MV.FIX = set("abcdef"); MV.HALF_V, MV.HALF_A = 0.25, 0.4
    MV.FOCAL_MIX = ""; MV.PARSE_TAG = ""; MV.AGG3D_P = None; MV.STATIC_FRAC = 0.25; MV.MEASURE_SCENE = False
    MV.WINGED_LEN_SHORT = False; MV.MOGE_OPTS = {"fbrej"}
    MV.M1.SCALE_DEPTH = True
    tp = f"{R}/geo/out/owl_threshold_v2.json"
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))["T_owl"]


def load_moge(split, vids):
    MV.MOGE.clear()
    for v in vids:
        p = f"{R}/geo/out/moge_{split}/{v}.npz"
        if os.path.exists(p):
            z = np.load(p)
            MV.MOGE[v] = dict(f=float(np.median(z["fx"])), J=[int(j) for j in z["j"]], depth=z["depth"], W=int(z["W"]))


def detect_roll(V):
    """-> (rolled?, evidence list) from the person-class tracks of one video"""
    ev = []
    seen = set()
    for key, tr in V.tr.items():
        ph = str(key).split("|")[0]
        if tr is None or id(tr) in seen or not PER.search(ph) or LIE.search(ph):
            continue
        seen.add(id(tr))
        js = tr.okc
        if len(js) < 10:
            continue
        B = np.array([tr.st[j]["box"] for j in js], float)
        w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
        ar = w / np.maximum(h, 1e-6)
        C = np.array([tr.st[j]["c"] for j in js], float)
        rx = float(np.percentile(C[:, 0], 95) - np.percentile(C[:, 0], 5))
        ry = float(np.percentile(C[:, 1], 95) - np.percentile(C[:, 1], 5))
        L = float(np.median(np.maximum(w, h)))
        e = dict(key=key, n=len(js), ar=round(float(np.median(ar)), 2), frac=round(float(np.mean(ar > 1.5)), 2),
                 size=round(L / min(V.W, V.H), 3), travel=round(math.hypot(rx, ry) / L, 2), ry_rx=round(ry / max(rx, 1e-6), 2))
        e["hit"] = bool(e["ar"] > 1.5 and e["frac"] >= 0.8 and e["size"] >= 0.10 and e["travel"] >= 0.5 and e["ry_rx"] > 1.5)
        if e["ar"] > 1.2:
            ev.append(e)
    return any(e["hit"] for e in ev), ev


def rot_person_focal(rs, V, vj):
    """person focal (geo/person_focal.py recipe, H = 1.70 m) from the ROTATED person box at the stated distances"""
    PERk = re.compile(r"person|people|\bman\b|woman|girl|boy|human|player|child|kid|lady|guy", re.I)
    ents = {(e["key"], e["t"], e["d"]) for r in rs for e in (r.get("depth") or []) if PERk.search(e["key"]) and e["t"] is not None}
    if len({k for k, _, _ in ents}) != 1:
        return None
    trs = [tr for key, tr in V.tr.items() if tr is not None and PERk.search(str(key).split("|")[0])]
    if not trs:
        return None
    tr = max(trs, key=lambda t_: len(t_.okc))
    fps = float(vj["fps"]); fidx = list(vj["fidx"]); ests = []
    for k, t, Z in sorted(ents, key=lambda x: x[1]):
        j = int(np.argmin(np.abs(np.array(fidx) - round(t * fps))))
        st = tr.st[j] if j < len(tr.st) else None
        if not st or not st.get("box"):
            continue
        x1, y1, x2, y2 = st["box"]
        if x1 <= 2 or x2 >= V.W - 2:
            continue
        hpx = x2 - x1
        u, v = (x1 + x2) / 2 - V.W / 2, (y1 + y2) / 2 - V.H / 2
        f = hpx * Z / 1.70
        for _ in range(3):
            z = Z / np.sqrt(1 + (u / f) ** 2 + (v / f) ** 2); f = hpx * z / 1.70
        ests.append(dict(key=k, t=t, Z=Z, j=j, h_px=float(hpx), f_over_W=float(f / V.W)))
    if len(ests) < 2:
        return None
    fw = np.array([e["f_over_W"] for e in ests])
    return dict(f_over_W=float(np.median(fw)), n=len(ests), spread=float(fw.max() / fw.min()), ests=ests)


def remeasure(split, tracks, want):
    """want: {video_id: rolled?} -> {row id: (base dict, rule dict)}"""
    setup(tracks)
    byv, vjs = MV.load_split_tracks(split, "e")
    fmed = json.load(open(f"{R}/geo/out/focal_medians_test.json"))["f_over_W"]
    allf = float(np.median(list(fmed.values())))
    MV.SCENE_RULER.clear(); MV.FPERSON.clear()
    fn_ = f"{R}/geo/out/scene_ruler_{split}.json"
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get("ok")})
    fp0 = json.load(open(f"{R}/geo/out/person_focal_{split}.json"))
    load_moge(split, list(want))
    out, info = {}, {}
    for v in want:
        if v not in vjs:
            continue
        rs = byv[v]; W = vjs[v]["W"]
        fdd = {r["video_source"]: fmed.get(r["video_source"], allf) * W for r in rs}
        res = {}
        for on in (False, True):
            STATE.update(on=on, vid=v)
            MV.FPERSON.clear(); MV.FPERSON.update(fp0)
            if on and v in ROLL:
                pf = rot_person_focal(rs, MV.Video(vjs[v], "e"), vjs[v])
                info[v] = pf
                if pf is not None:
                    MV.FPERSON[v] = pf
            o, _ = MV.measure_video([dict(r) for r in rs], vjs[v], fdd, "e")
            res[on] = o
        STATE.update(on=False, vid=None)
        for r, b, a in zip(rs, res[False], res[True]):
            out[int(r["id"])] = (b, a)
    return out, info


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    # ---------------------------------------------------------------- triggers over every video of the split
    setup("v2sc")
    byv, vjs = MV.load_split_tracks(split, "e")
    want, evid = {}, {}
    for v, rs in byv.items():
        if v not in vjs:
            continue
        V = MV.Video(vjs[v], "e")
        hit, ev = detect_roll(V)
        if ev:
            evid[v] = ev
        if hit:
            ROLL.add(v); want[v] = True
        for r in rs:
            pr = (r.get("priors") or [{}])[0]
            if pr.get("kind") == "size" and pr.get("dim") == "height" and LIE.search((r.get("prior_obj") or {}).get("phrase") or ""):
                want.setdefault(v, False)
    print(f"[{split}] person-class tracks with median w/h > 1.2:")
    for v, ev in sorted(evid.items()):
        for e in ev:
            print(f"   {v:28s} {e}")
    print(f"[{split}] rolled videos: {sorted(ROLL)} | lying-height-prior videos: {sorted(k for k, x in want.items() if not x)}")
    # ---------------------------------------------------------------- re-measure (detector tracks and VLM tracks)
    m_det, inf = remeasure(split, "v2sc", want)
    m_vlm, _ = remeasure(split, "v3ntmv", want)
    print("rotated person focal:", json.dumps(inf))
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    rows = []
    for i in sorted(m_det):
        b, a = m_det[i]
        bv, av = m_vlm.get(i, (dict(), dict()))
        mode0 = str(raw.meas_mode.get(i))
        useV = mode0.startswith("vlm_")
        B, A_ = (bv, av) if useV else (b, a)
        rows.append(dict(id=i, video_id=raw.video_id[i], mode_raw=mode0, geo_raw=raw.geo_value.get(i), src_tr="vlm" if useV else "det",
                         base=B.get("val_si") if B.get("ok") else np.nan, base_mode=B.get("mode"), base_f=B.get("f"),
                         rule=A_.get("val_si") if A_.get("ok") else np.nan, rule_mode=A_.get("mode"), rule_f=A_.get("f"),
                         rule_mpp=A_.get("mpp"), base_mpp=B.get("mpp"), why=A_.get("why")))
    COLS = ["id", "video_id", "mode_raw", "geo_raw", "src_tr", "base", "base_mode", "base_f", "rule", "rule_mode", "rule_f", "rule_mpp", "base_mpp", "why"]
    T = pd.DataFrame(rows, columns=COLS).set_index("id")
    T.to_csv(f"{R}/cache/tmp/ag_roll_meas_{split}.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.to_string())
