# rul013: speed question whose geo abstained (static / 2-decade sanity / implausible) because the e-config track of the target key
# barely moves, while the d-config track of the SAME key is a different object (median IoU < 0.1) that moves > 5 of its own sizes.
# A speed question implies the target moves, so the moving d-track is taken as the target.  When the prior is a size of that same
# object (prior phrase == target phrase), the d-track is its own ruler: speed = |v_px(t)| * fps * prior / size_px (planar, top-down).
# fps = the dataset CSV fps (never the container fps).  Final answer follows the pipeline fusion: within 2x of CaW -> geometric mean,
# otherwise the measurement (a 2d_ratio-type strong mode).
import json, sys, os, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRI = re.compile(r"\s*(?:the )?(diameter|length|width|height|size) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*(m|cm|mm)\s*$")

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)

def arr(T, o):
    L = T["objs"].get(str(o)) or []
    f = [T["fidx"][k] for k, s in enumerate(L) if s and s.get("st")]
    b = np.array([s["st"]["box"] for s in L if s and s.get("st")], float).reshape(-1, 4)
    return np.array(f, float), b

size_px = lambda b: float(np.median(np.maximum(b[:, 2]-b[:, 0], b[:, 3]-b[:, 1])))
def excursion(b):
    c = (b[:, :2] + b[:, 2:]) / 2
    return float(np.max(np.linalg.norm(c - np.median(c, 0), axis=1)) / max(size_px(b), 1e-9))

def local_speed_px(fd, bd, ft):
    """|linear-fit centre velocity| in px/frame at frame ft: +-4 frames, or a one-sided 6-frame window (before, else after) when the
    dominant axis reverses inside the +-4 window."""
    cd = (bd[:, :2] + bd[:, 2:]) / 2
    ax = int(np.argmax(np.ptp(cd, 0)))
    for lo, hi in [(ft - 4, ft + 4), (ft - 6, ft), (ft, ft + 6)]:
        k = (fd >= lo) & (fd <= hi)
        if k.sum() < 3: continue
        dv = np.diff(cd[k][:, ax])
        if np.all(dv >= -0.5) or np.all(dv <= 0.5):
            return float(np.linalg.norm(np.polyfit(fd[k], cd[k], 1)[0])), (lo, hi)
    return None, None

def run(split, verbose=True):
    D = pd.read_csv(f"{R}/submissions/sub_xt_Nzn_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out, cnt = {}, dict(g1=0, g2=0, g3=0, g4=0, g5=0, g6=0, g7=0, win=0)
    for i, d in D.iterrows():
        if str(d.answer_source).replace("mo_guard:", "") not in ("caw_nogeo", "dis_caw_routed"): continue
        cnt["g1"] += 1
        p = P.get(i) or {}; q = p.get("q") or {}
        if q.get("kind") != "speed" or q.get("t") is None: continue
        cnt["g2"] += 1
        if not re.search(r"static|sanity|implausible", str(raw.meas_why[i])): continue
        cnt["g3"] += 1
        ph = ((p.get("target") or {}).get("phrase") or "").lower()
        po = p.get("prior_obj") or {}
        pph = (po.get("phrase") or "").lower() if isinstance(po, dict) else ""
        m = PRI.match(str(raw.ground_truth_prior[i]).lower())
        if not m or not ph or ph != pph or m.group(2).strip() != ph: continue
        cnt["g4"] += 1
        prior_m = float(m.group(3)) * SI[m.group(4)]
        f = f"{R}/geo/out/tracks_{split}_v2sc/{d.video_id}.json"
        if not os.path.exists(f): continue
        T = json.load(open(f)); fps = float(raw.fps[i])            # CSV fps
        qe = [x for x in T.get("queries") or [] if x["key"].split("|")[0] == ph]
        qd = [x for x in T.get("queries_d") or [] if x["key"].split("|")[0] == ph]
        if not qe or not qd or qe[0].get("obj") is None or qd[0].get("obj") is None: continue
        fe, be = arr(T, qe[0]["obj"]); fd, bd = arr(T, qd[0]["obj"])
        if len(fe) < 10 or len(fd) < 10: continue
        cnt["g5"] += 1
        common = sorted(set(fe) & set(fd))
        ie = {k: n for n, k in enumerate(fe)}; idd = {k: n for n, k in enumerate(fd)}
        io = float(np.median([iou(be[ie[k]], bd[idd[k]]) for k in common])) if common else 0.0
        if io >= 0.1: continue
        cnt["g6"] += 1
        exe, exd = excursion(be), excursion(bd)
        if exe >= 2.5 or exd <= 5: continue
        cnt["g7"] += 1
        vpx, win = local_speed_px(fd, bd, q["t"] * fps)
        if vpx is None: continue
        cnt["win"] += 1
        szd = size_px(bd)
        unit = q.get("unit") or "m/s"
        meas = vpx * fps * prior_m / szd / SI.get(unit.split("/")[0], 1.0)
        cs = pd.to_numeric(d.cs, errors="coerce")
        if pd.notna(cs) and cs > 0 and max(meas / cs, cs / meas) <= 2:
            val, how = float(np.sqrt(meas * cs)), "agree_gm"
        else:
            val, how = meas, "dis_geo"
        out[str(i)] = dict(value=round(val, 4), why=(f"rul013 {how}: e-track obj {qe[0]['obj']} static ({exe:.1f} sizes), d-track obj "
                           f"{qd[0]['obj']} moves ({exd:.1f}), IoU {io:.2f}; v {vpx:.2f}px/f x {fps:g}fps x {prior_m}m/{szd:.1f}px, win {win}"))
        if verbose:
            print(split, i, d.video_id, d["cat"], "|", str(raw.question[i])[:60], "| old", d.answer, d.answer_source, "->", out[str(i)])
    json.dump(out, open(f"{R}/cache/tmp/ag_c20s0alt_patch_{split}.json", "w"), indent=1)
    print(split, "gate counts", cnt, "changed", len(out))
    return out

if __name__ == "__main__":
    for s in ("test", "val"):
        run(s)
