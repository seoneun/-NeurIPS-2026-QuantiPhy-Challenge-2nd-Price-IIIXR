#!/usr/bin/env python
import os, sys, json
import numpy as np, pandas as pd, cv2
R = os.path.abspath(os.environ.get("QP_ROOT") or "@QP_ROOT@")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul217"; WRITE = "write" in sys.argv[1:]
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}
SCENE = {"room", "ceiling", "wall", "the room", "the ceiling", "the wall"}
MIN_MOVE = 10.0
MIN_FRAC = 0.5


def obj_of(tr, phrase):
    ph = str(phrase or "").strip().lower()
    for qk in ("queries", "queries_d"):
        for q in tr.get(qk) or []:
            if str(q.get("phrase", "")).strip().lower() == ph and q.get("obj") is not None:
                return q["obj"]
    return None


def mover_speed(tr, obj):
    st = tr["objs"].get(str(obj)) or []; fps = float(tr.get("fps") or 24.0)
    k = np.array([j for j, s in enumerate(st) if s and s.get("st")])
    if len(k) < 0.6 * len(st) or len(k) < 5:
        return np.nan
    c = np.array([[(st[j]["st"]["box"][0] + st[j]["st"]["box"][2]) / 2, (st[j]["st"]["box"][1] + st[j]["st"]["box"][3]) / 2] for j in k])
    t = k / fps
    return float(np.hypot(np.polyfit(t, c[:, 0], 1)[0], np.polyfit(t, c[:, 1], 1)[0]))


def median_frame(sp, vid):
    cap = cv2.VideoCapture(f"{R}/data/{sp}_480p/{vid}.mp4"); F = []; k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k % 5 == 0:
            F.append(im)
        k += 1
    return np.median(np.stack(F), axis=0).astype(np.uint8) if F else None


def wall_edges(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)); H, W = g.shape
    w = max(40, W // 6); best = None
    for x0 in range(int(0.05 * W), int(0.95 * W) - w + 1, 10):
        p = np.median(gy[:, x0:x0 + w], axis=1); T = max(6.0, 0.1 * float(p.max()))
        smooth = p < T; run = None; s = None
        for y in range(H + 1):
            if y < H and smooth[y]:
                s = y if s is None else s
            elif s is not None:
                if run is None or y - s > run[1] - run[0]:
                    run = (s, y)
                s = None
        if run is None:
            continue
        top, bot = run
        if top < 8 or bot > H - 8:
            continue
        yc = int(top - 8 + np.argmax(p[top - 8:top])); yf = int(bot + np.argmax(p[bot:bot + 8]))
        if best is None or bot - top > best[4]:
            best = (x0, x0 + w, yc, yf, bot - top)
    return best


def rule(sp, debug=True):
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
    PP = {p["id"]: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    out = {}; cache = {}
    for i, r in A.iterrows():
        p = PP.get(i)
        if p is None:
            continue
        q = p["q"]
        if not (r["cat"] in ("S2", "D2") and q.get("kind") == "size" and q.get("dim") == "height" and q.get("unit") in UNIT
                and str(r.source) == "caw_nogeo" and str(r.meas_why).startswith("scene_target")
                and str((p.get("target") or {}).get("phrase") or "").strip().lower() in SCENE):
            continue
        pr = [x for x in p.get("priors") or [] if x.get("kind") == "speed" and x.get("value_si")]
        if len(pr) != 1 or not pr[0].get("obj"):
            continue
        pr = pr[0]
        fv = f"{R}/geo/out/tracks_{sp}_v3ntmv/{r.video}.json"
        if not os.path.exists(fv):
            continue
        tv = json.load(open(fv)); po = obj_of(tv, (p.get("prior_obj") or {}).get("phrase") or pr["obj"])
        if po is None:
            continue
        vpx = mover_speed(tv, po)
        if not (vpx >= MIN_MOVE):
            continue
        if r.video not in cache:
            im = median_frame(sp, r.video); cache[r.video] = (im, wall_edges(im) if im is not None else None)
        im, e = cache[r.video]
        if e is None:
            continue
        x0, x1, yc, yf, run = e; dy = yf - yc
        if dy < MIN_FRAC * im.shape[0]:
            continue
        scale = vpx / float(pr["value_si"])
        val = dy / scale / UNIT[q["unit"]]
        why = ('t6')
        out[i] = (float(val), why)
        if debug:
            os.makedirs(f"{R}/cache/audit/{TAG}", exist_ok=True)
            vis = im.copy(); W = vis.shape[1]
            cv2.line(vis, (x0, 0), (x0, vis.shape[0]), (255, 0, 0), 1); cv2.line(vis, (x1, 0), (x1, vis.shape[0]), (255, 0, 0), 1)
            cv2.line(vis, (0, yc), (W, yc), (0, 0, 255), 1); cv2.line(vis, (0, yf), (W, yf), (0, 200, 0), 1)
            panel = np.full((80, W, 3), 255, np.uint8)
            for n, s in enumerate([f"{i} {r.video} {str(r.question)[:70]}", f"red wall-ceiling y={yc}, green wall-floor y={yf}: {dy}px; mover {vpx:.1f}px/s / {pr['value_si']:.4g}m/s = {scale:.1f}px/m",
                                   f"value {val:.3f} {q['unit']}  (current answer {r.answer:.3g}); blue = column band x{x0}-{x1}"]):
                cv2.putText(panel, s, (6, 22 + 24 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(f"{R}/cache/audit/{TAG}/row_{i}.png", np.vstack([vis, panel]))
    return out


if __name__ == "__main__":
    res = {}
    for sp in ("test", "val"):
        res[sp] = rule(sp); print(sp, len(res[sp]), "rows")
        A = pd.read_csv(f"{R}/analysis/v3/calib/audit_{sp}.csv").set_index("id0")
        for i, (v, why) in res[sp].items():
            lab = f" | y {A.y[i]:.4g} MRA {A.mra[i]:.1f} -> {float(FC.row_mra(np.array([v]), np.array([A.y[i]]))[0]):.1f}" if sp == "val" else ""
            print(f"  {i} {A.video[i]} {A['cat'][i]} {str(A.question[i])[:55]}: {A.answer[i]:.4g} -> {v:.4g} ({A.source[i]}){lab}  [{why}]")
    if WRITE:
        os.makedirs(f"{R}/cache/tmp", exist_ok=True)
        for sp in res:
            json.dump({str(i): dict(value=round(float(v), 7), why=f"{TAG}: {why}") for i, (v, why) in res[sp].items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print('')
