#!/usr/bin/env python
import os, sys, json, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t061"; GT = sys.argv[2] if len(sys.argv) > 2 else "ffmv"
TAG = "c80mastlenS2"
_av = sys.argv; sys.argv = [_av[0], BASE]
_sp = importlib.util.spec_from_file_location("c80m", f"{R}/analysis/v3/ag/rul153/rule.py")
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C); sys.argv = _av
TDIRS = {"v2": ["_v2scSIL", "_v2sc", "_v2"], "v1": ["_v3ntmvSIL", "_v3ntmv", "", "_v1"]}

for split in ("test",):
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv", index_col=0)
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out, log = {}, []
    for i in det.index:
        if i not in P or i not in raw.index:
            continue
        pr = (P[i].get("priors") or [None])[0]
        if not pr or pr.get("kind") != "size" or pr.get("dim") != "length":
            continue
        ph = (P[i].get("prior_obj") or {}).get("phrase") or ""; tg = (P[i].get("target") or {}).get("phrase") or ""
        if not (C.MAST.search(ph) or C.MAST.search(tg)):
            continue
        v = str(P[i]["video_id"]); mode, mpp, ver = raw.loc[i, "meas_mode"], raw.loc[i, "meas_mpp"], str(raw.loc[i, "meas_tracks"])
        rec = dict(split=split, id=int(i), v=v, ph=ph, mode=mode, mpp=mpp, ver=ver)
        if mode not in ("2d_ratio", "vlm_2d_ratio") or not (pd.notna(mpp) and float(mpp) > 0):
            rec["why"] = "mode / mpp"; log.append(rec); continue
        hpx = float(pr["value_si"]) / float(mpp); best = None
        for suf in TDIRS.get(ver, ["_" + ver]):
            p = f"{R}/geo/out/tracks_{split}{suf}/{v}.json"
            if not os.path.exists(p):
                continue
            st = C.track_states(json.load(open(p)), ph)
            if not st or len(st) < 5:
                continue
            b = np.array([s["box"] for s in st], float); c = np.array([s["c"] for s in st], float)
            bw, bh = float(np.median(b[:, 2] - b[:, 0])), float(np.median(b[:, 3] - b[:, 1]))
            dx, dy = np.abs(np.diff(c[:, 0])).sum(), np.abs(np.diff(c[:, 1])).sum()
            cand = dict(suf=suf, n=len(st), bw=bw, bh=bh, hpx=hpx, dx=dx, dy=dy)
            if abs(bh / hpx - 1) < 0.03:
                best = cand; break
        rec["track"] = best
        if best is None:
            rec["why"] = "no own track reproduces the scale"; log.append(rec); continue
        g3, g4 = best["bh"] > 1.1 * best["bw"], best["dx"] > 3 * best["dy"]
        if not (g3 and g4):
            rec["why"] = f"g3 {g3} g4 {g4}"; log.append(rec); continue
        k = best["bh"] / best["bw"]; g = det.loc[i, "g"]
        kd = str(det.loc[i, "kind"]) if "kind" in det.columns else str(P[i]["q"].get("kind"))
        kind = "size" if kd == "size" else ("speed" if kd in C.SPEEDK else kd)
        if kind not in ("size", "speed") or not (pd.notna(g) and g > 0):
            rec["why"] = f"kind {kd} / g {g}"; log.append(rec); continue
        g2 = float(g) * k
        val, src = C.fuse(kind, g2, det.loc[i, "cs"])
        rec["why"] = f"PATCH {val:.4g}"; log.append(rec)
        out[str(int(i))] = dict(value=float(val), why=f"{TAG}: own length prior on own track box height h{best['bh']:.0f}/w{best['bw']:.0f} "
                                f"({best['suf']}); geo {g:.4g}*k{k:.3f}={g2:.4g} -> {src}; was {det.loc[i, 'answer']:.4g} ({det.loc[i, 'answer_source']})")
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
    print(f"[{split}] {TAG}: {len(out)} rows")
    for x in log:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != "track"}, x.get("track") and {k: (round(v, 1) if isinstance(v, float) else v) for k, v in x["track"].items()})
