#!/usr/bin/env python
import os, sys, json, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.environ["RA_Q"]); WO = os.path.abspath(os.environ["RA_OUT"])
BASE = sys.argv[1]; GT = sys.argv[2]
TAG = "rul206"
_sp = importlib.util.spec_from_file_location("c95", f"{R}/analysis/v3/ag/rul205/rul205.py")
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
sys.path.insert(0, f"{R}/analysis/v3/compliance/rules"); import refuse
ALTS = ("v3nt",)


def run(split):
    D = pd.read_csv(f"{R}/submissions/{BASE}_detail.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {j["id"]: j for j in map(json.loads, open(f"{R}/geo/out/parse_{split}_v2.jsonl"))}
    out = {}; log = []; cache = {}
    for i in raw.index:
        p = P.get(i) or {}; q = p.get("q") or {}
        K = C.key(p.get("prior_obj"))
        if K is None or q.get("kind") != "size" or i not in D.index:
            continue
        v = raw.video_id[i]
        m = C.PRI.match(str(raw.ground_truth_prior[i]).lower())
        mode = str(raw.meas_mode[i]); src = str(D.answer_source[i])
        mpp = pd.to_numeric(raw.meas_mpp[i], errors="coerce"); gv = pd.to_numeric(raw.geo_value[i], errors="coerce")
        if not m or mode not in ("2d_ratio", "vlm_2d_ratio") or not (pd.notna(mpp) and mpp > 0 and gv > 0):
            continue
        if not (src.startswith("agree_") or src in ("dis_geo", "dis_size_gm")):
            continue
        ck = (v, K, str(raw.meas_tracks[i]), m.group(1), round(float(mpp), 9))
        if ck not in cache:
            cache[ck] = None
            ub = C.used(split, v, K, str(raw.meas_tracks[i]))
            su = C.stats(ub) if ub else None
            if su is None or su["med_exc"] >= 0.3:
                continue
            passing = []
            for d in ALTS:
                pth = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
                if not os.path.exists(pth): continue
                T = json.load(open(pth)); ab = C.boxes(T, K)
                if not ab: continue
                sa = C.stats(ab)
                if sa is None: continue
                ious = [C.iou(u, a) for u, a in zip(ub, ab) if u is not None and a is not None]
                mi = float(np.median(ious)) if ious else 1.0
                pxn = C.dim_px(sa["b"], m.group(1), T.get("W") or 1e9, T.get("H") or 1e9)
                log.append(f"alt {split} {i} {v} {K} {d} n{sa['n']} iou{mi:.2f} exc{sa['rng_exc']:.2f} px{pxn} (used med_exc {su['med_exc']:.3f})")
                if sa["n"] >= 20 and mi < 0.1 and sa["rng_exc"] >= 1.0 and pxn: passing.append((d, pxn, sa["rng_exc"]))
            if not passing: continue
            ext = [x[1] for x in passing]
            if max(ext) / min(ext) > 1.10: continue
            cache[ck] = passing[0]
        best = cache[ck]
        if best is None: continue
        px_old = float(m.group(3)) * C.SI[m.group(4)] / mpp; px_new = best[1]
        k = px_old / px_new
        if not (0.1 <= abs(np.log(k)) <= 0.7):
            log.append(f"k-fail {split} {i} {v} {k:.3f}"); continue
        gnew = float(gv) * k
        cs = pd.to_numeric(D.cs[i], errors="coerce")
        if not (pd.notna(cs) and cs > 0 and abs(np.log(gnew / cs)) <= np.log(2)):
            log.append(f"band-fail {split} {i} {v} gnew {gnew:.4g} cs {cs}"); continue
        val, how = refuse.fuse(D.loc[i], gnew, "size")
        if val is None:
            log.append(f"refuse-none {split} {i} {how}"); continue
        out[str(int(i))] = dict(value=float(val), why='t166')
        print(split, i, v, D.loc[i, "cat"], "|", str(raw.question[i])[:60], "| old", round(float(D.answer[i]), 3), src, "->", out[str(int(i))])
    print("\n".join(log))
    json.dump(out, open(f"{WO}/ag_{TAG}_patch_{split}.json", "w"), indent=0)
    print(f"===== {split}: patch {len(out)}")


run("test")
