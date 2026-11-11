"""rul205: coverage extension of rul094 (prior object asked as moving, geo's ruler track for it is static).
Gates (code only; no ids, no labels):
 1 prior key K = prior_obj.phrase|','.join(where). Static witness: a row of the video with meas_why containing 'static',
   parse target key == K and q.kind in {speed, avgspeed, accel, disp, path}.
 2 used prior track (raw meas_tracks: v1 -> tracks_<split> det_boxes, v2 -> tracks_<split>_v2sc queries_d[K]) has
   median centre excursion < 0.3 of its median side (it really is static).
 3 alternates: queries_d[K] in v3nt, v3, v3nt32 with >= 20 boxes, median per-frame IoU with used < 0.1,
   excursion (norm of centre range) >= 1.0 of own median side; all passing alternates agree on extent within 10%.
 4 px_new = median prior-dim extent (size/length -> max(w,h), height -> h, width -> w) over non-border frames of the first
   passing alternate; px_old = prior_SI / meas_mpp; |ln(px_old/px_new)| in [0.1, 0.7].
 5 rescale same-video SIZE rows, meas_mode in {2d_ratio, vlm_2d_ratio}, same meas_mpp, answer_source agree_*/dis_geo/dis_size_gm.
   g' = g*px_old/px_new; within 2x of CaW -> g'^w cs^(1-w) (w recovered from the row's own fusion, 0.5 for plain agree_gm);
   outside 2x -> g' in strong modes, unchanged in vlm modes.  Labels only evaluate."""
import json, os, re, sys, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C95_SUB", "sub_xt_Nzz29")
TAG = "rul205"
SI = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRI = re.compile(r"\s*(?:the )?(?:total )?(diameter|length|width|height|size) of (?:the )?(.+?)\s*=\s*([\d.]+)\s*(m|cm|mm)\s*$")
MOV = ("speed", "avgspeed", "accel", "disp", "path")
ALTS = ("v3nt", "v3", "v3nt32")


def key(o):
    if not isinstance(o, dict) or not o.get("phrase"): return None
    return o["phrase"].lower() + "|" + ",".join(o.get("where") or [])


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)


def boxes(T, K, cfg="queries_d"):
    for q in T.get(cfg) or []:
        if isinstance(q, dict) and q.get("key") == K and q.get("obj") is not None:
            return [s["st"]["box"] if s and s.get("st") else None for s in T["objs"].get(str(q["obj"])) or []]
    return None


def used(split, v, K, mt):
    if mt == "v1":
        p = f"{R}/geo/out/tracks_{split}/{v}.json"
        if not os.path.exists(p): return None
        T = json.load(open(p))
        for q in T.get("queries") or []:
            if q.get("key") == K: return [b if b else None for b in (q.get("det_boxes") or [])]
        return None
    p = f"{R}/geo/out/tracks_{split}_v2sc/{v}.json"
    if not os.path.exists(p): return None
    return boxes(json.load(open(p)), K)


def stats(bl):
    b = np.array([x for x in bl if x is not None], float).reshape(-1, 4)
    if len(b) == 0: return None
    c = (b[:, :2] + b[:, 2:]) / 2
    side = float(np.median(np.maximum(b[:, 2]-b[:, 0], b[:, 3]-b[:, 1])))
    d = np.linalg.norm(c - np.median(c, 0), axis=1)
    return dict(b=b, n=len(b), side=side, med_exc=float(np.median(d)) / max(side, 1e-9),
                rng_exc=float(np.linalg.norm(np.ptp(c, 0))) / max(side, 1e-9))


def dim_px(b, dim, W, H):
    inb = (b[:, 0] > 2) & (b[:, 1] > 2) & (b[:, 2] < W - 2) & (b[:, 3] < H - 2)
    if inb.sum() < 3: return None
    b = b[inb]; w, h = b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]
    return float(np.median(h if dim == "height" else w if dim == "width" else np.maximum(w, h)))


def run(split):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out = {}; log = []
    for v, g in raw.groupby("video_id"):
        wit = set()
        for i in g.index:
            p = P.get(i) or {}
            if (p.get("q") or {}).get("kind") in MOV and "static" in str(raw.meas_why[i]) and key(p.get("target")):
                wit.add(key(p.get("target")))
        if not wit: continue
        cache = {}
        for i in g.index:
            p = P.get(i) or {}; q = p.get("q") or {}
            K = key(p.get("prior_obj"))
            if K not in wit or q.get("kind") != "size": continue
            m = PRI.match(str(raw.ground_truth_prior[i]).lower())
            mode = str(raw.meas_mode[i]); src = str(D.answer_source.get(i, ""))
            mpp = pd.to_numeric(raw.meas_mpp[i], errors="coerce"); gv = pd.to_numeric(raw.geo_value[i], errors="coerce")
            if not m or mode not in ("2d_ratio", "vlm_2d_ratio") or not (pd.notna(mpp) and mpp > 0 and gv > 0):
                log.append(f"skip-mode {split} {i} {v} {mode}"); continue
            if not (src.startswith("agree_") or src in ("dis_geo", "dis_size_gm")):
                log.append(f"skip-src {split} {i} {v} {src}"); continue
            ck = (K, str(raw.meas_tracks[i]), m.group(1), round(float(mpp), 9))
            if ck not in cache:
                cache[ck] = None
                ub = used(split, v, K, str(raw.meas_tracks[i]))
                su = stats(ub) if ub else None
                if su is None or su["med_exc"] >= 0.3:
                    log.append(f"used-fail {split} {i} {v} {K} {su and round(su['med_exc'], 2)}"); continue
                passing = []
                for d in ALTS:
                    pth = f"{R}/geo/out/tracks_{split}_{d}/{v}.json"
                    if not os.path.exists(pth): continue
                    T = json.load(open(pth)); ab = boxes(T, K)
                    if not ab: continue
                    sa = stats(ab)
                    if sa is None: continue
                    ious = [iou(u, a) for u, a in zip(ub, ab) if u is not None and a is not None]
                    mi = float(np.median(ious)) if ious else 1.0
                    pxn = dim_px(sa["b"], m.group(1), T.get("W") or 1e9, T.get("H") or 1e9)
                    log.append(f"alt {split} {v} {K} {d} n{sa['n']} iou{mi:.2f} exc{sa['rng_exc']:.2f} px{pxn} (used med_exc {su['med_exc']:.3f})")
                    if sa["n"] >= 20 and mi < 0.1 and sa["rng_exc"] >= 1.0 and pxn: passing.append((d, pxn, sa["rng_exc"]))
                if not passing: continue
                ext = [x[1] for x in passing]
                if max(ext) / min(ext) > 1.10: log.append(f"alt-disagree {v} {ext}"); continue
                cache[ck] = passing[0]
            best = cache[ck]
            if best is None: continue
            px_old = float(m.group(3)) * SI[m.group(4)] / mpp; px_new = best[1]
            k = px_old / px_new
            if not (0.1 <= abs(np.log(k)) <= 0.7): log.append(f"k-fail {split} {i} {v} {k:.3f}"); continue
            gnew = gv * k; ans = float(D.answer[i]); cs = pd.to_numeric(D.cs.get(i), errors="coerce")
            if pd.notna(cs) and cs > 0 and abs(np.log(gnew / cs)) <= np.log(2):
                w = 0.5
                if src.startswith("agree_") and abs(np.log(gv / cs)) > 1e-6:
                    w = float(np.clip(np.log(ans / cs) / np.log(gv / cs), 0, 1))
                val, how = float(gnew ** w * cs ** (1 - w)), f"agree w{w:.2f}"
            elif mode == "2d_ratio" or not (pd.notna(cs) and cs > 0):
                val, how = float(gnew), "dis_geo"
            else:
                log.append(f"vlm-dis-unchanged {split} {i}"); continue
            out[str(i)] = dict(value=round(val, 4), why=f"{TAG} {how}: prior '{K}' ruler static ({px_old:.1f}px); moving "
                               f"{best[0]} track {px_new:.1f}px (exc {best[2]:.2f}), k {k:.3f}; geo {gv:.4g}->{gnew:.4g}, CaW {cs}")
            print(split, i, v, D.loc[i, "cat"], "|", raw.question[i][:60], "| old", round(ans, 3), src, "->", out[str(i)])
    print("\n".join(log))
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print(split, "changed", len(out))
    return out


if __name__ == "__main__":
    for s in ("test", "val"): run(s)
