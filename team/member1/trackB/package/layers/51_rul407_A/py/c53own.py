import os, sys, re, json
os.environ['QP_ROOT'] = os.path.abspath(os.environ["RA_Q"])
sys.path.insert(0, os.path.abspath(os.environ["RA_MODS"] + "/rul094"))
import numpy as np, pandas as pd
import rul094 as C
R = os.environ['QP_ROOT']; WO = os.path.abspath(os.environ["RA_OUT"])
SUB = sys.argv[1]
SF = 0.25
for split in ('test',):
    D = pd.read_csv(f"{R}/submissions/{SUB}_detail.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out = {}
    FW = json.load(open(f"{WO}/flatw2own_{split}.json"))
    for i in raw.index:
        p = P.get(i) or {}; q = p.get("q") or {}
        if q.get("kind") != "size": continue
        m = C.PRI.match(str(raw.ground_truth_prior[i]).lower())
        po = p.get("prior_obj") or {}
        pph = (po.get("phrase") or "").lower() if isinstance(po, dict) else ""
        if not m or not pph: continue
        asrc = str(D.answer_source.get(i, ""))
        fwg = None
        if asrc == "patch:rul291":
            fo = FW.get(str(i))
            if not fo: continue
            fwg = float(re.search(r"g'\s*([\d.eE+-]+)", fo["why"]).group(1))
        elif asrc.startswith("patch"): continue
        tph = ((p.get("target") or {}).get("phrase") or "").lower()
        if tph == pph: continue
        mpp = pd.to_numeric(raw.meas_mpp[i], errors="coerce"); gv = pd.to_numeric(raw.geo_value[i], errors="coerce")
        if not (pd.notna(mpp) and mpp > 0 and pd.notna(gv) and gv > 0) or str(raw.meas_mode[i]) != "2d_ratio": continue
        v = raw.video_id[i]
        prior_m = float(m.group(3)) * C.SI[m.group(4)]; px_old = prior_m / mpp
        ub = C.used_boxes(split, v, pph, str(raw.meas_tracks[i]))
        if not ub or len(ub) < 3: continue
        b = np.asarray(ub, float).reshape(-1, 4); c = (b[:, :2] + b[:, 2:]) / 2
        rng = float(np.linalg.norm(c.max(0) - c.min(0))); size = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
        if not rng < SF * size: continue
        ext = C.dim_px(b, m.group(1))
        if not (max(ext, px_old) / min(ext, px_old) <= 1.25): continue
        alts = [a for a in C.alt_tracks(split, v, pph) if a["ex"] >= 1.5]
        best = None
        for a in alts:
            io = float(np.median([max(C.iou(u, bb) for bb in a["b"]) for u in ub]))
            if io < 0.1: best = a; break
        if best is None: continue
        px_new = C.dim_px(best["b"][best["inb"]], m.group(1)); k = px_old / px_new
        if not (1.2 <= max(k, 1 / k) <= 3): continue
        gnew = (fwg if fwg is not None else gv) * k; cs = pd.to_numeric(D.cs.get(i), errors="coerce"); w = 0.7 if (D.loc[i, "cat"] == "D2" or fwg is not None) else 0.65
        if pd.notna(cs) and cs > 0 and abs(np.log(gnew / cs)) <= np.log(2): val, how = float(gnew ** w * cs ** (1 - w)), f"agree_wgm{w}"
        else: val, how = float(gnew), "dis_geo"
        out[str(i)] = dict(value=round(val, 4), why=f"c53own {how}: own used P '{pph}' static (range {rng:.0f}px < 0.25x{size:.0f}), ext {ext:.0f} vs ruler {px_old:.0f}px; moving {best['dir']}/{best['cfg']} {px_new:.1f}px exc {best['ex']:.2f} k {k:.3f}; geo {gv:.4g}->{gnew:.4g} CaW {cs}")
        print(split, i, v, D.loc[i, "cat"], D.answer_source[i], round(float(D.answer[i]), 4), '->', out[str(i)])
    json.dump(out, open(f"{WO}/c53own_{split}.json", "w"), indent=1)
    print(split, "rows", len(out), sorted(out))
