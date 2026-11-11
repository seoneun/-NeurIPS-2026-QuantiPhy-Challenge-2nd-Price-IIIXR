"""rul094 v2 (final): the prior object P moves, but geo's ruler box for P is a static look-alike.

GATES (all code conditions on inputs; no ids):
 1 video V has a speed/avgspeed/accel row whose target phrase is P and whose geo meas_why contains 'static'
   (geo's track for P did not move although the question says P moves).
 2 the P boxes geo used (v1 det_boxes, or the v2/v2sc obj) have median IoU < 0.1 with a P track in v3nt or v2sc
   (queries or queries_d) that has >= 5 boxes, >= 3 border-free ('uncut') boxes and an excursion >= 1.5 of its own size.
 3 target rows: size rows of V, mode 2d_ratio, prior object P, target != P, valid meas_mpp and geo_value.
MEASURE: px_new = median prior-dimension extent of the moving track over uncut frames; px_old = prior / meas_mpp;
   k = px_old / px_new, require 1.2 <= max(k, 1/k) <= 3.  g' = geo * k (geo is linear in mpp; prior-echo rows too).
FUSION (as the pipeline): unpatched rows -> |log(g'/cs)| <= log 2: g'^w cs^(1-w) with w = 0.65 (agree_w size:0.65:strong),
   0.7 on D2 (--probe D2:agree_wgm:size:w0.7); else g' (dis_geo, strong mode); no CaW -> g'.
   rows patched by rul291 (its g' is also linear in mpp): recover rul291's g' from its value, rescale by k, re-run its own
   fusion (within 2x: g'^0.7 cs^0.3, else g').  Rows patched by anything else are left alone.
Labels only evaluate."""
import json, os, re, sys, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/rul094")
import rul094 as C           # prototype helpers (used_boxes, alt_tracks, dim_px, iou, PRI, SI, MOV); its run() is not called
SUB = os.environ.get("C53_SUB", "sub_xt_Nzz7")
WRITE = "--nowrite" not in sys.argv


def flatw2_gprime(split):
    p = f"{R}/cache/tmp/ag_flatw2_patch_{split}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def run(split):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    FW = flatw2_gprime(split)
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out = {}
    for v, g in raw.groupby("video_id"):
        statph = set()
        for i in g.index:
            q = (P.get(i) or {}).get("q") or {}
            ph = (((P.get(i) or {}).get("target") or {}).get("phrase") or "").lower()
            if q.get("kind") in C.MOV and ph and re.search(r"static", str(raw.meas_why[i])): statph.add(ph)
        if not statph: continue
        for i in g.index:
            p = P.get(i) or {}; q = p.get("q") or {}
            m = C.PRI.match(str(raw.ground_truth_prior[i]).lower())
            po = p.get("prior_obj") or {}
            pph = (po.get("phrase") or "").lower() if isinstance(po, dict) else ""
            if not m or pph not in statph or q.get("kind") != "size": continue
            tph = ((p.get("target") or {}).get("phrase") or "").lower()
            if tph == pph: continue
            src = str(D.answer_source.get(i, ""))
            if src.startswith("patch") and not (src == "patch:rul291" and str(i) in FW): continue
            mpp = pd.to_numeric(raw.meas_mpp[i], errors="coerce"); gv = pd.to_numeric(raw.geo_value[i], errors="coerce")
            if not (pd.notna(mpp) and mpp > 0 and pd.notna(gv) and gv > 0) or str(raw.meas_mode[i]) != "2d_ratio": continue
            prior_m = float(m.group(3)) * C.SI[m.group(4)]; px_old = prior_m / mpp
            ub = C.used_boxes(split, v, pph, str(raw.meas_tracks[i]))
            if not ub: continue
            best = None
            for a in C.alt_tracks(split, v, pph):
                if a["ex"] < 1.5: continue
                io = float(np.median([max(C.iou(u, bb) for bb in a["b"]) for u in ub]))
                if io < 0.1: best = a; break
            if best is None: continue
            px_new = C.dim_px(best["b"][best["inb"]], m.group(1))
            k = px_old / px_new
            if not (1.2 <= max(k, 1 / k) <= 3): continue
            cs = pd.to_numeric(D.cs.get(i), errors="coerce")
            if src == "patch:rul291":                         # rul291: value = g'^0.7 cs^0.3 (agree) or g' -> recover g'
                fv, fwhy = float(FW[str(i)]["value"]), FW[str(i)]["why"]
                g_old = (fv / cs ** 0.3) ** (1 / 0.7) if "wgm" in fwhy else fv
                gnew = g_old * k; w = 0.7; base = f"rul291 g' {g_old:.4g}"
            else:
                gnew = gv * k; w = 0.7 if D.loc[i, "cat"] == "D2" else 0.65; base = f"geo {gv:.4g}"
            if pd.notna(cs) and cs > 0 and abs(np.log(gnew / cs)) <= np.log(2):
                val, how = float(gnew ** w * cs ** (1 - w)), f"agree_wgm{w}"
            else:
                val, how = float(gnew), "dis_geo"
            out[str(i)] = dict(value=round(val, 4), why=f"rul094 {how}: prior '{pph}' ruler was a static look-alike ({px_old:.0f}px); "
                               f"moving {best['dir']}/{best['cfg']} obj {best['obj']} {px_new:.1f}px (exc {best['ex']:.2f}), k {k:.3f}; "
                               f"{base}->{gnew:.4g}, CaW {cs}")
            print(split, i, v, D.loc[i, "cat"], "|", raw.question[i][:60], "| old", round(float(D.answer[i]), 3), src, "->", out[str(i)])
    if WRITE: json.dump(out, open(f"{R}/cache/tmp/ag_c53pmove_patch_{split}.json", "w"), indent=1)
    print(split, "changed", len(out))
    return out


if __name__ == "__main__":
    for s in ("test", "val"): run(s)
