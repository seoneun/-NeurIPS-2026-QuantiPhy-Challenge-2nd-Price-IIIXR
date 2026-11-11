"""R8 NEGPRIOR (text gate only).

A prior whose wording carries a parenthetical negation, "(not (the)? X)", tells us that a confusable
look-alike X sits in the scene next to the ruler object. Open-vocabulary detectors (GroundingDINO / OWLv2)
cannot parse negation, so the prior track is free to hop onto X; the 2D pixel ruler then has an unknown
scale and is no longer a strong measurement. Such rows are demoted from strong 2D (2d_ratio) to a weak
mode and re-fused with the pipeline's own weak-mode recipe (sub_xt_Nzg flags):
  agree within 2x  -> gm(g, cs)                      (agree_gm; dist2/avgspeed agreements keep geo: --geo_kinds)
  disagree, size   -> gm(g, cs)                      (--dis_size_gm all)
  disagree, other  -> CaW cs                         (dis_caw_routed; --dis_motion speed is 3D-only, --caw_shrink is size-only)
Gate inputs: prior text + mode + g/cs only (no CV / presence gates). Rows already patched are skipped.
Writes cache/tmp/ag_negprior_patch_{test,val}.json with FINAL answers.
"""
import json, re, sys
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = "sub_xt_Nzg"; GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
RX = re.compile(r"\(\s*not\s+(?:the\s+)?[^)]*\)", re.I)
GEO_KINDS = {"dist2", "avgspeed"}


def kind_of(pj):
    return (pj.get("q") or {}).get("kind")


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = j
    mode = raw["meas_mode"].reindex(det.index).fillna("").astype(str)
    out, log = {}, []
    for i in det.index:
        pr = str(raw.at[i, "ground_truth_prior"])
        if not RX.search(pr):
            continue
        src = str(det.at[i, "answer_source"])
        if src.startswith("patch"):
            log.append((i, "skip: already patched")); continue
        if mode[i] != "2d_ratio":
            log.append((i, f"skip: mode {mode[i] or 'none'}")); continue
        g = pd.to_numeric(raw.at[i, "geo_value"], errors="coerce")
        cs = pd.to_numeric(det.at[i, "cs"], errors="coerce") if "cs" in det else np.nan
        if not (np.isfinite(g) and g > 0 and np.isfinite(cs) and cs > 0):
            log.append((i, "skip: no g or cs")); continue
        k = kind_of(P.get(int(i), {}))
        agree = abs(np.log10(g / cs)) <= np.log10(2)
        if agree:
            v, why = (g, "agree_geo_kind") if k in GEO_KINDS else (float(np.sqrt(g * cs)), "agree_gm")
        elif k == "size":
            v, why = float(np.sqrt(g * cs)), "dis_size_gm"
        else:
            v, why = float(cs), "dis_caw_routed"
        old = float(det.at[i, "answer"])
        if abs(np.log10(v / old)) < 1e-6:
            log.append((i, "unchanged")); continue
        m = RX.search(pr).group(0)
        out[str(i)] = {"value": float(v), "why": f"rul314: prior '{m}' -> 2d_ratio weak; {why} (g={g:.3g}, c={cs:.3g})"}
        log.append((i, f"{old:.4g} -> {v:.4g} {why}"))
    json.dump(out, open(f"{R}/cache/tmp/ag_negprior_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] gate rows {len(log)}; changed {len(out)}")
    for i, s in log:
        print(f"  {i} {det.at[i, 'video_id']} {det.at[i, 'cat']} | {str(det.at[i, 'question'])[:80]} | {s}")


if __name__ == "__main__":
    for sp in ("test", "val"):
        run(sp)
