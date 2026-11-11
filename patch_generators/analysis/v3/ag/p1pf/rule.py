"""P1 rule: prior identity collapse -> gated person focal (n == 1 allowed under the gate). Writes the FINAL-answer patch.

Pipeline (all label-free; run in this order):
  collapse.py   prior d-track == a differently named object's track (IoU > 0.8, conflict.phrase_conflict) -> prior-calibrated
                focal invalid for that video
  gate.py       the person focal resting on one estimate is accepted only when its box is a standing non-rider
                (no bicycle / motorcycle overlap, h/w >= 2) and every other stated-distance frame with one such box agrees
                within 15%
  meas.py       measure_v3 re-run (read-only import, production env / args) on the P1 videos = collapse AND gate, with the
                gated focal in FPERSON (ref run reproduces production exactly)
  build_geo.py  production geo build + re-measured rows (personf_merge refinements a, c) -> preds/*_geo_v4ntrj5sQ2p1pf*
  submit        the xt_Nze command on that geo (main + u build, mo_guard) -> cache/tmp/ag_p1pf_sub_p1pf_*
  rule.py       FINAL answers:
                - motion rows: the pipeline's answer on the re-measured geo (sub_p1pf);
                - D3 size rows whose production answer used the collapsed-calibration geo: the CaW route (d3size_caw =
                  full-video CaW x d3size_scale), since the rule invalidates that geo;
                - rows already patched (patch:*) and rows of the rul323 / rul432 / rul324 patches are skipped.
usage: python analysis/v3/ag/p1pf/rule.py   -> cache/tmp/ag_p1pf_patch_{test,val}.json"""
import sys, json, re, math
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A  # noqa: E402
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 64)
BASE = "sub_xt_Nze"
SKIP_TAGS = ("rul323", "rul432", "rul324")
HGT = re.compile(r"\b(?:height|tall)\b", re.I)
SCALE = {"simulation": (1.1, 1.05)}          # --d3size_scale simulation:1.1:-height,simulation:1.05:+height
CAWR = {"dis_caw_routed", "caw_nogeo", "caw_c1"}
GEOR = {"dis_geo", "dis_motion_geo", "agree_geo_kind"}
GEO_INDEP = re.compile(r"^(d3size_caw|scene_prior|xrow_prior|caw_nogeo|caw_c1|dis_caw_routed|patch:)")

out = {}
for split in ("test", "val"):
    kd = "detail" if split == "test" else "valdetail"
    d0 = pd.read_csv(f"{R}/submissions/{BASE}_{kd}.csv").set_index("Unnamed: 0")
    d1 = pd.read_csv(f"{R}/cache/tmp/ag_p1pf_sub_p1pf_{kd}.csv").set_index("Unnamed: 0")
    rq = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    for c_ in ("question", "video_id"):
        if c_ not in d0:
            d0[c_] = rq[c_].reindex(d0.index)
    if "kind" not in d0:
        d0["kind"] = rq.meas_kind.reindex(d0.index)
    skip = set()
    for t in SKIP_TAGS:
        try:
            skip |= {int(k) for k in json.load(open(f"{R}/cache/tmp/ag_{t}_patch_{split}.json"))}
        except FileNotFoundError:
            pass
    B = pd.read_csv(f"{R}/cache/tmp/ag_p1pf_buildlog.csv")
    B = B[(B.split == split) & (B.u.fillna("") == "") & B.how.str.match(r"^[mv]: ")]
    full = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_caw9b_orig.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs() \
        if split == "test" else pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_caw9b.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs()
    P, rows = {}, []
    PQ = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        x_ = json.loads(ln); PQ[int(x_["id"])] = x_
    # check the CaW-route formula on every production d3size_caw simulation row
    chk = d0[(d0.answer_source == "d3size_caw") & d0.video_id.astype(str).str.startswith("simulation")]
    if len(chk):
        k_ = np.where(chk.question.astype(str).str.contains(HGT), SCALE["simulation"][1], SCALE["simulation"][0])
        okf = np.isclose(chk.answer.astype(float), full.reindex(chk.index) * k_, rtol=1e-6)
        print(f"[{split}] CaW-route formula reproduces {int(okf.sum())}/{len(chk)} production d3size_caw simulation rows")
    cand = set(B.id.astype(int)) | set(d0.index[~np.isclose(d0.answer.astype(float), d1.answer.astype(float).reindex(d0.index), rtol=1e-9, equal_nan=True)])
    for i in sorted(cand):
        r0, r1 = d0.loc[i], d1.loc[i]
        s0 = str(r0.answer_source)
        if s0.startswith("patch") or i in skip:
            rows.append((i, r0.video_id, r0["cat"], r0.kind, s0, r0.answer, None, "skip (patched)")); continue
        if r0["cat"] == "D3" and str(r0.kind) == "size":
            if GEO_INDEP.match(s0):
                rows.append((i, r0.video_id, r0["cat"], r0.kind, s0, r0.answer, None, "size answer did not use geo: keep")); continue
            typ = str(r0.video_id).split("_")[0]
            kk = SCALE.get(typ, (1.0, 1.0))[1 if HGT.search(str(r0.question)) else 0]
            v = float(full.get(i, np.nan)) * kk
            if not (np.isfinite(v) and v > 0):
                rows.append((i, r0.video_id, r0["cat"], r0.kind, s0, r0.answer, None, "no CaW: keep")); continue
            how = f"D3 size: collapsed-prior geo invalid -> CaW route (full CaW {full[i]:.3g} x {kk})"
        else:
            v = float(r1.answer)
            how = f"re-measured with gated person focal: g {r0.g:.3g}->{r1.g:.3g}, {s0}->{r1.answer_source}"
            q_ = (PQ.get(int(i)) or {}).get("q") or {}
            if (str(r1.answer_source) in CAWR and s0 in GEOR and q_.get("kind") in ("avgspeed", "speed") and q_.get("t") is not None
                    and q_.get("t1") is None and q_.get("t2") is None and np.isfinite(r1.g) and r1.g > 0):
                # 'average speed of X at 2.5s' is a speed AT a time point (parse t set, no interval): --dis_motion speed,3d,nofov
                # keeps 3D motion geo on a CaW disagreement for such rows (sibling 'speed of X at 2.5s' rows get dis_motion_geo);
                # the weak 3d_personf label alone must not move it to CaW
                v = float(r1.g)
                how = f"re-measured with gated person focal: g {r0.g:.3g}->{r1.g:.3g}; speed at t={q_['t']}s -> 3D motion geo kept (dis_motion), not {r1.answer_source}"
        if abs(v / float(r0.answer) - 1) < 1e-6:
            rows.append((i, r0.video_id, r0["cat"], r0.kind, s0, r0.answer, v, "unchanged")); continue
        P[str(int(i))] = dict(value=float(v), why=how)
        rows.append((i, r0.video_id, r0["cat"], r0.kind, s0, r0.answer, v, how))
    json.dump(P, open(f"{R}/cache/tmp/ag_p1pf_patch_{split}.json", "w"), indent=1)
    out[split] = (d0, P)
    T = pd.DataFrame(rows, columns=["id", "vid", "cat", "kind", "src", "old", "new", "how"])
    print(f"===== {split}: candidate rows {len(T)}, patched {len(P)}")
    if len(T):
        T["q"] = d0.question.reindex(T.id).str[:60].values
        print(T.to_string(index=False))
    if split == "test" and len(P):
        print("changed by category:", d0.loc[[int(k) for k in P]].cat.value_counts().to_dict())

# ---------------------------------------------------------------- evaluation 1: labelled rows (val + all-237)
from lab_rows import labelled  # noqa: E402
L = labelled(BASE)
L = L.reset_index().rename(columns={"index": "rid", "Unnamed: 0": "rid"})
sel = [n for n in range(len(L)) if str(int(L.rid.iloc[n])) in out[L.split.iloc[n]][1]]
print("labelled rows changed:", len(sel))
if sel:
    Z = L.iloc[sel].copy()
    Z["new"] = [out[s][1][str(int(i))]["value"] for s, i in zip(Z.split, Z.rid)]
    Z["m_new"] = A.row_mra(Z.new.values, Z.y.values)
    print(Z[["rid", "split", "video_id", "cat", "answer_source", "y", "answer", "new", "m", "m_new"]].to_string())
    L2 = L.copy(); L2.loc[L2.index[sel], "m"] = Z.m_new.values
    print("all-237 macro", A.macro(L.m, L.cat)[0], "->", A.macro(L2.m, L2.cat)[0])
    V, V2 = L[L.split == "val"], L2[L2.split == "val"]
    print("val macro", A.macro(V.m, V.cat)[0], "->", A.macro(V2.m, V2.cat)[0])

# ---------------------------------------------------------------- evaluation 2: scene template (val simulation_0060 labels)
# simulation_0058-0062 are views of one animation: val simulation_0060 carries labels for the same objects. Labels score only.
dv = pd.read_csv(f"{R}/preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
dv["y"] = pd.to_numeric(dv.ground_truth_posterior, errors="coerce")
tv = dv[dv.video_id == "simulation_0060"]
KEYS = [("speed", "bicycle"), ("length", "turning yellow car"), ("height", "turning yellow car"), ("diameter", "wheel"),
        ("speed", "pedestrian"), ("height", "pedestrian"), ("width", "green car"), ("width", "purple car"),
        ("length", "white car"), ("width", "lane"), ("speed", "yellow car")]


def tkey(q):
    q = q.lower()
    for a_, b_ in KEYS:
        if a_ in q and b_ in q:
            return (a_, b_)
    return None


tmpl = {tkey(q): y for q, y in zip(tv.question, tv.y) if tkey(q)}
d0, P = out["test"]
rows = []
for k, x in P.items():
    i = int(k); q = str(d0.question[i]); kk = tkey(q)
    y = tmpl.get(kk) if kk else None
    if y is None:
        rows.append((i, d0.video_id[i], q[:58], d0.answer[i], x["value"], np.nan, np.nan, np.nan)); continue
    m0, m1 = A.row_mra(np.array([d0.answer[i]]), np.array([y]))[0], A.row_mra(np.array([x["value"]]), np.array([y]))[0]
    rows.append((i, d0.video_id[i], q[:58], d0.answer[i], x["value"], y, m0, m1))
E = pd.DataFrame(rows, columns=["id", "vid", "q", "old", "new", "y_tmpl", "m_old", "m_new"])
print("===== scene-template evaluation (val simulation_0060 labels; 'speed of the yellow car at 2.5s' vs the val AVERAGE speed is approximate)")
print(E.to_string(index=False))
e = E.dropna(subset=["y_tmpl"])
print(f"template rows {len(e)}: sum MRA {e.m_old.sum():.2f} -> {e.m_new.sum():.2f} (delta {e.m_new.sum() - e.m_old.sum():+.2f} row-units); "
      f"without the yellow-car speed rows: delta {(e[~e.q.str.contains('yellow car at')].m_new - e[~e.q.str.contains('yellow car at')].m_old).sum():+.2f}")
