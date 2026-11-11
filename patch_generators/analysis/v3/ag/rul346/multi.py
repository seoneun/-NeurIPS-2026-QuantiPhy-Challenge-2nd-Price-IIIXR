import sys, json, pickle, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"; sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
sys.path.insert(0, f"{R}/analysis/v3/ag/rul307"); import remeasure as RM; MV = RM.MV
B = pd.read_pickle(f"{R}/analysis/v3/ag/rul346/blk.pkl")
rng = np.random.default_rng(0)
print("(a) expected block gain if truth = new x exp(sigma z) (hypothesis right, geo noise sigma); 2000 draws")
for sg in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
    g = []
    for _ in range(2000):
        y = B.new.values * np.exp(sg * rng.standard_normal(len(B)))
        g.append(FC.row_mra(B.new.values, y).sum() - FC.row_mra(B.old.values, y).sum())
    print(f"   sigma {sg:.2f}: mean {np.mean(g):+.1f}  10% {np.quantile(g,.1):+.1f}  90% {np.quantile(g,.9):+.1f}")
print("    same, truth = old x exp(sigma z) (hypothesis wrong)")
for sg in (0.0, 0.15, 0.25, 0.3):
    g = [FC.row_mra(B.new.values, y).sum() - FC.row_mra(B.old.values, y).sum() for y in (B.old.values * np.exp(sg * rng.standard_normal((2000, len(B)))))]
    print(f"   sigma {sg:.2f}: mean {np.mean(g):+.1f}")
print("(b) 0035a vs 0035b: chain speed |v| (cm/s) and F at common times, per ball")
rr = [pickle.load(open(f"{R}/cache/tmp/ag_labmo_traj_test_{v}_prodP_q3.pkl", "rb")) for v in "vm"]
def prof(i):
    rec = rr[0][i]; c = rec["calls"][-1]; out = []
    for tq in np.arange(0.2, 1.85, 0.1):
        tr = RM._T(c["tt"], c["uv"], c.get("sg")); P = np.asarray(c["P"], float)
        try:
            v1 = MV.px_quantity(tr, "speed", round(tq, 2), None, None, None, P=P)
            P2 = P.copy(); P2[:, :2] *= 1.3
            v13 = MV.px_quantity(tr, "speed", round(tq, 2), None, None, None, P=P2)
            out.append((round(tq, 1), v1 * 100, v13 / v1))
        except Exception as e: out.append((round(tq, 1), np.nan, np.nan))
    return out
for ball, ia, ib in (("soccer", 2611, 2638), ("basketball", 2612, 2639)):
    pa, pb = prof(ia), prof(ib)
    print("  ", ball, " ".join(f"t{a[0]}: a {a[1]:.0f}/{a[2]:.2f} b {b[1]:.0f}/{b[2]:.2f} |" for a, b in zip(pa, pb)))
print("(c) S3 lab motion rows of block recordings not in the block")
A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv"); T = pd.read_pickle(f"{R}/analysis/v3/ag/rul346/tab.pkl"); T = T[T.split == "test"].set_index("id0")
b21 = pd.read_csv(f"{R}/submissions/sub_xt_best21v_id1.csv").parsed_value.values; b18 = pd.read_csv(f"{R}/submissions/sub_xt_best18v_id1.csv").parsed_value.values
tags = {}
import glob, os
for p in glob.glob(f"{R}/cache/tmp/ag_*_patch_test.json"):
    tg = os.path.basename(p)[3:-16]
    if tg.startswith("r2_") and tg != "rul345": continue
    try: js = json.load(open(p))
    except Exception: continue
    for k in js: tags.setdefault(int(k), []).append(tg)
recs = B.rec.unique()
A["rec"] = A.video.str.replace(r"(captured_\d+[ab]?)[sxX]?$", r"\1", regex=True)
sel = A[A.rec.isin(recs) & (A.cat == "S3") & A.kind.isin(["speed", "avgspeed", "disp", "path", "accel", "dist2"]) & ~A.index.isin(B.index)]
for i, r in sel.iterrows():
    F = T.F.get(i, np.nan)
    print(f"   {i} {r.video} {r.kind} scored={r.scored} src={r.source} ans={r.answer:.4g} b18={b18[i]:.4g} b21={b21[i]:.4g} g={r.g:.4g} F={F:.3f} tags={','.join(tags.get(i, []))[:80]} | {r.question[:60]}")
