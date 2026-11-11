"""rul001 rule: a 2D scale taken from an ACCELERATION prior (non-gravity: 'acceleration of the X = a') is unreliable when the
prior object's track is cut by the frame border in the frames the acceleration fit uses (>= 20% of them), when that fit has
fewer than 8 frames, or when one quadratic explains the fitted positions poorly (R^2 < 0.95). Then geo is weak: rows whose
answer is geo-driven (plain fusion sources) take the fusion without geo, i.e. the CaW value cs.

Why: an acceleration is a second derivative of the centroid. While an object enters / leaves the frame, the centroid of its
visible part moves at about half the object's speed, and the switch to full speed when it is fully inside is a spurious
acceleration. The px acceleration (and with it the scale value / a_px) is then off by a factor ~2 or more, and every row of
the video inherits the error. Gravity priors are excluded: freefall() already demands a significant ballistic parabola with
consistent halves, and CaW's dynamic answers there are formula values (v = g t) that score 0 on labelled rows.

inputs: cache/tmp/ag_accguard_diag_{split}.csv (analysis/v3/ag/rul001/diag.py: tracks, parse, geo raw), the detail of the
base submission. writes cache/tmp/ag_accguard_patch_{test,val}.json"""
import sys, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nu"
CLIP, NMIN, R2MIN = 0.2, 8, 0.95
GEO_SRC = {"agree_gm", "agree_wgm", "dis_geo", "agree_geo_kind", "dis_motion_geo", "cqc_geo", "dis_size_gm"}


def flags(D):
    D = D.copy()
    ok = (D.grav == False) & D["mode"].astype(str).str.startswith("2d_ratio") & D.n.notna()
    D["f_clip"] = ok & (D["clip"] >= CLIP)
    D["f_short"] = ok & (D.n < NMIN)
    D["f_r2"] = ok & (D.r2 < R2MIN)
    D["flag"] = D.f_clip | D.f_short | D.f_r2
    D["why"] = [("prior accel fit unreliable: " + ",".join(w for w, f in (("border-cut %d/%d frames" % (nc, n), a), ("%d frames" % n, b), ("R2 %.3f" % r2, c)) if f))
                if (a or b or c) else "" for nc, n, r2, a, b, c in zip(D.nclip.fillna(0).astype(int), D.n.fillna(0).astype(int), D.r2, D.f_clip, D.f_short, D.f_r2)]
    return D


out = {}
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_accguard_diag_{split}.csv", index_col=0)
    D = flags(D)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    D = D.join(det[[c for c in ["question", "answer", "answer_source", "g", "cs"] if c in det]])
    P = {}
    for i, r in D[D.flag].iterrows():
        if str(r.answer_source) in GEO_SRC and np.isfinite(r.cs) and r.cs > 0 and abs(r.cs - r.answer) > 1e-9 * max(1.0, abs(r.answer)):
            P[str(int(i))] = dict(value=float(r.cs), why=r.why + " -> geo weak, CaW")
    json.dump(P, open(f"{R}/cache/tmp/ag_accguard_patch_{split}.json", "w"), indent=1)
    out[split] = (D, P, det)
    print(f"[{split}] candidate rows (non-gravity accel prior, 2d_ratio) {int(((D.grav == False) & D['mode'].astype(str).str.startswith('2d_ratio')).sum())}, "
          f"flagged {int(D.flag.sum())} (clip {int(D.f_clip.sum())}, short {int(D.f_short.sum())}, r2 {int(D.f_r2.sum())}), patched {len(P)}")

# ---------------------------------------------------------------- evaluation
L = labelled(SUB)
for split in ("val", "test"):
    D, P, det = out[split]
    ids = [int(k) for k in P]
    Ls = L[(L.split == split) & L.index.isin(ids)]
    print(f"labelled {split} rows changed: {len(Ls)}")
    if len(Ls):
        new = np.array([P[str(i)]["value"] for i in Ls.index])
        print(pd.DataFrame({"vid": Ls.video_id, "y": Ls.y, "old": Ls.answer, "new": new, "m_old": Ls.m, "m_new": A.row_mra(new, Ls.y.values)}).to_string())
# labelled rows inside the candidate set (controls) and whether they are flagged
for split in ("val", "test"):
    D = out[split][0]
    Ls = L[(L.split == split)]
    J = D.join(Ls[["y", "m", "mg", "mc"]], how="inner")
    print(f"labelled {split} rows in the accel/gravity 2d_ratio diagnostics: {len(J)}; flagged {int(J.flag.sum())}; videos {sorted(J.vid.unique())}; "
          f"MRA ans {J.m.mean():.3f} geo {J.mg.mean():.3f} caw {J.mc.mean():.3f}")

D, P, det = out["test"]
C = D[(D.grav == False) & D["mode"].astype(str).str.startswith("2d_ratio")].copy()
# label-free check 1: simulation kinematics (answer_source kin = exact propagation of the prior from rest) vs geo on the same rows
K = C[C.answer_source == "kin"].copy()
K["m_g_vs_kin"] = A.row_mra(K.g.values, K.answer.values)
print("\nkin rows (pseudo-truth = kinematic answer): geo MRA by flag")
print(K.groupby("flag").agg(n=("vid", "size"), videos=("vid", "nunique"), m_geo=("m_g_vs_kin", "mean")).round(3).to_string())
print(K[["vid", "question", "answer", "g", "cs", "clip", "n", "r2", "r2c", "flag", "m_g_vs_kin"]].round(3).to_string())
# label-free check 2: pseudo-labels (sub_v21 scene_prior / xrow_prior answers = priors of linked videos)
pl = pd.read_csv(f"{R}/submissions/sub_v21_detail.csv", index_col=0)
plx = pl.index[pl.answer_source.isin(["scene_prior", "xrow_prior"])]
Q = C[C.index.isin(plx)].copy(); Q["y"] = pl.answer.reindex(Q.index)
Q["m_g"] = A.row_mra(Q.g.values, Q.y.values); Q["m_c"] = A.row_mra(Q.cs.values, Q.y.values)
print("\npseudo-labelled rows in the candidate set:")
print(Q[["vid", "question", "y", "g", "cs", "clip", "flag", "m_g", "m_c"]].round(3).to_string())
# per flagged group: geo-vs-CaW ratio and row changes
print("\nflagged groups:")
F = C[C.flag]
print(F.groupby("vid").agg(win=("win", "first"), n=("n", "first"), clip=("clip", "first"), r2=("r2", "first"), r2c=("r2c", "first"), rows=("qk", "size"),
                           patched=("question", lambda s: int(sum(str(i) in P for i in s.index)))).to_string())
# rows changed by category, examples, and the self-ruler alternative (target's own CaW size / geo size)
ch = det.loc[[int(k) for k in P]].copy()
ch["new"] = [P[str(i)]["value"] for i in ch.index]; ch["why"] = [P[str(i)]["why"] for i in ch.index]
print("\ntest rows changed by category:", ch.cat.value_counts().to_dict())
sr = []
for i, r in ch.iterrows():
    tg = D.loc[i, "tgt"]; v = r.video_id
    sz = D[(D.vid == v) & (D.tgt == tg) & (D.qk == "size")]
    f_ = float(np.exp(np.mean(np.log(sz.cs / sz.g)))) if len(sz) and (sz.g > 0).all() else np.nan
    sr.append(r.g * f_ if D.loc[i, "qk"] != "size" else np.nan)
ch["selfruler"] = sr
print(ch[["video_id", "question", "answer_source", "answer", "new", "selfruler", "why"]].to_string())
