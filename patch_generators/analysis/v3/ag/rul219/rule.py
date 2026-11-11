"""rul219: whole-clip displacement of a constant-speed object that leaves the frame = v x T.

Rows: 2D simulation / segmentation videos, question 'displacement (or distance travelled) of X' with NO time window, prior
'speed of X = v' (time-less, not windowed) on the same object (same SAM2 track), current answer from the pipeline's kin vT
path (answer_source 'kin': v x T(geo), i.e. the geo displacement in the prior's own scale).
Problem: geo's displacement runs only from the first to the last VISIBLE frame of the track. When the object passes through
the frame and leaves it (the track starts or ends with a box cut by the image border, and its visible span is shorter
than the clip), the truncated track - plus border-cut centroids that move at about half speed - under-reads the
displacement (shark: 10.5 m of geo for 7 m/s over a 1.96 s clip).
Rule: if the in-frame uncut run shows constant velocity (robust CV of 0.25 s chord speeds <= 0.05, first/second-half speed
ratio within 10%, straight line within max(2 px, 1% of the object size)) and the object moves toward the border it is cut
at when it leaves (away from the one it enters through), the whole-clip displacement is v x T with
T = (n_frames - 1) / fps (first to last frame; n = frames of the 480p file, fps = the dataset's fps column, never the
container rate). The value replaces the kin value (the pipeline takes kin rows as is: --kin sim:grav:before:vT, no gm).
Not QP_DSPAN (that restricted the time span of 3D size extents); this changes only whole-clip displacement rows.
inputs: cache/tmp/ag_clipdisp_diag_{split}.csv (analysis/v3/ag/rul219/diag.py), submissions/{SUB}_detail / _valdetail.csv
writes cache/tmp/ag_clipdisp_patch_{test,val}.json"""
import sys, json, re, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 60)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzh"
CV_MAX = float(os.environ.get("CD_CV", "0.05"))
HALF = float(os.environ.get("CD_HALF", "0.10"))
REQ_KIN = os.environ.get("CD_REQ_KIN", "1") == "1"
SAME = os.environ.get("CD_SAME", "1") == "1"      # 0: also other targets, v = in-frame run px speed x meas_mpp (diagnostic only)
SPAN_FR = 3            # visible span shorter than the clip by more than 3 frames
NMIN, DMIN, DISPF = 6, 0.25, 0.3
TIME_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?)\b|\bstart\b|\bbetween\b|\bduring\b|\bafter\b|\bbefore\b|\buntil\b|\bwhen\b", re.I)
WIN_RE = re.compile(r"\b(?:at|after|before|between|during|from|until|when)\b|\d+(?:\.\d+)?\s*s\b", re.I)


def dir_ok(side, vx, vy, leaving):
    """the object moves toward the cut border when leaving it, away from it when entering"""
    sg = 1 if leaving else -1
    ok = True
    for c in side:
        if c == "L": ok &= sg * vx < 0
        if c == "R": ok &= sg * vx > 0
        if c == "T": ok &= sg * vy < 0
        if c == "B": ok &= sg * vy > 0
    return bool(ok)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def evaluate(split, D, det):
    rows, patch = [], {}
    for i, r in D.iterrows():
        g = dict(id=i, vid=r.vid)
        src = str(r.get("vsrc", "")).lower()
        s_first = r.s_first if isinstance(r.s_first, str) else ""
        s_last = r.s_last if isinstance(r.s_last, str) else ""
        n, fps = fnum(r.n), fnum(r.fps)
        T = (n - 1) / fps
        span = fnum(r.get("t_last")) - fnum(r.get("t_first"))
        ans_src = str(det.answer_source.get(i, ""))
        g.update(cat=r["cat"], qk=r.qk, ans=det.answer.get(i), src=ans_src, s_first=s_first, s_last=s_last, span=round(span, 3), T=round(T, 3),
                 cv=r.get("cv_run"), half=r.get("half_ratio"))
        fail = []
        if str(r.vt)[1] != "2": fail.append("not2D")
        if not src.startswith(("simulation", "segment")): fail.append("not_sim")
        if r.qk not in ("disp", "path"): fail.append("kind")
        if TIME_RE.search(str(det["question"].get(i, r.question) if "question" in det else r.question)): fail.append("q_time")
        same_obj = bool(r.same) and str(r.get("same_tr")) == "True"
        if r.pkind != "speed" or np.isfinite(fnum(r.pt)) or np.isfinite(fnum(r.pt1)) or WIN_RE.search(str(r.praw).split("=")[0]):
            fail.append("prior_not_timeless_speed")
        if SAME and not same_obj: fail.append("other_object")
        if not str(r.get("mode", "")).startswith("2d_ratio"): fail.append("mode")
        if ans_src.startswith("patch"): fail.append("patched")
        if REQ_KIN and ans_src != "kin": fail.append("not_kin")
        if not (np.isfinite(span) and span < T - SPAN_FR / fps - 1e-9 and (s_first or s_last)): fail.append("in_frame_whole_clip")
        run_ok = (fnum(r.get("run_n")) >= NMIN) and (fnum(r.get("run_t1")) - fnum(r.get("run_t0")) >= DMIN - 1e-9) and (fnum(r.get("run_disp")) >= DISPF * r["size"])
        if not run_ok: fail.append("short_run")
        if not (fnum(r.get("cv_run")) <= CV_MAX and abs(fnum(r.get("half_ratio")) - 1) <= HALF and fnum(r.get("lin_res")) <= max(2.0, 0.01 * r["size"])):
            fail.append("not_constant")
        vx, vy = fnum(r.get("lin_vx")), fnum(r.get("lin_vy"))
        if (s_last and not dir_ok(s_last, vx, vy, True)) or (s_first and not dir_ok(s_first, vx, vy, False)): fail.append("direction")
        g["fail"] = ",".join(fail)
        if not fail:
            v_si = float(r.vsi) if same_obj else float(r.chord_run) * fnum(r.mpp)
            val = v_si * T
            # label-free consistency: the in-frame run's px speed over the whole clip vs frame width + object length
            g.update(new=val, px_clip=round(float(r.chord_run) * T, 1), W_plus_L=round(float(r.W) + float(r["size"]), 1), geo_px=round(float(r.px_disp), 1))
            patch[str(int(i))] = dict(value=val, why=f"rul219: track visible {r.t_first:.2f}-{r.t_last:.2f}s of 0-{T:.2f}s, "
                                      f"cut {s_first or '-'}/{s_last or '-'}; in-frame run {r.run_t0:.2f}-{r.run_t1:.2f}s constant "
                                      f"(cv {r.cv_run:.3f}, halves {r.half_ratio:.2f}); v {v_si:g} m/s x T {T:.3f}s (was {ans_src} {det.answer.get(i):.3f})")
        rows.append(g)
    return (pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()), patch


out = {}
tag = os.environ.get("CD_TAG", "")
for split in ("test", "val"):
    D = pd.read_csv(f"{R}/cache/tmp/ag_clipdisp_diag_{split}.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    E, patch = evaluate(split, D, det)
    out[split] = (E, patch, det)
    if not tag:
        json.dump(patch, open(f"{R}/cache/tmp/ag_clipdisp_patch_{split}.json", "w"), indent=1)
    print(f"== {split}: {len(D)} whole-clip disp/path rows, patched {len(patch)}")
    if len(E):
        print(E.round(3).to_string())
    for k, v in patch.items():
        print(" ", k, v)

# labelled evaluation (val + test copies of val questions = all-237)
L = labelled(SUB)
print("\n== labelled rows changed")
for split in ("val", "test"):
    patch = out[split][1]
    ids = [int(k) for k in patch]
    Ls = L[(L.split == split) & L.index.isin(ids)]
    print(split, "labelled changed:", len(Ls))
    if len(Ls):
        new = np.array([patch[str(i)]["value"] for i in Ls.index])
        print(pd.DataFrame(dict(y=Ls.y, m_old=Ls.m, m_new=A.row_mra(new, Ls.y.values))).to_string())
print("labelled whole-clip disp/path rows at all:", int(L.index.isin(out["val"][0].index if len(out["val"][0]) else []).sum()), "(val)",
      int(((L.split == "test") & L.index.isin(out["test"][0].index)).sum()), "(test copies)")
E = out["test"][0]
if len(E):
    ch = E[E.fail == ""]
    print("\ntest rows changed by cat:", ch.cat.value_counts().to_dict())
