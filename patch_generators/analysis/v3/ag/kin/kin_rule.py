"""kin start-from-rest check -> patch.

The submit step answers "speed of X at t" under "acceleration of X = a" with a*t (constant acceleration from rest at t = 0).
Here the prior object's own track tests that assumption. In 2D the test is scale-free: fit x(t), y(t) with quadratics,
v(t) = v0 + A t, and r = |v(t_q)| / (|A| t_q) is the factor by which the true speed differs from a*t (pixels cancel; only
the prior's own acceleration sets the scale, so the corrected answer is a * t_q * r).

A row is changed only when the evidence is unanimous:
  * three fit windows (the motion segment containing t_q, the track up to t_q + 0.4 s, the whole track up to the prior's
    window end) each fit with R^2 >= 0.99, and each says |v(0)| > 0.15 a t_q AND the zero-speed time ts is >= 1.5 frames
    away from 0 with the same sign in all three (sub-frame offsets are the 1-frame label convention, not motion);
  * the three r agree within 10 % (else the "motion" is a segmentation / wing-beat artefact);
  * the track is not cut by the image border before t_q (an object entering the frame moves its mask centroid at a
    fraction of its speed) and the camera is static: phase-correlation motion < 2 % of the width, or another tracked
    object of the video stays within 3 px (phase correlation on rendered scenes follows the moving object);
  * 2D videos only (D2 / S2): in 3D scenes perspective bends pixel trajectories, and with only two depth keyframes the
    depth is interpolated linearly, which itself fakes a constant radial speed (simulation_0300, whose labelled val copy
    is from rest, reads 'moving at t=0' in 2D and 'starting at 0.4-0.6 s' in 3D).
Output: cache/tmp/ag_kin_patch_{test,val}.json  {row_id: {"value", "why"}}, value = a * t_q * median(r) (question unit)."""
import sys, os, json
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/kin")
import numpy as np, pandas as pd
import kin_fit as KF
import kinematic as K

split = KF.split
det, raw = KF.det, KF.raw
WINS = ("seg", "early", "all")


def decide(z, fps, gates=True):
    """returns (flag, r_med, reason) for one row of kin_fit output (gates=False: fits only, for the risk report)"""
    if z.get("cat") not in ("D2", "S2") and gates:
        return False, np.nan, "3d_scene"
    if not np.isfinite(z.get("n", np.nan)):
        return False, np.nan, "no_track"
    if z.get("clip0", 0) > 0.2 and gates:
        return False, np.nan, "border_clipped"
    if z.get("cam", 0) > 0.02 and not z.get("static_ref", 0) >= 1 and gates:
        return False, np.nan, "camera_unknown"
    rs, zs, tss = [], [], []
    for w in WINS:
        if not np.isfinite(z.get(f"{w}_r", np.nan)) or z.get(f"{w}_r2", 0) < 0.99:
            return False, np.nan, f"fit_{w}"
        rs.append(z[f"{w}_r"]); zs.append(z[f"{w}_z"]); tss.append(z[f"{w}_ts"])
    tmin = 1.5 / fps
    if not all(zz > 0.15 for zz in zs):
        return False, float(np.median(rs)), "from_rest"
    if not (all(t_ <= -tmin for t_ in tss) or all(t_ >= tmin for t_ in tss)):
        return False, float(np.median(rs)), "ts_disagree"
    if max(rs) / min(rs) > 1.10:
        return False, float(np.median(rs)), "r_disagree"
    return True, float(np.median(rs)), "moving_at_0" if tss[0] < 0 else "late_start"


def main():
    Z = KF.run(det.index[det.answer_source == "kin"])
    patch, rep = {}, []
    for i, z in Z.iterrows():
        fl, rm, why = decide(z.to_dict(), float(z["fps"]))
        fl2, rm2, why2 = decide(z.to_dict(), float(z["fps"]), gates=False)
        rep.append(dict(id=i, vid=z["vid"], cat=z["cat"], tq=z["tq"], a=z["a"], kin=z["kin"], g=z["g"], y=z.get("y", np.nan), cam=z.get("cam"),
                        sref=z.get("static_ref"), clip=z.get("clip0"), r=rm, why=why, flag=fl, r_nogate=rm2, why_nogate=why2))
        if fl:
            v = float(z["kin"] * rm)
            ts_ = float(z["seg_ts"])
            patch[str(int(i))] = dict(value=v, why=f"kin_rest_check: prior object already moving at t=0 (2D quadratic fits: "
                                                f"v(t_q) = {rm:.2f} x a t_q, zero-speed time {ts_:+.2f} s)")
    P = pd.DataFrame(rep).set_index("id")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(P.round(3).to_string())
    print(P.why.value_counts().to_string())
    out = f"{R}/cache/tmp/ag_kin_patch_{split}.json"
    json.dump(patch, open(out, "w"), indent=1)
    print(f"[{split}] {len(patch)} rows changed -> {out}")
    for k, v in patch.items():
        print(k, v)


if __name__ == "__main__":
    main()
