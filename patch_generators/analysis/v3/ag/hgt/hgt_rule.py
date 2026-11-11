"""ag hgt: height-above-surface rows that geo left unmeasured (CaW fallback).
Two code measurements, 2D rows only, scale = the video's own 2D mpp from the raw geo csv (other rows of the same video):
 (A) static "distance/height of X to/above the ground" (untimed): the ground point is the bottom of the tallest tracked
     support whose box spans the target horizontally and starts at or above the target's bottom (stand / pole under a
     bird or sign); value = (support_bottom - target_bottom) px * mpp.
 (B) "distance between (the bottom (corner) of) X and the floor at t" with a gravity prior, rows the pipeline failed on
     (target_no_track / unparsed): fall axis = the axis of the track's largest net displacement; leading edge = the box
     edge along that direction; floor = leading edge at first impact (first frame after the fall onset whose step drops
     below 25% of the previous step); value = (floor - lead(t)) px * mpp. Check: a quadratic fit of the fall gives g.
usage: python hgt_rule.py  -> cache/tmp/ag_hgt_patch_{test,val}.json"""
import json, os, re, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
RAWT = "geo_v4ntrj5sdGsrngsr_d_vw"


def tpath(split, v):
    for sub in ("_v2sc", "_v2", "_v3nt", ""):
        p = f"{R}/geo/out/tracks_{split}{sub}/{v}.json"
        if os.path.exists(p):
            return p


def load_tracks(split, v):
    p = tpath(split, v)
    if p is None:
        return None
    tj = json.load(open(p))
    fps = float(tj["fps"])
    t = np.array(tj["fidx"], float) / fps
    objs = {}
    for o, lst in tj["objs"].items():
        B = np.full((len(t), 4), np.nan)
        for k, x in enumerate(lst):
            if x is not None and x.get("st"):
                B[k] = x["st"]["box"]
        objs[o] = B
    keys = {}
    for q in tj["queries"]:
        if q.get("obj") is not None:
            keys.setdefault(q["key"].split("|")[0].strip(), str(q["obj"]))
    return dict(t=t, objs=objs, keys=keys, W=float(tj["W"]), H=float(tj["H"]))


def video_mpp(raw, v):
    s = raw[(raw.video_id == v) & raw.meas_mode.astype(str).str.startswith("2d_ratio") & raw.meas_mpp.notna()]
    if not len(s):
        return None, 0
    m = s.meas_mpp.astype(float)
    if m.max() / m.min() > 1.05:          # one scale per video expected for 2D rows
        return None, len(s)
    return float(m.median()), len(s)


def find_obj(T, phrase):
    ph = phrase.strip().lower()
    if ph in T["keys"]:
        return T["keys"][ph]
    hn = ph.split()[-1]
    c = [k for k in T["keys"] if k.split()[-1:] == [hn]]
    return T["keys"][c[0]] if len(c) == 1 else None


RX_A = re.compile(r"^what is the (?:distance|height) of (?:the )?([a-z][a-z \-]*?) (?:to|above|from) (?:the )?ground in (meters|cm)\??$")
RX_B = re.compile(r"\bdistance (?:between|of|from) (?:the )?(?:(?:bottom|lowest point|base)(?: corner)? of (?:the )?)?([a-z][a-z \-]*?) "
                  r"(?:and|to|from|above) the (?:floor|ground) at (\d+(?:\.\d+)?)\s*s\b")


def rule_A(T, tgt_o, mpp):
    B = T["objs"][tgt_o]
    tb = np.nanmedian(B, 0)
    if not np.all(np.isfinite(tb)):
        return None, "no_target_box"
    cx, bot = (tb[0] + tb[2]) / 2, tb[3]
    best = None
    for o, S in T["objs"].items():
        if o == tgt_o:
            continue
        sb = np.nanmedian(S, 0)
        if not np.all(np.isfinite(sb)):
            continue
        w, h = sb[2] - sb[0], sb[3] - sb[1]
        if w > 0.5 * T["W"] or h > 0.6 * T["H"] or sb[3] >= T["H"] - 2:     # ground / background boxes
            continue
        if not (sb[0] - 2 <= cx <= sb[2] + 2) or sb[1] > bot + 3 or sb[3] - bot < max(20, 3 * (tb[3] - tb[1])):
            continue
        if h < 1.5 * w:                                                      # a stand / pole is tall and narrow
            continue
        if np.nanstd(S[:, 3]) > 3:                                           # the support must be static
            continue
        if best is None or sb[3] > best[1]:
            best = (o, sb[3])
    if best is None:
        return None, "no_support"
    return (best[1] - bot) * mpp, f"support obj{best[0]} bottom y{best[1]:.0f} - target bottom y{bot:.0f}"


def rule_B(T, tgt_o, t_q, mpp):
    t, B = T["t"], T["objs"][tgt_o]
    ok = np.all(np.isfinite(B), 1)
    t, B = t[ok], B[ok]
    if len(t) < 10:
        return None, "short_track"
    C = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    d = C[-1] - C[0]
    ax = int(np.argmax(np.abs(d)))
    sgn = 1.0 if d[ax] > 0 else -1.0
    lead = sgn * (B[:, 2 + ax] if sgn > 0 else B[:, ax])                   # leading edge along the fall direction
    step = np.diff(lead)
    on = np.where(step > 3)[0]
    if not len(on):
        return None, "no_fall"
    k0 = on[0]
    imp = None
    for k in range(k0 + 1, len(step)):
        if step[k] < 0.25 * max(step[k - 1], 1e-6) and step[k - 1] > 3:
            imp = k; break
    if imp is None:
        return None, "no_impact"
    floor = lead[imp]
    kq = int(np.argmin(np.abs(t - t_q)))
    if abs(t[kq] - t_q) > 0.05 or not (k0 <= kq <= imp + 1 or kq < k0):
        return None, "t_out"
    v = (floor - lead[kq]) * mpp
    # fall check: quadratic fit of the leading edge between onset and impact -> g
    ff = slice(k0, imp + 1)
    tt, yy = t[ff], lead[ff] * mpp
    g = 2 * np.polyfit(tt, yy, 2)[0] if len(tt) >= 4 else np.nan
    lim = T["W"] if ax == 0 else T["H"]
    if not (6.5 <= g <= 13.0):                                              # not a clean free fall (throws, bounces)
        return None, f"fit_g={g:.1f}"
    if sgn * floor >= lim - 3 or sgn * floor <= 3:                          # impact cut by the frame edge
        return None, "floor_at_edge"
    return v, f"axis{'xy'[ax]}{'+' if sgn > 0 else '-'} floor={sgn*floor:.0f} (impact t={t[imp]:.2f}s) lead(t)={sgn*lead[kq]:.0f} fit_g={g:.1f}"


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv")
    out, log = {}, []
    for i, r in raw.iterrows():
        q = str(r.question).strip().lower()
        why = str(r.meas_why)
        if isinstance(r.depth_info, str) and r.depth_info.strip():
            continue                                                   # 2D rows only
        mA, mB = RX_A.search(q), RX_B.search(q)
        if not (mA or mB) or pd.notna(r.geo_value):
            continue
        T = load_tracks(split, r.video_id)
        mpp, nm = video_mpp(raw, r.video_id)
        if T is None or mpp is None:
            log.append((i, r.video_id, "no_tracks_or_mpp", q)); continue
        unit = 100.0 if (q.endswith("in cm?") or " in cm" in q) else 1.0
        if mA:
            o = find_obj(T, mA.group(1))
            if o is None:
                log.append((i, r.video_id, "no_target_track", q)); continue
            v, w = rule_A(T, o, mpp)
        else:
            if "gravity" not in str(r.ground_truth_prior).lower():
                continue
            o = find_obj(T, mB.group(1))
            if o is None:
                log.append((i, r.video_id, "no_target_track", q)); continue
            v, w = rule_B(T, o, float(mB.group(2)), mpp)
        if v is None or not np.isfinite(v) or v <= 0:
            log.append((i, r.video_id, w, q)); continue
        out[str(i)] = dict(value=float(v * unit), why=f"hgt_{'A' if mA else 'B'}: {w} * mpp {mpp:.5f} (n{nm})")
        log.append((i, r.video_id, f"{v*unit:.3f} | {w}", q))
    json.dump(out, open(f"{R}/cache/tmp/ag_hgt_patch_{split}.json", "w"), indent=1)
    print(f"== {split}: {len(out)} rows patched")
    for x in log:
        print(" ", x)
    return out


if __name__ == "__main__":
    for s in ("test", "val"):
        run(s)
