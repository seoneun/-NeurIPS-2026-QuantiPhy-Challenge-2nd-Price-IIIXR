"""rul016 (final): phase-restricted projectile questions under a gravity prior, 2D modes only.

rul016 branch - 'horizontal speed ... while it is in the air / when it leaves the table', 'horizontal distance /
displacement ... during its parabolic motion'.  geo measures the whole clip (ramp + rolling on the table + fall +
rolling on the floor; for 'distance' it even takes the 2D Euclidean first->last displacement).  Here the airborne
segment of the target track is found (strictly between the support level = median cy of the first n/6 frames and the
floor level = median cy of the last n/6, 8% margins, longest run >= 5 frames, drop >= 60 px), cy = c0 + c1 t + c2 t^2
is fitted there (R^2 > 0.995), takeoff = the vertex (vy = 0: roll-off / horizontal launch; it must precede the first
airborne frame), landing = the later root at the floor level; cx is fitted linearly (velocity residual < 15%).
The metres-per-pixel come from the INPUT gravity prior at the faller's own depth, mpp = g / (2 c2), which must lie
within 1.3x of geo's own mpp.   disp/dist = |vx| (t_land - t_takeoff) mpp;   hvel/speed = |vx| mpp.

c21drop branch - 'height of the table/desk/counter' under a gravity prior, 2D, when another tracked object passes the
same airborne fit and starts on that support (start bottom within 1.5 box heights of the support's top edge):
height = the faller's centre drop from its TAKEOFF level to the floor level = g (t_land - t_takeoff)^2 / 2 (centre is
one radius above the surface both on the table and on the floor).  The takeoff level is max(start level, vertex level)
in image y, so a faller that first rolls down a ramp standing on the table does not add the ramp height.

Final answers follow the pipeline fusion with the row's CaW value (cs, already prescaled as in the build):
agreement within 2x -> geometric mean (D2 strong size: g^0.7 c^0.3; geo kinds dist2/avgspeed: geo);
disagreement in a strong mode -> geo.  The fusion emulation is checked against the current answer with the old geo.
Fps comes from the track JSON (equals the CSV fps; the container fps is never used)."""
import json, re, sys, math, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/analysis/v3/ag/c21s3")
from trk import load, series

G = 9.8
PH = re.compile(r"(during (?:its|the) (?:parabolic|projectile) motion|while (?:it is )?(?:in the air|airborne)|"
                r"when it leaves (?:the )?\w+|during (?:its|the) (?:flight|fall))", re.I)
STRONG = ("2d_ratio", "3d_calib")


def air_fit(T, X, Y):
    n = len(T); k = max(3, n // 6)
    if n < 12: return None
    ys, yf = float(np.median(Y[:k])), float(np.median(Y[-k:]))
    if yf - ys < 60: return None
    m = (Y > ys + 0.08 * (yf - ys)) & (Y < yf - 0.08 * (yf - ys))
    idx = np.where(m)[0]
    if len(idx) < 5: return None
    runs = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1); idx = max(runs, key=len)
    if len(idx) < 5: return None
    t, x, y = T[idx], X[idx], Y[idx]
    c2, c1, c0 = np.polyfit(t, y, 2)
    if c2 <= 0: return None
    yhat = np.polyval([c2, c1, c0], t)
    r2 = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
    disc = c1 * c1 - 4 * c2 * (c0 - yf)
    if disc < 0: return None
    tl = (-c1 + math.sqrt(disc)) / (2 * c2)          # later root at the floor level = landing
    tv = -c1 / (2 * c2)                               # vertex = takeoff for a roll-off / horizontal launch
    if tv > t[0] or tl <= tv: return None
    yv = c0 - c1 * c1 / (4 * c2)
    d1, d0 = np.polyfit(t, x, 1)
    xr = x - np.polyval([d1, d0], t)
    vres = float(np.std(np.diff(xr) / np.diff(t)) / max(abs(d1), 1e-6)) if len(t) > 3 else 9.0
    return dict(n=len(idx), te=float(tv), tl=float(tl), a=float(2 * c2), r2=float(r2), vx=float(d1), vres=vres,
                ys=ys, yf=yf, yv=float(yv))


def fuse(g, cs, mode, w3, cat, kind):
    """pipeline fusion for one row (recipe d_r2_gm_c1w3 + geo_kinds dist2,avgspeed + agree_w size:0.65:strong, D2 0.7)"""
    if not (cs == cs and cs > 0): return g, "geo_cawinvalid"
    if abs(math.log(g / cs)) <= math.log(2) + 1e-12:
        if kind == "size" and mode in STRONG:
            w = 0.7 if cat == "D2" else 0.65
            return g ** w * cs ** (1 - w), "agree_wgm"
        if kind in ("dist2", "avgspeed"): return g, "agree_geo_kind"
        return math.sqrt(g * cs), "agree_gm"
    if mode in STRONG and not w3: return g, "dis_geo"
    return None, "dis_weak"          # CaW-routed disagreement: not emulated -> the rule does not fire


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")
    det = pd.read_csv(f"{R}/submissions/sub_xt_Nzn_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out, rep = {}, []

    def emit(rid, gnew, why, extra):
        dr = det.loc[rid]; pk = ((P.get(rid) or {}).get("q") or {}).get("kind") or str(dr.kind)
        kind = "size" if str(dr.kind) == "size" else pk
        cs, mode, w3, cat = float(dr.cs), str(dr["mode"]), bool(dr.w3) if dr.w3 == dr.w3 else False, str(dr["cat"])
        old_g = float(dr.g) if dr.g == dr.g else np.nan
        chk, chk_s = fuse(old_g, cs, mode, w3, cat, kind) if old_g == old_g else (None, "nogeo")
        same = chk is not None and abs(chk - float(dr.answer)) < 1e-6 * max(1, abs(dr.answer))
        fin, src = fuse(gnew, cs, mode, w3, cat, kind)
        rep.append((rid, dr.video_id, cat, kind, "old", round(float(dr.answer), 4), dr.answer_source, "old_g", round(old_g, 4),
                    "cs", round(cs, 4), "new_g", round(gnew, 4), "final", None if fin is None else round(fin, 4), src,
                    "fusion_reproduces_old", same) + extra)
        if fin is None or not same:
            return
        out[str(rid)] = {"value": float(fin), "why": f"{why}; fusion {src} with CaW {cs:.3g}"}

    for i, r in raw.iterrows():                       # rul016 branch
        rid = int(r["Unnamed: 0"]); q = str(r.question)
        if not PH.search(q): continue
        p = P.get(rid) or {}; pq = p.get("q") or {}; pri = (p.get("priors") or [{}])[0]
        if not pri.get("gravity") or "3d" in str(r.meas_mode): continue
        kind = str(r.meas_kind)
        if kind not in ("disp", "dist", "hvel", "speed", "path"): continue
        if not (pq.get("horizontal") or "horizontal" in q.lower()): continue
        d, j = load(r.video_id, split)
        if j is None: continue
        S = series(j); tgt = (p.get("target") or {}).get("phrase") or pq.get("target")
        key = next((k for k in S if k.split("|")[0] == tgt), None)
        if key is None: rep.append((rid, "notarget", tgt)); continue
        T, X, Y, W, H, B, o = S[key]
        f = air_fit(T, X, Y)
        if f is None: rep.append((rid, "nofit")); continue
        mpp = G / f["a"]; gm = float(r.meas_mpp) if pd.notna(r.meas_mpp) else np.nan
        ok = f["r2"] > 0.995 and f["vres"] < 0.15 and (np.isnan(gm) or 1 / 1.3 < mpp / gm < 1.3)
        v = abs(f["vx"]) * mpp * ((f["tl"] - f["te"]) if kind in ("disp", "dist", "path") else 1.0)
        v *= {"cm": 100.0, "mm": 1000.0}.get(str(pq.get("unit") or "m"), 1.0)
        extra = ("fps", j["fps"], "mpp", round(mpp, 5), "geo_mpp", round(gm, 5), "te", round(f["te"], 3), "tl", round(f["tl"], 3),
                 "n", f["n"], "r2", round(f["r2"], 5), "vres", round(f["vres"], 3), "vx_px", round(f["vx"], 1), "ok", ok)
        if not ok: rep.append((rid, "gate_fail") + extra); continue
        emit(rid, float(v), f"rul016: airborne t={f['te']:.3f}-{f['tl']:.3f}s, |vx|={abs(f['vx']):.0f}px/s, mpp=g/a={mpp:.5f}", extra)

    for i, r in raw.iterrows():                       # c21drop branch
        rid = int(r["Unnamed: 0"]); q = str(r.question).lower()
        m_ = re.search(r"height of the (table|desk|counter)[^a-z]", q)
        if not m_ or "3d" in str(r.meas_mode): continue
        p = P.get(rid) or {}; pri = (p.get("priors") or [{}])[0]
        if not pri.get("gravity"): continue
        d, j = load(r.video_id, split)
        if j is None: continue
        S = series(j); tk = next((k for k in S if k.split("|")[0] == m_.group(1)), None)
        if tk is None: continue
        Tt, Xt, Yt, Wt, Ht, Bt, ot = S[tk]; ttop = float(np.median(Yt - Ht / 2))
        for k, (T, X, Y, W, H, B, o) in S.items():
            if k == tk or (o is not None and o == ot): continue
            f = air_fit(T, X, Y)
            if f is None or f["r2"] < 0.995: continue
            hf = float(np.median(H)); bot0 = f["ys"] + hf / 2
            if abs(bot0 - ttop) > 1.5 * hf: continue
            mpp = G / f["a"]; gm = float(r.meas_mpp) if pd.notna(r.meas_mpp) else np.nan
            if not (np.isnan(gm) or 1 / 1.3 < mpp / gm < 1.3): continue
            ylo = max(f["ys"], f["yv"])               # takeoff level (image y grows downward)
            v = (f["yf"] - ylo) * mpp
            v *= {"cm": 100.0, "mm": 1000.0}.get(str(((p.get("q") or {}).get("unit")) or "m"), 1.0)
            extra = ("faller", k, "mpp", round(mpp, 5), "start_lvl", round(f["ys"], 1), "takeoff_lvl", round(f["yv"], 1),
                     "floor_lvl", round(f["yf"], 1), "drop_from_start", round((f["yf"] - f["ys"]) * mpp, 4),
                     "g_dt2_2", round(0.5 * G * (f["tl"] - f["te"]) ** 2, 4), "support_top", round(ttop, 1), "faller_bot0", round(bot0, 1))
            emit(rid, float(v), f"c21drop: {k.split('|')[0]} leaves the {m_.group(1)}; centre drop {f['yf'] - ylo:.0f}px x g/a={mpp:.5f}", extra)
            break
    json.dump(out, open(f"{R}/cache/tmp/ag_c21air_patch_{split}.json", "w"), indent=1)
    print(split, "patched", len(out))
    for x in rep: print("  ", x)


if __name__ == "__main__":
    for s in ("test", "val"): run(s)
