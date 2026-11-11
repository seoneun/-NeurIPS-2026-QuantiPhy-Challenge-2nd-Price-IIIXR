"""rul016: phase-restricted projectile questions under a gravity prior (2D).
'horizontal displacement/distance ... during its parabolic motion', 'horizontal speed ... while it is in the air',
'horizontal speed ... when it leaves the table'. geo measures the whole clip (rolling on the table + fall + rolling on
the floor; 317 even takes the 2D Euclidean first->last displacement). Here: find the airborne segment of the target
track (between the support level and the floor level), fit y = c0 + c1 t + c2 t^2 there (a_px = 2 c2) and x linear;
metres-per-pixel from gravity at the faller's own depth: mpp = 9.8 / a_px.  Answers:
  disp/dist : |vx| (t_land - t_takeoff) mpp     hvel/speed : |vx| mpp
Gates: gravity prior, 2D mode, phase phrase, >= 5 airborne frames, drop >= 60 px, fit R^2 > 0.995,
mpp within 1.3x of geo's own mpp (same fall), constant x-speed in flight (rel. resid < 15%)."""
import json, os, re, sys, numpy as np, pandas as pd
R = "/data/iiixr/users/sehong/quantiphy"
sys.path.insert(0, R + "/analysis/v3/ag/c21s3")
from trk import load, series
PH = re.compile(r"(during (?:its|the) (?:parabolic|projectile) motion|while (?:it is )?(?:in the air|airborne)|"
                r"when it leaves (?:the )?\w+|during (?:its|the) (?:flight|fall))", re.I)
def air_fit(T, X, Y):
    n = len(T); k = max(3, n // 6)
    ys, yf = np.median(Y[:k]), np.median(Y[-k:])
    if yf - ys < 60: return None
    m = (Y > ys + 0.08 * (yf - ys)) & (Y < yf - 0.08 * (yf - ys))
    idx = np.where(m)[0]
    if len(idx) < 5: return None
    runs = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1); idx = max(runs, key=len)
    if len(idx) < 5: return None
    t, x, y = T[idx], X[idx], Y[idx]
    c2, c1, c0 = np.polyfit(t, y, 2)
    if c2 <= 0: return None
    yhat = np.polyval([c2, c1, c0], t); r2 = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
    def roots(yv):
        disc = c1 * c1 - 4 * c2 * (c0 - yv)
        if disc < 0: return None
        return sorted([(-c1 - np.sqrt(disc)) / (2 * c2), (-c1 + np.sqrt(disc)) / (2 * c2)])
    tv = -c1 / (2 * c2)                       # vertex (apex for a toss; for a roll-off ~ takeoff)
    rs, rf = roots(ys), roots(yf)
    if rf is None: return None
    # horizontal launch / roll-off: vy = 0 at takeoff -> takeoff is the vertex (must precede the first airborne frame)
    if tv > t[0]: return None
    te = tv
    tl = rf[1]
    d1, d0 = np.polyfit(t, x, 1)
    xr = x - np.polyval([d1, d0], t); vres = np.std(np.diff(xr) / np.diff(t)) / max(abs(d1), 1e-6) if len(t) > 3 else 9
    return dict(n=len(idx), te=te, tl=tl, a=2 * c2, r2=r2, vx=d1, vres=vres, ys=ys, yf=yf)
def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j["id"]] = j
    out = {}; rep = []
    for i, r in raw.iterrows():
        rid = int(r["Unnamed: 0"]); q = str(r.question)
        if not PH.search(q): continue
        p = P.get(rid) or {}; pq = p.get("q") or {}; pri = (p.get("priors") or [{}])[0]
        if not pri.get("gravity"): continue
        if "3d" in str(r.meas_mode): continue
        kind = str(r.meas_kind)
        if kind not in ("disp", "dist", "hvel", "speed", "path"): continue
        if not (pq.get("horizontal") or "horizontal" in q.lower()): continue
        d, j = load(r.video_id, split)
        if j is None: continue
        S = series(j); tgt = (p.get("target") or {}).get("phrase") or pq.get("target")
        key = next((k for k in S if k.split("|")[0] == tgt), None)
        if key is None: continue
        T, X, Y, W, H, B, o = S[key]
        f = air_fit(T, X, Y)
        if f is None: rep.append((rid, "nofit")); continue
        mpp = 9.8 / f["a"]; gm = float(r.meas_mpp) if pd.notna(r.meas_mpp) else np.nan
        ok = f["n"] >= 5 and f["r2"] > 0.995 and f["vres"] < 0.15 and (np.isnan(gm) or 1 / 1.3 < mpp / gm < 1.3) and f["tl"] > f["te"]
        if kind in ("disp", "dist", "path"): v = abs(f["vx"]) * (f["tl"] - f["te"]) * mpp
        else: v = abs(f["vx"]) * mpp
        unit = str(pq.get("unit") or "m")
        v = v * (100 if unit == "cm" else 1)
        rep.append((rid, r.video_id, kind, round(v, 4), "geo", round(float(r.geo_value), 4), "mpp", round(mpp, 5), "geo_mpp", gm,
                    "te", round(f["te"], 3), "tl", round(f["tl"], 3), "r2", round(f["r2"], 5), "vres", round(f["vres"], 3), "ok", ok))
        if ok:
            out[str(rid)] = {"value": float(v), "why": f"rul016: airborne fit t={f['te']:.2f}-{f['tl']:.2f}s, vx={abs(f['vx']):.0f}px/s, mpp=g/a={mpp:.5f}"}
    # c21drop: 'height of the table/desk/counter' under a gravity prior when a tracked object rolls off that support:
    # support height = the faller's centre drop (support level -> floor level) x g/a_px at the faller's depth
    for i, r in raw.iterrows():
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
            if k == tk or o is not None and o == ot: continue
            f = air_fit(T, X, Y)
            if f is None or f["r2"] < 0.995 or f["n"] < 5: continue
            hf = float(np.median(H)); bot0 = f["ys"] + hf / 2
            if abs(bot0 - ttop) > 1.5 * hf or f["yf"] - f["ys"] < 60: continue
            mpp = 9.8 / f["a"]; v = (f["yf"] - f["ys"]) * mpp
            rep.append((rid, r.video_id, "drop", k, round(v, 4), "geo", round(float(r.geo_value), 4), "mpp", round(mpp, 5), "support_top", round(ttop, 1), "faller_bot0", round(bot0, 1)))
            out[str(rid)] = {"value": float(v), "why": f"c21drop: {k.split('|')[0]} rolls off the {m_.group(1)}; centre drop {f['yf']-f['ys']:.0f}px x g/a={mpp:.5f}"}
            break
    json.dump(out, open(f"{R}/cache/tmp/ag_c21air_patch_{split}.json", "w"), indent=1)
    print(split, "patched", len(out))
    for x in rep: print("  ", x)
for s in ("test", "val"): run(s)
