import os, json, math, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def _cache(name, r):
    p = f"{os.environ.get('R14_CACHE', HERE + '/cache')}/{name}_{r['split']}_{int(r['id'])}.json"
    return json.load(open(p)) if os.path.exists(p) else None


USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6,
       "m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001, "m/s2": 1.0, "cm/s2": 0.01}


def _keys(p):
    tk = {}
    for e in p.get("depth") or []:
        if e.get("t") is None or e.get("d") is None:
            continue
        tk.setdefault(e["key"], {})[float(e["t"])] = float(e["d"])
    return tk


def _ok(v):
    try:
        return v is not None and np.isfinite(float(v)) and float(v) > 0
    except Exception:
        return False


SLOPE = re.compile(r"slope|ramp|incline")
ENDQ = re.compile(r"\b(?:end|bottom|foot) of the (?:slope|ramp|incline|track)")


def kinu(p, var="V1"):
    q = p.get("q") or {}; unit = q.get("unit"); qt = str(p.get("question", "")).lower()
    if q.get("kind") != "speed" or unit not in USI:
        return None
    acc = [pr for pr in (p.get("priors") or []) if pr.get("kind") in ("accel", "acceleration") and not pr.get("gravity") and _ok(pr.get("value_si"))]
    if len(acc) != 1:
        return None
    a = float(acc[0]["value_si"])
    sk = sorted({e["key"] for e in (p.get("depth") or []) if e.get("key") and SLOPE.search(e["key"])})
    if not sk:
        return None
    tk = _keys(p)
    cand = [(p.get("llm") or {}).get("target_depth_key")] + list(p.get("target_dkeys") or []) + [str(q.get("target") or "").strip().replace(" ", "_")]
    key = next((k for k in cand if k and k in tk and not SLOPE.search(k)), None)
    if key is None:
        return None
    ks = sorted(tk[key].items())
    if q.get("t") is not None:
        if var == "V2":
            return None
        tq = float(q["t"]); db = [d for t, d in ks if abs(t - tq) < 1e-6]
        if not db:
            return None
        fl = max([math.sqrt(2 * a * abs(db[0] - d)) for t, d in ks if t < tq - 1e-6] or [0.0])
        why = f"KINU: slope scene, own accel prior {a:g}, own key '{key}' key pairs before t={tq:g} -> v >= sqrt(2a|dd|) = {fl:.4f} m/s"
    elif ENDQ.search(qt):
        if len(ks) < 2:
            return None
        d0 = ks[0][1]; dirn = np.sign(ks[-1][1] - d0)
        if dirn == 0:
            return None
        ends = [(abs(e["d"] - d0), e["key"]) for e in (p.get("depth") or [])
                if e.get("key") in sk and e.get("t") is None and e.get("d") is not None and (e["d"] - d0) * dirn > 0]
        if not ends:
            return None
        dd, ke = max(ends)
        fl = math.sqrt(2 * a * dd)
        why = f"KINU: slope end '{ke}' in the motion direction of own key '{key}', |d_end-d_first|={dd:.4f} m, own accel prior {a:g} -> v_end >= {fl:.4f} m/s"
    else:
        return None
    if not fl > 0:
        return None
    return fl / USI[unit], why


def radf(p):
    q = p.get("q") or {}; k = q.get("kind"); unit = q.get("unit")
    if k not in ("disp", "path", "avgspeed", "dist2") or unit not in USI:
        return None
    tk = _keys(p)
    if k == "dist2":
        if q.get("t") is None:
            return None
        t = float(q["t"]); ll = p.get("llm") or {}; a = ll.get("target_depth_key"); b = ll.get("target2_depth_key")
        if a not in tk or b not in tk or a == b:
            return None
        da = [d for tt, d in tk[a].items() if abs(tt - t) < 1e-6]; db = [d for tt, d in tk[b].items() if abs(tt - t) < 1e-6]
        if not da or not db:
            return None
        fl = abs(da[0] - db[0]); why = f"RADF: |d_{a} - d_{b}| at t={t:g} = {fl:.4f} m"
    else:
        tg = p.get("target_dkeys") or []
        if len(tg) != 1 or tg[0] not in tk:
            return None
        t1, t2 = q.get("t1"), q.get("t2")
        if t1 is None or t2 is None:
            return None
        t1, t2 = float(t1), float(t2)
        ks = sorted((t, d) for t, d in tk[tg[0]].items() if t1 - 1e-6 <= t <= t2 + 1e-6)
        if len(ks) < 2 or t2 <= t1:
            return None
        if k == "disp":
            if abs(ks[0][0] - t1) > 1e-6 or abs(ks[-1][0] - t2) > 1e-6:
                return None
            fl = abs(ks[-1][1] - ks[0][1])
        else:
            fl = sum(abs(b[1] - a[1]) for a, b in zip(ks, ks[1:]))
            if k == "avgspeed":
                fl = fl / (t2 - t1)
        why = f"RADF: own key '{tg[0]}' radial travel ({k}) >= {fl:.4f} SI"
    if not fl > 0:
        return None
    return fl / USI[unit], why


def _raise(r, res):
    if res is None:
        return None
    v, why = res; x = float(r["cur"])
    if not (np.isfinite(v) and v > 0) or v <= x * (1 + 1e-9):
        return None
    return float(v), why


def KINU_V1(r):
    return _raise(r, kinu(r.get("parse") or {}, "V1"))


def RADF_V1(r):
    return _raise(r, radf(r.get("parse") or {}))


def GFALL_U7(r):
    c = _cache("rul297", r)
    return (float(c["value"]), c.get("why", "")) if c else None


def FFRE(r):
    c = _cache("ffre", r)
    return (float(c["value"]), c.get("why", "")) if c else None


def VLB_U8(r):
    c = _cache("vlb", r)
    if not c:
        return None
    fl = float(c["floor"])
    return (fl, "rmap3 U8 vlb raise-only: " + c.get("why", "")) if fl > float(r["cur"]) * 1.05 else None


def DZLB_V2(r):
    p = r.get("parse") or {}; q = p.get("q") or {}; ans = float(r["cur"])
    if q.get("kind") != "dist2" or not _ok(ans):
        return None
    t = q.get("t"); ka, kb = set(p.get("target_dkeys") or []), set(p.get("target2_dkeys") or [])
    if t is None or not ka or not kb:
        return None
    da = [d["d"] for d in p.get("depth") or [] if d.get("key") in ka and d.get("t") is not None and abs(float(d["t"]) - float(t)) < 1e-6 and d.get("d")]
    db = [d["d"] for d in p.get("depth") or [] if d.get("key") in kb and d.get("t") is not None and abs(float(d["t"]) - float(t)) < 1e-6 and d.get("d")]
    if len(da) != 1 or len(db) != 1:
        return None
    u = {"m": 1.0, "cm": 100.0, "mm": 1000.0}.get(q.get("unit") or "m")
    if u is None:
        return None
    dz = abs(float(da[0]) - float(db[0])) * u; c = r.get("s1_cs")
    if not _ok(dz):
        return None
    if ans < dz * (1 - 1e-9):
        return float(dz), f"U11 dzlb: distance >= own depth-text |dZ| {dz:.4g} at t={t}; answer {ans:.4g} -> {dz:.4g}"
    if _ok(c) and abs(float(c) / dz - 1) <= 0.15 and ans > 3 * dz:
        v = float(np.sqrt(dz * ans))
        return v, f"U2/U11 dzlb V2: own CaW {float(c):.4g} = own depth-text |dZ| {dz:.4g}; answer {ans:.4g} > 3|dZ| -> sqrt -> {v:.4g}"
    return None


ACCEPTED = [("r14_U7_rul297", GFALL_U7), ("r14_rul106", FFRE), ("r14_KINU_V1", KINU_V1), ("r14_U8_vlb", VLB_U8),
            ("r14_RADF_V1", RADF_V1), ("r14_U2U11_dzlb", DZLB_V2)]
