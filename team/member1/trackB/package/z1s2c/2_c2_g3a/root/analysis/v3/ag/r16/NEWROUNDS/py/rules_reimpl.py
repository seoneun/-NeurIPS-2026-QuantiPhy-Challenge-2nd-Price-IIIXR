import math, re, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6,
       "m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001, "m/s2": 1.0, "cm/s2": 0.01}


def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) and x > 0 else None
    except Exception:
        return None


def _q(r):
    return ((r.get("parse") or {}).get("q")) or {}


def _kind(r):
    k = _q(r).get("kind")
    return str(k if k else (r.get("geo") or {}).get("meas_kind"))


def _med(a):
    a = [x for x in (a or []) if _f(x)]
    return float(np.median(a)) if a else None


def _lmean(a):
    a = [x for x in (a or []) if _f(x)]
    return float(np.exp(np.mean(np.log(a)))) if a else None


def _same(a, b, tol=1e-6):
    return a is not None and b is not None and abs(math.log(a / b)) < tol


def _geo_meas(r):
    g = r.get("geo") or {}
    if not bool(g.get("meas_valid")):
        return None
    return _f(g.get("meas_value"))


def _rul263(r):
    return r.get("cat") == "D3" and _kind(r) == "size"


def _r8_pool(r, srcs, var):
    if r.get("s1_src") not in srcs:
        return None
    full = [x for x in r.get("caw_full") or [] if _f(x)]
    crop = [x for x in r.get("caw_crop") or [] if _f(x)]
    if var == "V1":
        v = _med(full)
    else:
        v = _lmean(full + crop)
        if v is not None and var == "V3":
            v = math.sqrt(v * r["cur"])
    return (v, f"r8 own CaW pooling {var}") if v else None


def r8_U1_V1(r): return _r8_pool(r, ("rul263_caw",), "V1") if _rul263(r) else None
def r8_U1_V2(r): return _r8_pool(r, ("rul263_caw",), "V2") if _rul263(r) else None
def r8_U1_V3(r): return _r8_pool(r, ("rul263_caw",), "V3") if _rul263(r) else None
R8_U2 = ("caw_nogeo", "dis_caw_routed", "caw_c1")
def r8_U2_V1(r): return _r8_pool(r, R8_U2, "V1")
def r8_U2_V2(r): return _r8_pool(r, R8_U2, "V2")
def r8_U2_V3(r): return _r8_pool(r, R8_U2, "V3")


def _r8_u3(r, var):
    g, f, c, cur = _f(r.get("s1_g")), _f(r.get("caw_full_greedy")), _f(r.get("caw_crop_greedy")), _f(r.get("cur"))
    if not (g and f and c and cur):
        return None
    if var == "V2" and not r.get("caw_crop"):
        return None
    if var == "V3" and not re.match(r"^(agree_|dis_|caw_|rul263_)", str(r.get("s1_src"))):
        return None
    est = {"g": g, "f": f, "c": c}
    for k, v in est.items():
        if abs(math.log(cur / v)) > 0.01:
            continue
        a, b = [est[x] for x in est if x != k]
        if abs(math.log(a / b)) <= math.log(1.25) and abs(math.log(cur / a)) > math.log(2) and abs(math.log(cur / b)) > math.log(2):
            return math.sqrt(a * b), f"r8 U3 majority of own estimates (cur={k})"
    return None


def r8_U3_V1(r): return _r8_u3(r, "V1")
def r8_U3_V2(r): return _r8_u3(r, "V2")
def r8_U3_V3(r): return _r8_u3(r, "V3")






def _A_u2(r, src):
    if not _rul263(r): return None
    c, cur = _f(r.get("s1_cs")), _f(r.get("cur"))
    full = [x for x in r.get("caw_full") or [] if _f(x)]
    if not (c and cur) or not _same(cur, c) or len(full) < 8: return None
    if src and r.get("video_source") != src: return None
    return float(np.median(full)), "A U2 own full CaW median"


def A_U2a(r): return _A_u2(r, None)
def A_U2b(r): return _A_u2(r, "lab")
def A_U2c(r): return _A_u2(r, "simulation")


def _A_u3(r, var):
    if not _rul263(r): return None
    c, cur, g = _f(r.get("s1_cs")), _f(r.get("cur")), _geo_meas(r)
    m = str((r.get("geo") or {}).get("meas_mode"))
    if not (c and cur and g) or not _same(cur, c) or m not in ("3d_calib", "3d_calib_tdepth=prior", "3d_personf"): return None
    if not (0.5 <= g / cur <= 2): return None
    if var == "b":
        full = [x for x in r.get("caw_full") or [] if _f(x)]
        if len(full) < 4 or np.percentile(full, 75) / np.percentile(full, 25) <= 1.3: return None
    if var == "c" and r.get("video_source") != "simulation": return None
    return math.sqrt(cur * g), "A U3 gm(cur, own geo)"


def A_U3a(r): return _A_u3(r, "a")
def A_U3b(r): return _A_u3(r, "b")
def A_U3c(r): return _A_u3(r, "c")


def _A_u4(r, var):
    if not _rul263(r): return None
    c, cur = _f(r.get("s1_cs")), _f(r.get("cur"))
    full = [x for x in r.get("caw_full") or [] if _f(x)]
    if not (c and cur) or len(full) < 8 or _same(cur, c): return None
    q25, q75 = np.percentile(full, [25, 75]); med = float(np.median(full))
    if q75 / q25 > (1.10 if var == "b" else 1.15) or abs(math.log(c / med)) > math.log(1.1): return None
    if var == "c":
        m = str((r.get("geo") or {}).get("meas_mode")); g = _geo_meas(r)
        if g is not None and not ("mogefov" in m or "gravnofall" in m): return None
    return c, "A U4 confident own CaW"


def A_U4a(r): return _A_u4(r, "a")
def A_U4b(r): return _A_u4(r, "b")
def A_U4c(r): return _A_u4(r, "c")


def _lam(r):
    p = r.get("parse") or {}; pr = p.get("priors") or []
    if len(pr) != 1 or pr[0].get("kind") != "speed" or not _f(pr[0].get("value_si")): return None
    v = float(pr[0]["value_si"]); keys = set(p.get("prior_dkeys") or [])
    s = sorted({(float(e["t"]), float(e["d"])) for e in p.get("depth") or [] if e.get("key") in keys and e.get("t") is not None and e.get("d") is not None})
    if len(s) < 2 or s[-1][0] - s[0][0] <= 0: return None
    rr = abs(s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]) / v
    return math.sqrt(max(0.0, 1 - rr * rr))


def _A_u5(r, thr, need_dis):
    if not _rul263(r): return None
    lam = _lam(r)
    c, cur = _f(r.get("s1_cs")), _f(r.get("cur"))
    if lam is None or lam >= thr or not (c and cur) or _same(cur, c): return None
    if need_dis:
        g = _geo_meas(r)
        if not g or abs(math.log(g / c)) < math.log(1.5): return None
    return c, f"A U5 radial lam={lam:.2f} -> own CaW"


def A_U5a(r): return _A_u5(r, 0.5, True)
def A_U5b(r): return _A_u5(r, 0.6, False)
def A_U5c(r): return _A_u5(r, 0.3, True)


def _tk(p):
    tk = {}
    for e in p.get("depth") or []:
        if e.get("t") is None or e.get("d") is None or not e.get("key"):
            continue
        tk.setdefault(e["key"], {})[float(e["t"])] = float(e["d"])
    return tk


def _floor(r):
    p = r.get("parse") or {}; q = p.get("q") or {}; k = q.get("kind"); u = USI.get(str(q.get("unit")))
    if u is None or k not in ("disp", "path", "avgspeed", "dist2"): return None
    tk = _tk(p)
    if k == "dist2":
        if q.get("t") is None: return None
        t = float(q["t"]); ll = p.get("llm") or {}; a, b = ll.get("target_depth_key"), ll.get("target2_depth_key")
        if a not in tk or b not in tk or a == b: return None
        da = [d for tt, d in tk[a].items() if abs(tt - t) < 1e-6]; db = [d for tt, d in tk[b].items() if abs(tt - t) < 1e-6]
        if not da or not db: return None
        fl = abs(da[0] - db[0])
    else:
        tg = p.get("target_dkeys") or []
        if len(tg) != 1 or tg[0] not in tk or q.get("t1") is None or q.get("t2") is None: return None
        t1, t2 = float(q["t1"]), float(q["t2"])
        d1 = [d for tt, d in tk[tg[0]].items() if abs(tt - t1) < 1e-6]; d2 = [d for tt, d in tk[tg[0]].items() if abs(tt - t2) < 1e-6]
        if not d1 or not d2 or t2 <= t1: return None
        fl = abs(d2[0] - d1[0])
        if k == "avgspeed": fl /= (t2 - t1)
    return fl / u if fl > 0 else None


def _I_u1(r, var):
    fl = _floor(r); cur = _f(r.get("cur"))
    if not (fl and cur): return None
    if var == "c":
        if not cur < fl * (1 - 1e-9): return None
    elif cur > fl * 1.001:
        return None
    if var == "b":
        c = [r.get("s1_g"), (r.get("geo_u") or {}).get("meas_value"), r.get("caw_full_greedy"), r.get("caw_crop_greedy"), r.get("qwen")]
        c += list(r.get("caw_full") or []) + list(r.get("caw_crop") or [])
        c = [float(x) for x in c if _f(x) and float(x) > fl * 1.001]
        v = float(np.exp(np.median(np.log(c)))) if c else fl
    else:
        c = [x for x in (_f(r.get("s1_g")), _f((r.get("geo_u") or {}).get("meas_value"))) if x and x > fl * 1.001]
        v = _lmean(c) if c else fl
    return v, f"I U1 depth floor {fl:.4g} ({var})"


def I_U1_V1a(r): return _I_u1(r, "a")
def I_U1_V1b(r): return _I_u1(r, "b")
def I_U1_V1c(r): return _I_u1(r, "c")


def _I_u2(r, var):
    p = r.get("parse") or {}; q = p.get("q") or {}; k = q.get("kind"); u = USI.get(str(q.get("unit")))
    kinds = ("dist2", "disp", "path", "size") if var == "c" else ("dist2", "disp", "path")
    if k not in kinds or u is None or "camera" in str(r.get("question", "")).lower(): return None
    cs = _f(r.get("s1_cs"))
    if not cs: return None
    ds = [float(e["d"]) for e in p.get("depth") or [] if _f(e.get("d"))]
    vals = [abs(a - b) / u for i, a in enumerate(ds) for b in ds[i + 1:] if abs(a - b) > 0]
    if var == "c": vals += [d / u for d in ds]
    ech = [e for e in vals if abs(cs / e - 1) <= 0.005]
    if not ech: return None
    e = ech[0]; g = _f(r.get("s1_g"))
    ne = [x for x in r.get("caw_full") or [] if _f(x) and abs(x / e - 1) > 0.005]
    nem = float(np.median(ne)) if len(ne) >= 4 else None
    if var == "b":
        cands = [x for x in (g, nem) if x and x > e]
        if not cands: return None
        return (math.sqrt(cands[0] * cands[1]) if len(cands) == 2 else cands[0]), "I U2 echo -> gm(geo, non-echo CaW)"
    if g is not None:
        return (g, "I U2 echo -> geo") if g > e else None
    return (nem, "I U2 echo -> non-echo CaW median") if nem and nem > e else None


def I_U2_V2a(r): return _I_u2(r, "a")
def I_U2_V2b(r): return _I_u2(r, "b")
def I_U2_V2c(r): return _I_u2(r, "c")


UNIT_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
TOL10 = math.log(1.10)








def _L2b(r, kinds, hedge_cur):
    if r.get("cat") != "S3" or r.get("video_source") != "lab": return None
    if str((r.get("geo") or {}).get("meas_mode")) != "3d_mogefov" or _kind(r) not in kinds: return None
    g, cs, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur) or abs(math.log(g / cs)) <= math.log(2): return None
    if not abs(math.log(cur / g)) < abs(math.log(cur / cs)): return None
    return (math.sqrt(cur * cs) if hedge_cur else math.sqrt(g * cs)), "L2b mogefov speed equal weight"


def L2b_V1(r): return _L2b(r, ("speed",), False)
def L2b_V2(r): return _L2b(r, ("speed",), True)
def L2b_V3(r): return _L2b(r, ("speed", "accel", "avgspeed"), False)


def _Q1(r, var):
    if str(r.get("video_type", ""))[1:2] != "3" or _kind(r) not in ("speed", "avgspeed", "path", "disp"): return None
    p = r.get("parse") or {}; keys = set(p.get("target_dkeys") or [])
    if not keys: return None
    ts = {float(e["t"]) for e in p.get("depth") or [] if e.get("key") in keys and e.get("t") is not None}
    if len(ts) > 1: return None
    geo = r.get("geo") or {}; m = str(geo.get("meas_mode"))
    if not bool(geo.get("meas_valid")) or not m.startswith("3d_"): return None
    g, cs, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur): return None
    if var != "V3":
        pr = p.get("priors") or []
        if not pr or pr[0].get("kind") != "size": return None
    v = max(cur, g) if var == "V2" else max(g, cs)
    if abs(math.log(v / cur)) < math.log(1.01): return None
    return v, f"Q1 radial lower bound {var}"


def Q1_V1(r): return _Q1(r, "V1")
def Q1_V2(r): return _Q1(r, "V2")
def Q1_V3(r): return _Q1(r, "V3")


def _V1(r, kinds, T):
    geo = r.get("geo") or {}
    if str(geo.get("answer_source")) != "meas" or "mogefov" not in str(geo.get("meas_mode")) or _kind(r) not in kinds: return None
    g, cs, cur = _f(geo.get("geo_value")), _f(r.get("s1_cs")), _f(r.get("cur"))
    if not (g and cs and cur) or abs(math.log(g / cs)) <= math.log(T): return None
    gm = math.sqrt(g * cs)
    if abs(math.log(cur / gm)) <= math.log(1.25): return None
    return gm, "V1 mfg gm(geo, CaW) uncalibrated focal"


def V_V1a(r): return _V1(r, ("speed", "avgspeed"), 2.0)
def V_V1b(r): return _V1(r, ("speed", "avgspeed", "disp", "path", "dist2"), 2.0)
def V_V1c(r): return _V1(r, ("speed", "avgspeed"), 1.5)


NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _echo(x, text):
    for s in NUM.findall(str(text)):
        try:
            v = abs(float(s))
        except Exception:
            continue
        if v > 0 and abs(x / v - 1) <= 0.01: return True
    return False


def _X(r, tol):
    g, cs, q, cur = _f(r.get("s1_g")), _f(r.get("s1_cs")), _f(r.get("qwen")), _f(r.get("cur"))
    if not (g and cs and q and cur): return None
    if _echo(q, r.get("prior", "")) or _echo(q, r.get("question", "")): return None
    if abs(math.log(g / q)) >= math.log(tol) or abs(math.log(cur / cs)) >= math.log(1.25): return None
    if abs(math.log(cur / g)) <= math.log(2) or abs(math.log(cur / q)) <= math.log(2): return None
    return math.sqrt(g * q), f"X indep gm(geo, qwen) tol {tol}"


def X_a(r): return _X(r, 1.5)
def X_b(r): return _X(r, 1.25)
def X_c(r): return _X(r, 2.0)


WIN = [re.compile(r"\b(?:from|between|over)\s+(\d*\.?\d+)\s*s?\s*(?:to|and|-)\s*(\d*\.?\d+)\s*s?\b"),
       re.compile(r"\bfrom\s+r?om\s+start\s+to\s+(\d*\.?\d+)\s*s\b"),
       re.compile(r"\bfrom\s+(\d*\.?\d+)\s*s\s+to\s+the\s+end\b")]


def _Y1(r, var):
    q = _q(r)
    if q.get("t1") is not None or q.get("t2") is not None or _kind(r) not in ("disp", "path", "avgspeed", "accel"): return None
    qt = str(r.get("question", "")).lower()
    if not any(w.search(qt) for w in WIN): return None
    cur = _f(r.get("cur"))
    gs = [_f((r.get(k) or {}).get("geo_value")) for k in ("geo", "geo_u")]
    if not cur or not any(g and abs(cur / g - 1) < 1e-6 for g in gs): return None
    fm, cm = _med(r.get("caw_full")), _med(r.get("caw_crop"))
    if not fm: return None
    v = math.sqrt(fm * cm) if (var == "V2" and cm) else fm
    return v, f"Y1 window missed -> CaW ({var})"


def Y1_V1(r): return _Y1(r, "V1")
def Y1_V2(r): return _Y1(r, "V2")






def _H(src, variant):
    def rule(row):
        if row.get("s1_g") is not None: return None
        m = _hm(row)
        if not m or m.get("src") != src: return None
        a, v = row["cur"], m["value"]
        if not (a > 0 and v > 0): return None
        if abs(math.log(v / a)) <= math.log(2): return math.sqrt(a * v), f"remeasure_{src}:agree_gm"
        if variant == "V2" and _caw_unstable(row): return v, f"remeasure_{src}:caw_unstable"
        return None
    return rule


H1_grav_V1 = _H("grav", "V1"); H1_grav_V2 = _H("grav", "V2"); H2_speed_V1 = _H("speed", "V1")

ALL = {k: v for k, v in dict(globals()).items() if re.match(r"^(r8_|A_U|I_U|L1_V|L2b_V|Q1_|V_V1|X_[abc]$|Y1_|H[12]_)", k) and callable(v)}
