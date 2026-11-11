import os, re, math, json, importlib.util
_W = os.environ.get("QP_WORK", ""); _C = os.environ.get("QP_ROOT", "")
HERE = os.path.dirname(os.path.abspath(__file__))
CAWALONE = ("caw_nogeo", "dis_caw_routed", "caw_c1")
CAWALONE_MAPPED = ("patch:rm_s2u2_noshrink",)
FUSION = ("agree_gm", "agree_wgm", "dis_size_gm", "dis_size_low_gm")
UNITS = ("prescale", "shrink", "d3s_nh", "d3s_h", "dimgm_w", "dimgm_l", "pS2", "pS3", "pD2", "pD3")
K = dict(shrink=0.9, pS2=1.1111, pS3=1.1111, pD2=0.9, pD3=0.9, d3s_nh=1.1, d3s_h=1.05, dimgm_w=2.6, dimgm_l=1.6, prescale=1.12)
RX = {"height": r"\b(?:height|tall)\b", "width": r"\b(?:width|wide|wingspan)\b", "length": r"\b(?:length|long)\b"}
_KM = None
_RT = {}


def _kin():
    global _KM
    if _KM is None:
        s = importlib.util.spec_from_file_location("kinematic_old", f"{_C}/analysis/v3/kinematic.py"); m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m); m.EXTRA_BOUNDS = m.EXTRA2 = m.EXTRA3 = True; _KM = m
    return _KM


def route_rec(r):
    if r.get("r23") is not None: return r["r23"]
    sp = r["split"]
    if sp not in _RT:
        _RT[sp] = json.load(open(f"{os.path.dirname(HERE)}/out/route_{sp}.json"))
    return _RT[sp].get(str(r["id"]))


def _fin(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) and v > 0 else None


def kind(r):
    p = r.get("parse") or {}; k = (p.get("q") or {}).get("kind") if p else None
    return str(k if k else (r.get("geo") or {}).get("meas_kind"))


def kin_exempt(r, c0):
    x = r.get("parse") or {}
    if not x.get("priors"): return False
    pr = x["priors"][0]; q = x.get("q") or {}
    try:
        gv = pr.get("value_si") if (pr.get("gravity") or (pr.get("kind") == "accel" and 9.0 <= float(pr.get("value_si") or 0) <= 10.0)) else None
    except Exception:
        gv = None
    SI = {"m": 1, "cm": 0.01, "mm": 0.001, "m/s": 1, "cm/s": 0.01, "km/h": 1 / 3.6}
    u = str(q.get("unit") or "")
    if not gv or u not in SI or c0 is None: return False
    kv = None
    try:
        if q.get("kind") == "speed" and q.get("t"): kv = gv * float(q["t"])
        elif q.get("kind") == "disp" and q.get("t1") is not None and q.get("t2") is not None: kv = 0.5 * gv * (float(q["t2"]) ** 2 - float(q["t1"]) ** 2)
    except Exception:
        kv = None
    return bool(kv and kv > 0 and abs(c0 * SI[u] / kv - 1) < 0.01)


def classify(r):
    I = route_rec(r) or {}; rt = I.get("route"); late = I.get("writer"); v = _fin(r.get("cur")); cs, g = _fin(r.get("s1_cs")), _fin(r.get("s1_g"))
    if v is None: return "other", None
    if late is None and (rt in CAWALONE or rt in CAWALONE_MAPPED) and cs and 0.92 <= v / cs <= 1.0 + 1e-9:
        return "caw", cs
    if late is None and rt == "rul263_caw": return "d3caw", None
    fus = (late is None and rt in FUSION) or (late is not None and late.startswith("c2:") and rt in FUSION)
    if fus and g and cs and abs(math.log(cs / g)) > 1e-9:
        e = math.log(v / g) / math.log(cs / g)
        if -1e-6 <= e <= 1 + 1e-6: return "fusion", min(max(e, 0.0), 1.0)
    return "other", None


_PL = None


def _plaus():
    global _PL
    if _PL is None:
        p = os.environ["QP_R"] + "/analysis/v3/ag/r19/PLAUS/py/rules.py"
        s = importlib.util.spec_from_file_location("plaus_rules_r23", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); _PL = m
    return _PL


def plaus_ok(r, v_old, v_new):
    g = _plaus().rng_for(r)
    if g is None: return True
    lo, hi = g[0], g[1]
    out = lambda v: (math.log(lo / v) if lo > 0 and v < lo else 0.0) + (math.log(v / hi) if v > hi else 0.0)
    return out(v_new) <= out(v_old) + 1e-12


def make_rule(on, plaus=True):
    base = _make_rule(on)

    def rule(r):
        o = base(r)
        if o is None or not plaus: return o
        return o if plaus_ok(r, float(r["cur"]), float(o[0])) else None
    return rule


def _make_rule(on):
    on = set(on)

    def rule(r):
        cl, x = classify(r)
        if cl == "other": return None
        v = float(r["cur"]); cat = r["cat"]; kd = kind(r); sim = str(r["video_id"]).startswith("simulation"); q = str(r["question"]).lower()
        why = []; f = 1.0
        gv = _fin((r.get("geo") or {}).get("geo_value"))
        if "prescale" in on and sim and cat == "D2" and kd == "size" and gv and _fin(r.get("s1_cs")):
            e = 1.0 if cl == "caw" else (x if cl == "fusion" else 0.0)
            if e > 0: f *= K["prescale"] ** e; why.append(f"prescale x1.12^{e:.3f}")
        if cl == "fusion":
            if f == 1.0: return None
            return v * f, f"r23 SCALES [fusion] " + "; ".join(why) + f": {v:.4g} -> {v * f:.4g}"
        if cl == "caw":
            base = x
            if "shrink" in on and kd == "size": f *= K["shrink"]; why.append("shrink x0.9")
            if not kin_exempt(r, base):
                for u, c, kk in (("pS2", "S2", "size"), ("pS3", "S3", "size"), ("pD2", "D2", "any"), ("pD3", "D3", "any")):
                    if u in on and cat == c and (kk == "any" or kd == kk): f *= K[u]; why.append(f"{u} x{K[u]}")
            if f == 1.0 and abs(base / v - 1) < 1e-12: return None
            nv = base * f
            return nv, f"r23 SCALES [caw route {(route_rec(r) or {}).get('route')}] own CaW {base:.4g}; " + "; ".join(why or ["(factors cancel)"]) + f": {v:.4g} -> {nv:.4g}"
        if not (sim and cat == "D3"): return None
        nv = v; ish = re.search(RX["height"], q) is not None
        if "d3s_nh" in on and not ish: nv *= K["d3s_nh"]; why.append("rul263_scale x1.1 (not height)")
        if "d3s_h" in on and ish: nv *= K["d3s_h"]; why.append("rul263_scale x1.05 (height)")
        g = _fin(r.get("s1_g"))
        for u, dm in (("dimgm_w", "width"), ("dimgm_l", "length")):
            if u in on and g and re.search(RX[dm], q) and not ish and abs(math.log10(g / nv)) <= math.log10(K[u]):
                if dm == "width":
                    km = _kin(); b = km.class_bound(str(r["question"]), str(r["prior"]), "size")
                    if b is not None and not (b[0] <= g * km.q_si(str(r["question"])) <= b[1]): continue
                nv = math.sqrt(nv * g); why.append(f"rul263_dimgm {dm} gm with own geo {g:.4g} (within {K[u]}x)")
        if abs(nv / v - 1) < 1e-12: return None
        return nv, "r23 SCALES [rul263_caw] " + "; ".join(why) + f": {v:.4g} -> {nv:.4g}"
    return rule


KEPT = ["prescale", "shrink", "d3s_nh", "d3s_h", "dimgm_w", "dimgm_l", "pS2", "pS3", "pD2", "pD3"]
RULE = make_rule(KEPT) if KEPT else None
ACCEPTED = [RULE] if RULE else []
