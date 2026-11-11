import os, json, math, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("L0S_CACHE", os.path.join(os.path.dirname(HERE), "cache"))
QR = os.environ.get("QP_ROOT", "")
_R1 = {}


def _cache(unit, r):
    p = f"{CACHE}/{unit}_{r['split']}_{int(r['id'])}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def _r1why(r):
    sp = r["split"]
    if sp not in _R1:
        _R1[sp] = json.load(open(f"{QR}/clean/patches/ag_rul343_patch_{sp}.json"))
    return _R1[sp].get(str(int(r["id"])), {}).get("why", "")


def eff_src(r):
    if abs(math.log(r["cur"] / r["s1"])) > 1e-12:
        return "patch:stack"
    s = r["s1_src"]
    if s == "patch:rul343":
        m = re.search(r"last writer ([^;]+);", _r1why(r))
        if m:
            return m.group(1).strip()
    return s


def _num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


RX = re.compile(r"\(\s*not\s+(?:the\s+)?[^)]*\)", re.I)


def rul314(r):
    pr = str(r["prior"])
    if not RX.search(pr):
        return None
    if eff_src(r).startswith("patch"):
        return None
    if str(r["geo"].get("meas_mode")) != "2d_ratio":
        return None
    g = _num(r["geo"].get("geo_value"))
    cs = _num(r["s1_cs"]) if r["s1_cs"] is not None else float("nan")
    if not (np.isfinite(g) and g > 0 and np.isfinite(cs) and cs > 0):
        return None
    k = ((r["parse"] or {}).get("q") or {}).get("kind")
    if abs(np.log10(g / cs)) <= np.log10(2):
        v, why = (g, "agree_geo_kind") if k in ("dist2", "avgspeed") else (float(np.sqrt(g * cs)), "agree_gm")
    elif k == "size":
        v, why = float(np.sqrt(g * cs)), "dis_size_gm"
    else:
        v, why = float(cs), "dis_caw_routed"
    if abs(np.log10(v / r["cur"])) < 1e-6:
        return None
    return v, f"rul314: prior '{RX.search(pr).group(0)}' -> 2d_ratio weak; {why} (g={g:.3g}, c={cs:.3g})"


def rul091(r):
    c = _cache("rul091", r)
    if c is None:
        return None
    if eff_src(r).startswith("patch:"):
        return None
    if abs(np.log(c["val"] / r["cur"])) < np.log(1.15):
        return None
    g = c["got"]
    return float(c["val"]), (f"rul091: target track collapsed onto target2 (IoU {c['ie12']:.2f}); distinct target track {g['dir']}/{g['qk']} "
                             f"obj {g['obj']}; centre {g['dpx']:.1f}px x own mpp = {c['meas']:.4g}, {c['how']}")


def rul123(r):
    c = _cache("rul123", r)
    if c is None or abs(math.log(r["cur"] / c["cur_at_gen"])) > 1e-9:
        return None
    return float(c["value"]), c.get("why", "")


def rul044(r):
    c = _cache("rul044", r)
    if c is None or r["s1_g"] is None:
        return None
    g = float(r["s1_g"]); k = float(c["k"]); cur = r["cur"]
    cs = float(r["s1_cs"]) if r["s1_cs"] is not None else float("nan")
    why = f"rul044: window 'from {c['t1']:g} to {c['t2']:g}' re-parsed, own track re-measured x{k:.4f}"
    if abs(math.log(cur / g)) < 1e-6:
        return g * k, why + " (answer = geo)"
    if np.isfinite(cs) and cs > 0 and abs(math.log(cur / math.sqrt(g * cs))) < 1e-6:
        return math.sqrt(g * k * cs), why + " (answer = gm(geo, CaW))"
    return None


UNITS = [("L0S_rul314", rul314), ("L0S_rul091", rul091), ("L0S_rul123", rul123), ("L0S_rul044", rul044)]
