import os, json, math, re, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.dirname(HERE)
AG = os.environ["QP_R"] + "/analysis/v3/ag"; PATCHDIR = f"{AG}/r10/cl/clean/patches"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


DF = _load("r9_dfusion", f"{AG}/r10/work/r9/D_fusion/rules.py")
PL = _load("r19_plaus", f"{AG}/r19/PLAUS/py/rules.py")
CACHE = {sp: json.load(open(f"{X}/cache/{sp}.json")) for sp in ("test", "val")}
ORDER = ["agree_w065", "probe_D2_w07", "probe_S2_geo_simulation", "probe_S2_geo_internet", "probe_D2_geo_simulation",
         "probe_S3_geo_simulation", "probe_D3_geo_simulation"]
WT = {"agree_w065": 0.65, "probe_D2_w07": 0.7}
AGGR = ("rul325", "rul343", "rul411")
SKIPLOG = []
_PE = {}


def _patch_entry(tag, sp, i):
    if (tag, sp) not in _PE:
        _PE[(tag, sp)] = json.load(open(f"{PATCHDIR}/ag_{tag}_patch_{sp}.json"))
    return _PE[(tag, sp)].get(str(i)) or {}


def _same(a, b, tol=1e-9): return a is not None and b is not None and a > 0 and b > 0 and abs(math.log(a / b)) < tol


def route(r):
    s = str(r.get("s1_src"))
    if s == "agree_gm": return "agree_gm"
    if s.startswith("patch:") and s[6:] in AGGR:
        m = re.search(r"last writer (agree_w?gm)\b", str(_patch_entry(s[6:], r["split"], r["id"]).get("why", "")))
        if m: return f"{m.group(1)} via {s}"
    return None


def _agree_rel_value(r):
    rr = dict(r); rr["cur"] = r["s1"]
    v = DF.U1a(rr)
    if v is None: return None
    return v[0] if isinstance(v, tuple) else v


def eligible(r, allow_rel=True):
    rt = route(r)
    g, cs, cur = r.get("s1_g"), r.get("s1_cs"), r.get("cur")
    if rt is None or not g or not cs or g <= 0 or cs <= 0: return None
    gm = math.sqrt(g * cs)
    if not _same(r["s1"], gm, 1e-6): return None
    if _same(cur, gm, 1e-6): return rt, "gm"
    e = CACHE[r["split"]].get(str(r["id"])) or {}
    for u in e:
        if _same(cur, newval(u, g, cs)): return rt, "unit:" + u
    if allow_rel and _same(cur, _agree_rel_value(r)): return rt, "agree_rel"
    return None


def newval(unit, g, cs):
    w = WT.get(unit, 1.0)
    return g ** w * cs ** (1 - w)


def make(unit, allow_rel=True, plaus=True):
    def rule(r):
        e = (CACHE[r["split"]].get(str(r["id"])) or {}).get(unit)
        if not e: return None
        el = eligible(r, allow_rel)
        if el is None: return None
        v = newval(unit, r["s1_g"], r["s1_cs"])
        if plaus:
            rg = PL.rng_for(r)
            if rg is not None and not (rg[0] <= v <= rg[1]):
                SKIPLOG.append((r["split"], r["id"], unit, v, rg)); return None
        how = {0.65: "g^0.65 cs^0.35", 0.7: "g^0.7 cs^0.3"}.get(WT.get(unit), "own geo (w=1)")
        return v, f"{unit}: {how} (g={r['s1_g']:.5g}, cs={r['s1_cs']:.5g}); fusion route {e[0]}; c8 route {el[0]}, c8 value {el[1]}"
    rule.__name__ = unit
    return rule


KEPT = ["agree_w065", "probe_D2_w07", "probe_S3_geo_simulation"]
ACCEPTED = [make(u) for u in KEPT]
