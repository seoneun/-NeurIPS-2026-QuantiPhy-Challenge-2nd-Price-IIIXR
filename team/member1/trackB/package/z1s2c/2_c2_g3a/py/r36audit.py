import sys, os, re, math, json
import numpy as np, pandas as pd
sys.path.insert(0, os.environ["QP_WORK"] + "/h")
import harness as H
RR = os.environ["QP_R"]; AG = f"{RR}/analysis/v3/ag"; OD = sys.argv[1]; PO = f"{OD}/r36_audit"; os.makedirs(PO, exist_ok=True)
BASE = {sp: pd.read_csv(f"{OD}/series/pred_t127_{sp}.csv", index_col=0)[sp] for sp in ("test", "val")}
def chg(a, b): return a.index[np.abs(np.log(a / b.reindex(a.index))) > 1e-12]

STRONG = ("2d_ratio", "3d_calib")
NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")
LN2 = math.log(2.0)


def fin(x):
    try: x = float(x)
    except Exception: return None
    return x if (np.isfinite(x) and x > 0) else None


def kind(r): return ((r.get("parse") or {}).get("q") or {}).get("kind")
def gmode(r): return str((r.get("geo") or {}).get("meas_mode") or "")
def gc(r): return fin(r.get("s1_g")), fin(r.get("s1_cs"))


BADWHY = re.compile(r"copy|twin|cross-row|propagat|plain row|_segmented copy", re.I)
CACHE = {}


def fuse(r, g, strong):
    c = fin(r.get("s1_cs")); k = kind(r); sim = str(r["video_id"]).startswith("simulation")
    if c is None: return g
    if abs(math.log(g / c)) <= LN2:
        if k in ("dist2", "avgspeed"): return g
        if k == "size" and strong: return g if sim else g ** 0.65 * c ** 0.35
        return math.sqrt(g * c)
    return g if strong else None


Q3N = {}
def q3n(sp):
    if sp not in Q3N:
        d = pd.read_csv(f"{RR}/preds/{sp}_geo_v4ntrj5sQ3nsrngsr_d_vw_raw.csv").set_index("Unnamed: 0"); Q3N[sp] = d
    return Q3N[sp]


def regeo(tag, sp, i, val, w):
    r = H.ROWS[sp][i]; num = r"([0-9.eE+-]+)"
    if tag in ("rul098", "hgt", "second"): return val, "raw own measurement (old rule value)"
    if tag == "rul441":
        m = re.search(r"; g " + num + "->" + num, w)
        if not m: return None, "unparsed"
        k = float(m.group(2)) / float(m.group(1)); g9 = fin(r.get("s1_g")); gn = g9 * k if g9 else float(m.group(2))
        if "geo_cawinvalid" in w: return gn, f"uncut ruler k {k:.3f} -> geo (CaW invalid)"
        return fuse(r, gn, True), f"uncut ruler k {k:.3f}, own geo {gn:.4g}, c9n fusion"
    if tag == "rul016":
        m = re.search(r"centre drop " + num + r"px x g/a=" + num, w)
        if m: gn = float(m.group(1)) * float(m.group(2)); return fuse(r, gn, True), f"own drop height {gn:.4g}, c9n fusion"
        return val, "airborne fit (geo, as old)"
    if tag == "rul244":
        m = re.search(r"= " + num + r" px x own 2d_ratio mpp " + num, w)
        if not m: return None, "unparsed"
        gn = float(m.group(1)) * float(m.group(2)); return fuse(r, gn, True), f"own paint width {gn:.4g}, c9n fusion"
    if tag == "rul022":
        m = re.search(r"g' " + num, w)
        if not m: return None, "unparsed"
        gn = float(m.group(1).rstrip(".")); return fuse(r, gn, True), f"own long side {gn:.4g}, c9n fusion"
    if tag == "rul315":
        m = re.search(r"-> geo " + num, w)
        if not m: return None, "unparsed"
        gn = float(m.group(1)); return fuse(r, gn, True), f"own free-fall ruler geo {gn:.4g}, c9n fusion"
    if tag == "d09networld_rul302":
        m = re.search(r"r=" + num, w); g9 = fin(r.get("s1_g"))
        if not m or g9 is None: return None, "unparsed / no own geo"
        rr_ = float(m.group(1)); gn = g9 / rr_
        return fuse(r, gn, gmode(r) in STRONG), f"own geo / person-prior scale error r {rr_:.3f} = {gn:.4g}, c9n fusion"
    if tag == "rul424":
        q = q3n(sp); gn = fin(q.geo_value.get(i)); md = str(q.meas_mode.get(i))
        if gn is None: return None, "no narrowed geo"
        return fuse(r, gn, md in STRONG), f"own narrowed-xground geo {gn:.4g} mode {md}, c9n fusion"
    return val, "as old"


def load_cache(tag):
    out = {}
    for sp in ("test", "val"):
        p = f"{RR}/cache/tmp/ag_{tag}_patch_{sp}.json"; d = {}
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                i = int(k); w = str(v.get("why", "")) if isinstance(v, dict) else ""
                val = fin(v["value"] if isinstance(v, dict) else v)
                if val is None or i not in H.ROWS[sp] or H.ROWS[sp][i]["cat"] != "D2": continue
                if BADWHY.search(w): continue
                if tag == "rul424" and "(own row)" not in w: continue
                nv, note = regeo(tag, sp, i, val, w)
                if nv is None or not fin(nv): continue
                d[i] = (float(nv), note + " | old: " + w[:110])
        out[sp] = d
    return out


def make_cache_rule(tag):
    C = CACHE.setdefault(tag, load_cache(tag))
    def rule(r):
        x = C[r["split"]].get(int(r["id"]))
        return None if x is None else (x[0], f"{tag}: {x[1]}")
    return rule



LOG10_2 = math.log10(2)
def pos(x):
    try: x = float(x)
    except Exception: return None
    return x if np.isfinite(x) and x > 0 else None
def kind(r): return ((r.get("parse") or {}).get("q") or {}).get("kind")
def tphrase(r): return str(((r.get("parse") or {}).get("q") or {}).get("target") or "").lower()
def eq(a, b):
    a, b = pos(a), pos(b); return a is not None and b is not None and abs(math.log(a / b)) < 1e-9
def caw_led(cur, s1, tol=0.045):
    a, b = pos(cur), pos(s1); return a is not None and b is not None and abs(math.log10(a / b)) <= tol
def gmode(r): return str((r.get("geo") or {}).get("meas_mode") or "")
def gwhy(r): return str((r.get("geo") or {}).get("meas_why") or "")
def issim(r): return str(r["video_id"]).startswith("simulation")


PERSONT = re.compile(r"\b(pedestrian|person|human|man|woman|people|rider)\b", re.I)
def A6_brkt(tol=1.1, slack=0.97):
    def f(r):
        if r["cat"] != "D3" or kind(r) != "size" or not str(r["s1_src"]).startswith("rul263") or not caw_led(r["cur"], r["s1"]): return None
        g, c, a = pos(r["s1_g"]), pos(r["s1_cs"]), pos(r["cur"])
        if None in (g, c, a) or abs(math.log(g / c)) >= math.log(tol): return None
        if min(g, c) * slack <= a <= max(g, c) / slack: return None
        if PERSONT.search(tphrase(r)) or PERSONT.search(str(r["question"])): return None
        return math.sqrt(g * c), f"A6 bracket: geo {g:.4g} ~ CaW {c:.4g} (<10%), answer {a:.4g} outside -> gm"
    return f



LG2 = math.log(2.0)
def _f(x):
    try:
        x = float(x); return x if np.isfinite(x) and x > 0 else None
    except Exception: return None

def _kind(r): return ((r.get("parse") or {}).get("q") or {}).get("kind")

_Q3 = {}
def _q3(sp):
    if sp not in _Q3:
        cols = ["Unnamed: 0", "geo_value", "meas_mode", "answer_source", "merge_src"]
        a = pd.read_csv(f"{RR}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0, usecols=cols)
        b = pd.read_csv(f"{RR}/preds/{sp}_geo_v4ntrj5sQ3nsrngsr_d_vw_raw.csv", index_col=0, usecols=cols).reindex(a.index)
        _Q3[sp] = (a, b)
    return _Q3[sp]

def _fuse(g, c, kind, strong):
    if c is None: return g
    if abs(math.log(g / c)) <= LG2:
        if kind in ("dist2", "avgspeed"): return g
        return g ** 0.65 * c ** 0.35 if (strong and kind == "size") else math.sqrt(g * c)
    if strong: return g
    return math.sqrt(g * c) if kind == "size" else c

def STALE2_U(r):
    if r["cat"] != "S2": return None
    a, b = _q3(r["split"]); i = int(r["id"])
    if i not in a.index: return None
    ga, gb = a.geo_value.get(i), b.geo_value.get(i)
    same = (pd.isna(ga) and pd.isna(gb)) or (not pd.isna(ga) and not pd.isna(gb) and abs(math.log(float(ga) / float(gb))) < 1e-9)
    if same and str(a.meas_mode.get(i)) == str(b.meas_mode.get(i)) and str(a.answer_source.get(i)) == str(b.answer_source.get(i)): return None
    c = _f(r.get("s1_cs")); k = _kind(r)
    def val(g, mode, srcv):
        if str(srcv) != "meas" or _f(g) is None: return c
        return _fuse(float(g), c, k, str(mode) in ("2d_ratio", "3d_calib"))
    vb = val(gb, b.meas_mode.get(i), b.answer_source.get(i))
    if vb is None: return None
    return vb, f"rul424 own: Q2 geo {ga} ({a.meas_mode.get(i)}/{a.answer_source.get(i)}) -> Q3n {gb} ({b.meas_mode.get(i)}/{b.answer_source.get(i)}), CaW {c}"

UNITS = [("r36_D2_rul098", make_cache_rule("rul098")), ("r36_D2_stale2own", make_cache_rule("rul424")),
         ("r36_D3_A6_brkt", A6_brkt()), ("r36_S2_STALE2_U", STALE2_U),
         ("r36_D2_rul441", make_cache_rule("rul441")), ("r36_D2_hgt", make_cache_rule("hgt")),
         ("r36_D2_rul244", make_cache_rule("rul244")), ("r36_D2_rul016", make_cache_rule("rul016")),
         ("r36_D2_second", make_cache_rule("second")), ("r36_D2_rul315", make_cache_rule("rul315"))]
for nm, fn in UNITS:
    for sp in ("test", "val"):
        n_, w_ = H.apply(fn, sp, BASE)
        d = {str(int(i)): dict(value=float(n_[i]), before=float(BASE[sp][i]), why=w_.get(i, "")) for i in chg(n_, BASE[sp])}
        json.dump(d, open(f"{PO}/{nm}_{sp}.json", "w"), indent=0)
