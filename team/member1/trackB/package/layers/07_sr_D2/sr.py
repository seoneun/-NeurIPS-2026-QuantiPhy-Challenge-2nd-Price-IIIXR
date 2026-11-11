import sys, os, csv, json, importlib.util, numpy as np
sys.path.insert(0, os.environ["QP_WORK"] + "/h"); import harness as H
HERE = os.path.dirname(os.path.abspath(__file__)); RR = os.getcwd()
QR = HERE + "/../sr_common/mods/analysis/v3/compliance/rules"
OUT = os.environ["OUTDIR"]
def mod(n):
    sp = importlib.util.spec_from_file_location(n, f"{QR}/{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
PQ = mod("rul336"); PX = mod("pxq"); P3N = mod("p3qns"); P3 = mod("p3q"); PW = mod("pxwalk")
P = {"test": {int(json.loads(l)["id"]): json.loads(l) for l in open(f"{RR}/geo/out/parse_test_v2.jsonl")}}
def fl(v):
    try:
        v = float(v); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan
def med(x):
    x = [fl(a) for a in (x or [])]; x = [a for a in x if np.isfinite(a)]
    return float(np.median(x)) if x else np.nan
def est(r):
    gu = (r.get("geo_u") or {}).get("meas_value")
    return dict(fus=fl(r.get("s1")), g=fl(r.get("s1_g")), cs=fl(r.get("s1_cs")), cf=med(r.get("caw_full")), cc=med(r.get("caw_crop")), gu=fl(gu))
def cands(e, f):
    cs, cc, cf = e["cs"], e["cc"], e["cf"]
    other = cf if np.isfinite(cs) and np.isfinite(cc) and np.isclose(cs, cc) and not np.isclose(cc, cf) else cc
    return [("fusion", e["fus"] * f), ("g", e["g"] * f), ("cs", cs * f), ("caw_other", other * f), ("g_u", e["gu"] * f)]
def r_pxwalk(x, a, e):
    q = str(x["question"]).lower()
    if x.get("cat") not in PW.CATS or not str(x["video_id"]).startswith("simulation"): return None
    if not re.search(r"speed|velocity", q) or not PW.PERSON.search(q) or PW.NOT.search(q): return None
    if np.isfinite(a) and PW.LO <= a <= PW.HI: return None
    return PW.V
def r_flagq(L, lookup, prior_scaled, ent8):
    def f_(x, a0, e):
        en, cp = lookup(x["question"], x)
        if en is None: return None
        f = PQ.si_factor(x); a = a0 * f
        lo, hi = (en[5], en[6]) if ent8 else (en[4], en[5])
        if not PQ.flag(a, lo, hi) or prior_scaled(x): return None
        ch = PQ.choose(cands(e, f), lo, hi)
        return None if ch is None else ch[1] / f
    return f_
def r_clip(lookup, prior_scaled, ent8):
    def f_(x, a0, e):
        en, cp = lookup(x["question"], x)
        if en is None: return None
        f = PQ.si_factor(x); a = a0 * f
        lo, hi = (en[5], en[6]) if ent8 else (en[4], en[5])
        if not np.isfinite(a) or a <= 0 or lo <= a <= hi or PQ.flag(a, lo, hi) or prior_scaled(x): return None
        return min(max(a, lo), hi) / f
    return f_
import re
_av = sys.argv; sys.argv = [_av[0]]
_sp = importlib.util.spec_from_file_location("c79", HERE + "/../sr_common/mods/rul151/rule.py")
C79 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C79); sys.argv = _av
def _track_states(j, phrase):
    qs = j.get('queries')
    if isinstance(qs, list):
        for q in qs:
            if q.get('phrase') == phrase or q.get('key', '').split('|')[0] == phrase:
                if 'frames' in q:
                    return [x['st'] for x in q.get('frames', []) if x and x.get('st')]
                o = q.get('obj')
                if o is None: return None
                L = j.get('objs', {}).get(str(o)) or []
                return [x['st'] for x in L if x and x.get('st')]
        return None
    return C79.track_states(j, phrase)
TRK = {}
def r_ptwprior(x, a, e, r, sp):
    v = r["video_id"]
    if not str(v).startswith("simulation"): return None
    pr = (x.get("priors") or [None])[0]
    if not pr or pr.get("kind") != "size" or pr.get("dim") != "width": return None
    G = r.get("geo") or {}
    if G.get("meas_mode") not in ("2d_ratio", "vlm_2d_ratio"): return None
    ph = (x.get("prior_obj") or {}).get("phrase"); ver = G.get("meas_tracks")
    if not ph: return None
    S = float(pr["value_si"]); key = (sp, v, ph, ver)
    if key not in TRK:
        j = C79.load_tracks(sp, ver, v); TRK[key] = _track_states(j, ph) if j else None
    st = TRK[key]
    if not st or len(st) < 5: return None
    b = np.array([s["box"] for s in st], float)
    bw, bh = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1])
    rs = float(np.median([s["rr"][1] for s in st])); bord = float(np.mean([s.get("border", 0) != 0 for s in st]))
    mpp = fl(G.get("meas_mpp")); pxr = S / mpp if np.isfinite(mpp) and mpp > 0 else np.nan
    if not (bw > 1.3 * bh and abs(rs / bh - 1) < 0.08 and bord <= 0.2 and np.isfinite(pxr) and abs(pxr / rs - 1) < 0.08): return None
    g = e["g"]
    if not (np.isfinite(g) and g > 0 and abs(a / g - 1) < 1e-6): return None
    return float(g) * rs / bw
RULES = [("pxwalk", r_pxwalk),
         ("pxq", r_flagq(None, PX.lookup, PX.prior_scaled, True)), ("pxclip", r_clip(PX.lookup, PX.prior_scaled, True)),
         ("p3qns", r_flagq(None, P3N.lookup, P3N.prior_scaled, True)), ("p3clip", r_clip(P3.lookup, P3.prior_scaled, True)),
         ("rul336", r_flagq(None, PQ.lookup, PQ.prior_scaled, False)), ("rul335", r_clip(PQ.lookup, PQ.prior_scaled, False))]
def run_row(x, a, e, r=None, sp=None):
    hist = []
    v = r_ptwprior(x, a, e, r, sp)
    if v is not None and np.isfinite(v): hist.append("ptwprior"); a = float(v)
    for nm, fn in RULES:
        v = fn(x, a, e)
        if v is not None and np.isfinite(v) and not np.isclose(v, a, rtol=1e-9):
            hist.append(nm); a = float(v)
    return a, hist
h1 = {int(r["id0"]): r for r in csv.DictReader(open(os.environ["H1ROWS"]))}
h3 = {int(r["id0"]): r for r in csv.DictReader(open(HERE + "/inputs/h3_src.csv"))}
with open(OUT + "/cand_all.csv", "w", newline="") as f:
    wc = csv.writer(f); wc.writerow(["id0", "value", "rule"])
    for i, r in h3.items():
        if r["cat"] != "D2": continue
        cur = float(r["value"]); e = est(H.ROWS["test"][i])
        if h1[i]["reason"] == "S" and r["src"] == "h1" and not (np.isfinite(e["fus"]) and np.isclose(cur, e["fus"], rtol=1e-3)): continue
        new, hist = run_row(P["test"][i], cur, e, H.ROWS["test"][i], "test")
        if hist: wc.writerow([i, new, "+".join(hist)])
