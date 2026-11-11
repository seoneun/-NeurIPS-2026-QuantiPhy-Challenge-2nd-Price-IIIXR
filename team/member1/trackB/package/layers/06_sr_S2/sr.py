import sys, os, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.environ["QP_WORK"] + "/h"); import harness as H
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.getcwd(); HD = f"{HERE}/inputs"; OUT = os.environ["OUTDIR"]
L = f"{HERE}/../sr_common/mods/analysis/v3/compliance/rules"
sp_ = importlib.util.spec_from_file_location("rul336", f"{L}/rul336.py"); PQ = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(PQ)
PARSE = {"test": {int(x["id"]): x for x in map(json.loads, open(f"{R}/geo/out/parse_test_v2.jsonl"))}}
def ev(s):
    if s is None or isinstance(s, dict): return s or {}
    try: return eval(s, {"np": np, "nan": np.nan, "inf": np.inf})
    except Exception: return {}
def fnum(v):
    try: v = float(v); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan
BAND = np.log(2.0); MOTION = ("speed", "accel", "avgspeed", "disp", "dist2", "path")

def r_rul336(r):
    if r["cat"] != "S2": return None
    x = PARSE[r["split"]].get(int(r["id"]));  e, cp = PQ.lookup(x["question"], x) if x else (None, None)
    if e is None: return None
    f = PQ.si_factor(x); a = fnum(r["cur"]) * f; cls, rx, nx, kd, lo, hi, src = e
    if not PQ.flag(a, lo, hi) or PQ.prior_scaled(x): return None
    cs, cf, cc = fnum(r.get("s1_cs")), fnum(r.get("caw_full_greedy")), fnum(r.get("caw_crop_greedy"))
    other = cf if (np.isfinite(cs) and np.isfinite(cc) and np.isclose(cs, cc) and not np.isclose(cc, cf)) else cc
    gu = fnum(ev(r.get("geo_u")).get("geo_value"))
    cands = [("fusion", fnum(r.get("s1")) * f), ("g", fnum(r.get("s1_g")) * f), ("cs", cs * f), ("caw_other", fnum(other) * f), ("g_u", gu * f)]
    ch = PQ.choose(cands, lo, hi)
    if ch is None: return None
    return ch[1] / f, f"rul336 {cls} {a:.4g}->{ch[0]} {ch[1]:.4g}"
def r_rul335(r):
    if r["cat"] != "S2": return None
    x = PARSE[r["split"]].get(int(r["id"]));  e, cp = PQ.lookup(x["question"], x) if x else (None, None)
    if e is None: return None
    f = PQ.si_factor(x); a = fnum(r["cur"]) * f; cls, rx, nx, kd, lo, hi, src = e
    if not np.isfinite(a) or a <= 0 or lo <= a <= hi or PQ.flag(a, lo, hi) or PQ.prior_scaled(x): return None
    c = min(max(a, lo), hi); return c / f, f"rul335 {cls} {a:.4g}->{c:.4g}"
def r_s2mchain(r):
    if r["cat"] != "S2" or not str(r["video_id"]).startswith("internet"): return None
    x = PARSE[r["split"]].get(int(r["id"])) or {}
    if (x.get("q") or {}).get("kind") != "size" or str(r.get("s1_src")) != "dis_size_gm": return None
    a, g, cs = fnum(r["cur"]), fnum(r.get("s1_g")), fnum(r.get("s1_cs"))
    if not (np.isfinite(a) and np.isclose(a, fnum(r.get("s1")), rtol=1e-9)) or not (np.isfinite(g) and cs > 0): return None
    G = {"m": ev(r.get("geo")), "u": ev(r.get("geo_u"))}
    ch = next((c for c in "mu" if np.isclose(g, fnum(G[c].get("geo_value")), rtol=1e-6)), None)
    if ch is None: return None
    o = G["u" if ch == "m" else "m"]; g2 = fnum(o.get("geo_value"))
    if not (bool(o.get("meas_valid")) and np.isfinite(g2) and g2 > 0): return None
    if not (abs(np.log(g / cs)) > BAND and abs(np.log(g2 / cs)) <= BAND): return None
    return g2, f"s2mchain {ch} {g:.4g} vs cs {cs:.4g}; other {g2:.4g}"
def r_s2mdiscaw(r):
    if r["cat"] != "S2" or not str(r["video_id"]).startswith("internet") or str(r.get("s1_src")) != "dis_caw_routed": return None
    x = PARSE[r["split"]].get(int(r["id"])) or {}
    if (x.get("q") or {}).get("kind") not in MOTION: return None
    g, a = fnum(r.get("s1_g")), fnum(r["cur"])
    if not (np.isfinite(g) and g > 0) or not np.isclose(a, fnum(r.get("s1")), rtol=1e-9): return None
    return g, f"s2mdiscaw g {g:.4g} (cs {fnum(r.get('s1_cs')):.4g})"
RULES = [("rul336", r_rul336), ("rul335", r_rul335), ("s2mchain", r_s2mchain), ("s2mdiscaw", r_s2mdiscaw)]

h3 = pd.read_csv(f"{HD}/pred_t077_id1.csv"); h3v = dict(zip(h3["id"] - 1, h3.parsed_value))
hr = pd.read_csv(os.environ["H1ROWS"]).set_index("id0"); src = pd.read_csv(f"{HD}/h3_src.csv").set_index("id0")
def eligible(i, r):
    rs, st = hr.reason.get(i), src.src.get(i)
    if rs != "S" or st != "h1": return True
    return bool(np.isclose(h3v[i], fnum(r.get("s1")), rtol=1e-9))

rows = []
for i, r0 in H.ROWS["test"].items():
    if r0["cat"] != "S2" or not eligible(i, r0): continue
    r = dict(r0); r["cur"] = h3v[i]
    for nm, fn in RULES:
        o = fn(r)
        if o is None: continue
        v, why = o
        if not np.isclose(v, r["cur"], rtol=1e-9):
            rows.append(dict(id0=i, value=v, rule=nm)); r["cur"] = v
pd.DataFrame(rows).to_csv(f"{OUT}/all_changes.csv", index=False)
