import sys, os, re, math, copy, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.environ["QP_WORK"] + "/h")
import harness as h
HERE = os.path.dirname(os.path.abspath(__file__))
LV = HERE + '/../sr_common/mods/analysis/v3/compliance/rules'
OUT = os.environ['OUTDIR']
sp_ = importlib.util.spec_from_file_location("rul336", LV + "/rul336.py")
PQ = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(PQ)


def fin(v):
    try:
        return v is not None and np.isfinite(float(v)) and float(v) > 0
    except Exception:
        return False


def own_g(r):
    return r['s1_g'] if fin(r['s1_g']) else r['geo'].get('geo_value')


def ests(r):
    cs = r['s1_cs']; cf = r['caw_full_greedy']; cc = r['caw_crop_greedy']
    other = cf if (fin(cs) and fin(cc) and np.isclose(cs, cc) and not (fin(cf) and np.isclose(cc, cf))) else cc
    return [("fusion", r['s1']), ("g", own_g(r)), ("cs", cs), ("caw_other", other), ("g_u", r['geo_u'].get('geo_value'))]


def rul336(r):
    x = r['parse']
    if not x:
        return None
    e, cp = PQ.lookup(x["question"], x)
    if e is None:
        return None
    f = PQ.si_factor(x); a = r['cur'] * f
    cls, rx, nx, kd, lo, hi, src = e
    if not PQ.flag(a, lo, hi) or PQ.prior_scaled(x):
        return None
    cands = [(n, float(v) * f) for n, v in ests(r) if fin(v)]
    ch = PQ.choose(cands, lo, hi)
    if ch is None:
        return None
    nm, c, how = ch
    return c / f, "rul336: %s ans %.4g SI outside [%.4g,%.4g] -> own %s %.4g (%s)" % (cls, a, lo / 1.5, hi * 1.5, nm, c, how)


def rul335(r):
    x = r['parse']
    if not x:
        return None
    e, cp = PQ.lookup(x["question"], x)
    if e is None:
        return None
    f = PQ.si_factor(x); a = r['cur'] * f
    cls, rx, nx, kd, lo, hi, src = e
    if not np.isfinite(a) or a <= 0 or lo <= a <= hi or PQ.flag(a, lo, hi) or PQ.prior_scaled(x):
        return None
    c = min(max(a, lo), hi)
    return c / f, "rul335: %s ans %.4g SI outside [%s,%s] -> bound %.4g" % (cls, a, lo, hi, c)


PERSON = re.compile(r"\b(person|pedestrian|human|man|woman|walker)\b")
NOT = re.compile(r"bicycle|bike|cycl|car\b|ride|riding|rider|scoot|skat|astronaut|run|jog|swim|boat|horse|accel")


def pxwalk(r):
    q = r['question'].lower()
    if r['cat'] not in ("S2", "D2", "D3") or not r['video_id'].startswith("simulation"):
        return None
    if not re.search(r"speed|velocity", q) or not PERSON.search(q) or NOT.search(q):
        return None
    a = r['cur']
    if np.isfinite(a) and 1.0 <= a <= 1.6:
        return None
    return 1.30, "pxwalk: walking-person speed %.4g outside [1.0,1.6] -> 1.30" % a


U = {"cm": 100.0, "cm/s": 100.0, "mm": 1000.0, "mm/s": 1000.0}


def nrm(s):
    return re.sub(r"[\s_]+", "", str(s).lower())


def lodkcw(r):
    j = r['parse'] or {}; q = j.get("q") or {}
    t1, t2, kind = q.get("t1"), q.get("t2"), q.get("kind")
    if t1 is None or t2 is None or kind not in ("disp", "speed") or float(t2) <= float(t1):
        return None
    tk = {nrm(k) for k in (j.get("target_dkeys") or [])}
    if len(tk) != 1:
        return None
    t1, t2 = float(t1), float(t2)
    ks = sorted((float(e["t"]), float(e["d"])) for e in j.get("depth", [])
                if e.get("t") is not None and nrm(e["key"]) in tk and t1 - 1e-6 <= float(e["t"]) <= t2 + 1e-6)
    g = own_g(r); cs = r['s1_cs']
    if len(ks) < 2 or not (fin(g) and fin(cs)):
        return None
    g = float(g); cs = float(cs)
    u = U.get(str(q.get("unit")).lower(), 1.0) / ((t2 - t1) if kind == "speed" else 1.0)
    hit = next(((a, b) for x, a in enumerate(ks) for b in ks[x + 1:]
                if abs(b[1] - a[1]) > 0 and abs(np.log(cs / (abs(b[1] - a[1]) * u))) < 0.02), None)
    if hit is None or not g > 1.02 * cs:
        return None
    a = r['cur']
    if not (cs * (1 - 1e-9) <= a < g):
        return None
    return g, "lodkcw: own CaW %.4g = own depth-key diff |d(%g)-d(%g)| (radial lower bound); answer %.4g between -> own geo %.4g" % (
        cs, hit[1][0], hit[0][0], a, g)


RULES = [("rul336", rul336), ("rul335", rul335), ("pxwalk", pxwalk), ("lodkcw", lodkcw)]


def run(rows, cur):
    out = {}
    for i, r in rows.items():
        if r['cat'] != "D3":
            continue
        rr = copy.deepcopy(r); rr['cur'] = float(cur[i]); hist = []
        for nm, fn in RULES:
            v = fn(copy.deepcopy(rr))
            if v is None:
                continue
            val, why = v
            if np.isfinite(val) and val > 0 and abs(math.log(val / rr['cur'])) > 1e-12:
                hist.append((nm, rr['cur'], val, why)); rr['cur'] = val
        if hist:
            out[i] = hist
    return out


if __name__ == "__main__":
    H = HERE + '/inputs/'
    hs = pd.read_csv(H + 'h3_src.csv').set_index('id0')
    f3 = pd.read_csv(H + 'pred_t077_id1.csv'); cur = pd.Series(f3.parsed_value.values, index=f3.id.values - 1)
    T = {i: r for i, r in h.ROWS['test'].items() if i in hs.index}
    res = run(T, cur)
    pd.DataFrame([dict(id0=i, tag="+".join(x[0] for x in hist), new=hist[-1][2]) for i, hist in sorted(res.items())]).to_csv(OUT + '/fired_test.csv', index=False)
