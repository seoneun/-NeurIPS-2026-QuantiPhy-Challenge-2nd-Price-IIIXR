import sys, os, json, math, importlib.util
sys.path.insert(0, os.environ["QP_WORK"] + "/h")
import harness as H, numpy as np, pandas as pd
_N = {'c2': 'pred_t015', 'c3a': 'pred_t115', 'c3b': 'pred_t116', 'c3m': 'pred_t117', 'c4': 'pred_t118', 'c4m': 'pred_t119', 'c4s': 'pred_t120', 'c5': 'pred_t121', 'c5m': 'pred_t021', 'c6': 'pred_t122', 'c7': 'pred_t123', 'c8': 'pred_t124', 'c9': 'pred_t125', 'c9b': 'pred_t126', 'c9n': 'pred_t127', 'c9u': 'pred_t128', 'g2': 'pred_t129', 'g2a': 'pred_t130', 'g3': 'pred_t131', 'g3a': 'pred_t132', 'i1': 'pred_t133', 'j1': 'pred_t134', 'j6': 'pred_t135', 'j7': 'pred_t136', 'j7s': 'pred_t137', 'j9n': 'pred_t138', 'm3': 'pred_t139', 'p1': 'pred_t140', 'r1': 'pred_t141', 's1': 'pred_t142', 'y1': 'pred_t143', 'y1s': 'pred_t144', 'z1': 'pred_t145', 'z1s2': 'pred_t146', 'z1s2c': 'pred_t004'}
RR = os.environ["QP_R"]; AG = f"{RR}/analysis/v3/ag"; OUT = sys.argv[1]; SER = f"{OUT}/series"; os.makedirs(SER, exist_ok=True)
C2CSV = os.environ["C2CSV"]
BAD = ("H.OLD", "OLD_SRC", "H.B0", "submissions/", "pred_t002")
SPS = ("test", "val")
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def chg(a, b): return a.index[np.abs(np.log(a / b.reindex(a.index))) > 1e-12]
def wit_d(r, x):
    w = [v for v in (r.get("s1_g"), r.get("s1_cs")) if v is not None and np.isfinite(v) and v > 0]
    return min(abs(math.log(x / v)) for v in w) if w else None
RES = {}
def save(nm, cur):
    RES[nm] = {sp: cur[sp].copy() for sp in SPS}
    for sp in SPS: pd.DataFrame({sp: cur[sp]}).to_csv(f"{SER}/{_N[nm]}_{sp}.csv")
    print("BUILT", nm, flush=True)

X14 = f"{AG}/r14"
C2_ORDER = ["D_fusion", "F_textaudit", "E_kinematics", "K_D3", "L_S3", "M_S2", "R_wingspan"]
C2_SKIP = {"ag_r9_F_cls_caw", "ag_r9_L2b_mogefov_speed", "ag_r9_N1c_caw_mra_D2", "ag_r9_N2a_sim_unveto", "ag_r9_N3c_indep_majority",
           "ag_r9_S2_U1_s2plan_geo", "ag_r9_d3gnf_caw", "gkf_F2", "gkm_M2", "gkp_P2", "segbg_size_caw"}
_MOD = {}
def fam(k):
    if k in _MOD: return _MOD[k]
    p = f"{X14}/MILD/rules.py" if k == "MILD" else f"{H.W}/r9/{k}/rules.py"
    sys.path.insert(0, os.path.dirname(p)); _MOD[k] = load(f"r14_{k}", p); return _MOD[k]
cur = {sp: H.base(sp) for sp in SPS}
for k in C2_ORDER:
    for nm, fn in fam(k).ACCEPTED:
        if nm in C2_SKIP: continue
        cur = {sp: H.apply(fn, sp, cur)[0] for sp in SPS}
save("c2", cur)
KEEP14 = {d["rule"]: d["kept"] for d in json.load(open(f"{X14}/out/stack14.json"))["rules"]}
MILD = [("GEO_MISSING", "ag_r10_GM_T_V2"), ("A_rul263", "ag_r9_d3gnf_caw"), ("F_textaudit", "ag_r9_F_cls_caw"),
        ("N_D2", "ag_r9_N2a_sim_unveto"), ("N_D2", "ag_r9_N3c_indep_majority"),
        ("MILD", "r14_U7_rul297"), ("MILD", "r14_rul106"), ("MILD", "r14_KINU_V1"), ("MILD", "r14_U8_vlb"),
        ("MILD", "r14_RADF_V1"), ("MILD", "r14_U2U11_dzlb")]
MODERATE = [("C_variants", "segbg_size_caw"), ("G_gkwide", "gkm_M2")]
for lst, nmb in ((MILD, "c3a"), (MODERATE, "c3b")):
    for k, nm in lst:
        if not KEEP14[nm]: continue
        fn = dict(fam(k).ACCEPTED)[nm]
        cur = {sp: H.apply(fn, sp, cur)[0] for sp in SPS}
    save(nmb, cur)

OLDX = f"{AG}/r16/OLDROUNDS"; NEWX = f"{AG}/r16/NEWROUNDS"; R9 = f"{H.W}/r9"
os.environ.setdefault("GMD", f"{AG}/r10/GEO_MISSING")
sys.path.insert(0, f"{OLDX}/py")
RU16 = load("r16_oldrounds_rules", f"{OLDX}/py/rules.py")
OLD_IDS = ["r16_c86blobpair", "r16_rul041", "r16_rul060", "r16_rul147", "r16_rul109", "r16_rul069", "r16_rul186",
           "r16_rul148", "r16_rul194", "r16_rul071", "r16_rul055", "r16_rul096", "r16_second_all", "r16_rul022",
           "r16_d11roadfam", "r16_hgt", "r16_rul105", "r16_rul313", "r16_rul048", "r16_c83vclr", "r16_rul093", "r16_rul079",
           "r16_rul227", "r16_rul428", "r16_rul304", "r16_rul323", "r16_rul332", "r16_rul315", "r16_rul051_all", "r16_c80u",
           "r16_c94", "r16_c83_all", "r16_rul319", "r16_c88", "r16_c84", "r16_d10", "r16_c58", "r16_c47bcut", "r16_rul412_all", "r16_rul191"]
OLDFN = dict(RU16.ACCEPTED)
_M16 = {}
def load16(path, name):
    if name in _M16: return _M16[name]
    sys.path.insert(0, os.path.dirname(path)); _M16[name] = load(name, path); return _M16[name]
def fget(mod, key):
    if hasattr(mod, key) and callable(getattr(mod, key)): return getattr(mod, key)
    for tab in ("VARIANTS", "WVARIANTS", "VARIANTS_AB", "ALL"):
        t = getattr(mod, tab, None)
        if isinstance(t, dict):
            if key in t and callable(t[key]): return t[key]
            for v in t.values():
                if isinstance(v, list):
                    for f in v:
                        if getattr(f, "__name__", None) == key: return f
        elif isinstance(t, list):
            for e in t:
                if isinstance(e, tuple) and e[0] == key: return e[1]
    raise KeyError(key)
NEW = [("A_U5c", f"{NEWX}/py/rules_reimpl.py", "r16_reimpl", "A_U5c"),
       ("D_U4c", f"{R9}/D_fusion/rules.py", "r16_D", "U4c"),
       ("I_U1_V1a", f"{NEWX}/py/rules_reimpl.py", "r16_reimpl", "I_U1_V1a"),
       ("K_U2_V3", f"{R9}/K_D3/rules.py", "r16_K", "U2_V3"),
       ("L2b_V1", f"{NEWX}/py/rules_reimpl.py", "r16_reimpl", "L2b_V1"),
       ("M_U3_V1", f"{R9}/M_S2/rules_u3.py", "r16_Mu3", "U3_V1"),
       ("MO_V1", f"{AG}/r10/MOTION/rules.py", "r16_MO", "ag_r10_MOTION_V1_jumpy")]
CAND = [(n, OLDFN[n]) for n in OLD_IDS] + [(c, fget(load16(p, mn), k)) for c, p, mn, k in NEW]
def wrap16(nm, fn, earlier):
    def g(r):
        v = fn(r)
        if v is None: return None
        w = ""
        if isinstance(v, tuple): v, w = v
        if v is None or not np.isfinite(v) or v <= 0: return None
        e = float(r["cur"]); prev = earlier[r["split"]].get(int(r["id"]))
        if prev is not None and abs(math.log(v / e)) > 1e-12:
            de, dn = wit_d(r, e), wit_d(r, v)
            if de is not None and dn > de + 1e-12: return None
        return float(v), w
    g.__name__ = nm; return g
C3B = {sp: cur[sp].copy() for sp in SPS}
for NAME in ("pred_t118", "pred_t120"):
    L16 = json.load(open(f"{AG}/r16/STACK/out/stack_{NAME}.json")); SKIP16 = set(L16["skip"]); KEEP16 = {d["rule"]: d["kept"] for d in L16["rules"]}
    c = {sp: C3B[sp].copy() for sp in SPS}; EARLIER = {"test": {}, "val": {}}
    for nm, fn0 in CAND:
        if nm in SKIP16: continue
        snap = {sp: dict(EARLIER[sp]) for sp in EARLIER}
        nt, _ = H.apply(wrap16(nm, fn0, snap), "test", c); nv, _ = H.apply(wrap16(nm, fn0, snap), "val", c)
        if KEEP16[nm]:
            for sp, n_ in (("test", nt), ("val", nv)):
                for i in chg(n_, c[sp]): EARLIER[sp][int(i)] = nm
            c = {"test": nt, "val": nv}
    save({v: k for k, v in _N.items()}[NAME], c)
cur = {sp: RES["c4s"][sp].copy() for sp in SPS}

def patch_round(units, keep, tag):
    global cur
    PAT = {nm: {sp: {int(k): v for k, v in json.load(open(f"{p}_{sp}.json")).items()} for sp in SPS} for nm, p in units}
    EAR = {"test": {}, "val": {}}
    for nm, _ in units:
        if not keep[nm]: continue
        snap = {sp: dict(EAR[sp]) for sp in EAR}
        def g(r, nm=nm, snap=snap):
            e_ = PAT[nm][r["split"]].get(int(r["id"]))
            if e_ is None: return None
            v = float(e_["value"]); c = float(r["cur"])
            if not np.isfinite(v) or v <= 0: return None
            prev = snap[r["split"]].get(int(r["id"]))
            if prev is not None and abs(math.log(v / c)) > 1e-12:
                de, dn = wit_d(r, c), wit_d(r, v)
                if de is not None and dn > de + 1e-12: return None
            return v, str(e_.get("why", ""))
        nt, _ = H.apply(g, "test", cur); nv, _ = H.apply(g, "val", cur)
        for sp, n_ in (("test", nt), ("val", nv)):
            for i in chg(n_, cur[sp]): EAR[sp][int(i)] = nm
        cur = {"test": nt, "val": nv}
U17 = [("r17_d10_V2", "REFIRE/patches/r17_d10_V2"), ("r17_rul191_V2", "REFIRE/patches/r17_rul191_V2"), ("r17_kin_A", "REFIRE/patches/r17_kin_A"),
       ("U1_occ", "D3DEEP/patches/r17_U1_occ"), ("U2_short_guard", "D3DEEP/patches/r17_U2_short"), ("U3_tw", "D3DEEP/patches/r17_U3_tw"),
       ("U4_apx_V2", "D3DEEP/patches/r17_U4_apx_V2"), ("U5_rest", "D3DEEP/patches/r17_U5_rest"), ("U6_kch", "D3DEEP/patches/r17_U6_kch"),
       ("U8_curb", "D3DEEP/patches/r17_U8_curb"), ("S_V1", "S2DEEP/patches/S2DEEP_S_V1")]
K17 = {d["rule"]: d["kept"] for d in json.load(open(f"{AG}/r17/STACK/out/stack_pred_t121.json"))["units"]}
patch_round([(n, f"{AG}/r17/{p}") for n, p in U17], K17, "c5"); save("c5", cur)
U18 = [("r18_W3_cf_V1L", "D3ROUTE/patches/r18_W3_cf_V1L"), ("r18_W_V1_qwin", "D2DEEP/patches/r18_W_V1"), ("r18_F_V1_fallalt", "D2DEEP/patches/r18_F_V1"),
       ("r18_U1_rest_V2", "S3DEEP/patches/r18_U1_rest_V2"), ("r18_U2_facade", "S3DEEP/patches/r18_U2_facade"),
       ("r18_U3_rul428_own", "S3DEEP/patches/r18_U3_swap"), ("r18_U5_rul268_own", "S3DEEP/patches/r18_U5_disc")]
K18 = {d["rule"]: d["kept"] for d in json.load(open(f"{AG}/r18/STACK/out/stack_pred_t122.json"))["units"]}
patch_round([(n, f"{AG}/r18/{p}") for n, p in U18], K18, "c6"); save("c6", cur)

NX = load("next_rules", f"{AG}/r19/NEXT/py/rules.py")
for sp in SPS: cur[sp] = H.apply(NX.U1_V1, sp, cur)[0]
save("c7", cur)

LS = load("l0s_rules", f"{AG}/r21/L0_S/py/rules.py"); LD = load("l0d_rules", f"{AG}/r21/L0_D/py/rules.py")
U21 = [("L0_S", nm, fn) for nm, fn in LS.UNITS] + [("L0_D", nm, fn) for nm, fn in LD.ORDER]
by = {sp: {} for sp in SPS}
for k, nm, fn in U21:
    for sp in SPS:
        n, w = H.apply(fn, sp, cur)
        for i in list(w):
            if i in by[sp]:
                r = H.ROWS[sp][i]; dp, dn = wdist(r, float(cur[sp][i])), wdist(r, float(n[i]))
                if dp is not None and dn is not None and dn > dp:
                    n[i] = cur[sp][i]; del w[i]; continue
            by[sp][i] = nm
        cur[sp] = n
save("c8", cur)
C8 = {sp: cur[sp].copy() for sp in SPS}

SR = load("r23_scales", f"{AG}/r23/SCALES/py/rules.py"); WR = load("r23_weights", f"{AG}/r23/WEIGHTS/py/rules.py")
W_UNITS = list(WR.KEPT); S_UNITS = list(SR.KEPT); ALL = ["prescale"] + W_UNITS + [u for u in S_UNITS if u != "prescale"]
KEPT23 = json.load(open(f"{AG}/r23/STACK/out/final.json"))["kept"]
def pipeline(on, sp, base):
    on = [u for u in ALL if u in on]; c = base.copy()
    if "prescale" in on: c = H.apply(SR.make_rule(["prescale"]), sp, {sp: c})[0]
    for u in W_UNITS:
        if u in on: c = H.apply(WR.make(u), sp, {sp: c})[0]
    rest = [u for u in S_UNITS if u != "prescale" and u in on]
    if rest: c = H.apply(SR.make_rule(rest), sp, {sp: c})[0]
    return c
save("c9", {sp: pipeline(KEPT23, sp, C8[sp]) for sp in SPS})

RU23 = load("r23b_rules", f"{AG}/r23b/py/rules.py")
for sp in SPS:
    c = C8[sp].copy()
    for fn in RU23.stages({}):
        c = H.apply(fn, sp, {sp: c})[0]
    cur[sp] = c
save("c9b", cur)

I = 858; ROWS0 = H.ROWS
SV1 = {int(k): v for k, v in json.load(open(f"{AG}/r17/S2DEEP/patches/S2DEEP_S_V1_test.json")).items()}
pre = float(RES["c4s"]["test"][I])
stages = [("r19_U1_V1", NX.U1_V1)] + [(f"r21_{nm}", fn) for nm, fn in LS.UNITS] + [(f"r21_{nm}", fn) for nm, fn in LD.ORDER] + \
         [(f"r23b_{k}", fn) for k, fn in enumerate(RU23.stages())]
def replay858(start):
    H.ROWS = {s: {} for s in ROWS0}; H.ROWS["test"] = {I: ROWS0["test"][I]}
    try:
        c = pd.Series({I: start}, dtype=float)
        for nm, fn in stages: c = H.apply(fn, "test", {"test": c})[0]
        return float(c[I])
    finally:
        H.ROWS = ROWS0
V858 = replay858(pre); print("c9n row 858:", float(RES["c9b"]["test"][I]), "->", V858, "(pre-S_V1", pre, ")", flush=True)
c9n = {sp: RES["c9b"][sp].copy() for sp in SPS}; c9n["test"][I] = V858; save("c9n", c9n)


def csvp(nm): return f"{OUT}/{_N[nm]}_id1.csv"
def w_float(parent_csv, ref, new, out):
    sub = pd.read_csv(parent_csv, float_precision="round_trip"); s = sub.copy()
    ct = chg(new, ref); msk = (s.id - 1).isin(set(int(x) for x in ct)); s.loc[msk, "parsed_value"] = new.reindex(s.id[msk] - 1).values
    s.to_csv(out, index=False); return int(msk.sum())
def w_text(parent_csv, new, out):
    a = pd.read_csv(parent_csv, dtype=str, keep_default_na=False); b = a.copy(); k = 0
    for j in range(len(b)):
        i = int(b.id[j]) - 1; o = float(b.parsed_value[j]); v = float(new[i])
        if abs(math.log(v / o)) > 1e-12: b.loc[j, "parsed_value"] = repr(v); k += 1
    b.to_csv(out, index=False); return k
WR_ = {}
WR_["c3a"] = w_float(C2CSV, RES["c2"]["test"], RES["c3a"]["test"], csvp("c3a"))
WR_["c3b"] = w_float(C2CSV, RES["c2"]["test"], RES["c3b"]["test"], csvp("c3b"))
WR_["c4"] = w_float(csvp("c3b"), RES["c3b"]["test"], RES["c4"]["test"], csvp("c4"))
WR_["c4s"] = w_float(csvp("c3b"), RES["c3b"]["test"], RES["c4s"]["test"], csvp("c4s"))
WR_["c5"] = w_float(csvp("c4s"), RES["c4s"]["test"], RES["c5"]["test"], csvp("c5"))
WR_["c6"] = w_float(csvp("c5"), RES["c5"]["test"], RES["c6"]["test"], csvp("c6"))
WR_["c7"] = w_float(csvp("c6"), RES["c6"]["test"], RES["c7"]["test"], csvp("c7"))
WR_["c8"] = w_float(csvp("c7"), RES["c7"]["test"], RES["c8"]["test"], csvp("c8"))
WR_["c9"] = w_text(csvp("c8"), RES["c9"]["test"], csvp("c9"))
WR_["c9b"] = w_text(csvp("c8"), RES["c9b"]["test"], csvp("c9b"))
a = pd.read_csv(csvp("c9b"), dtype=str, keep_default_na=False); j = int(np.where(a.id.astype(int).values == I + 1)[0][0])
b = a.copy(); b.loc[j, "parsed_value"] = repr(V858); b.to_csv(csvp("c9n"), index=False); WR_["c9n"] = 1
print("CHAIN_DONE", flush=True)
