import sys, os, json, math, copy, random, importlib.util
sys.path.insert(0, os.environ["QP_WORK"] + "/h")
import harness as H, numpy as np, pandas as pd
RR = os.environ["QP_R"]; AG = f"{RR}/analysis/v3/ag"; OD = sys.argv[1]; OUT = f"{OD}/c9u"
os.makedirs(OUT, exist_ok=True)
_MOD = {}
def load(path, name):
    if name in _MOD: return _MOD[name]
    sys.path.insert(0, os.path.dirname(path))
    sp_ = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(m)
    _MOD[name] = m; return m

PT = {sp: {int(k): v for k, v in json.load(open(f"{AG}/rmap6/D3GPU/patches/ag_rul258_patch_{sp}.json")).items()} for sp in ("test", "val")}
OFF = {sp: pd.read_csv(f"{AG}/rmap6/D3GPU/off_{sp}_rul258.csv", index_col=0) for sp in ("test", "val")}
ROWS0 = H.ROWS
S1 = {sp: H.base(sp) for sp in ("test", "val")}
T = {sp: sorted(PT[sp]) for sp in ("test", "val")}
PRE = {"test": {}, "val": {}}; KIND = {"test": {}, "val": {}}
for sp in ("test", "val"):
    for i in T[sp]:
        v = float(PT[sp][i]["value"]); a = float(OFF[sp].answer[i]); b = float(OFF[sp].rS5[i]); s = float(S1[sp][i])
        if abs(math.log(s / v)) < 1e-12:
            PRE[sp][i] = s * b / a; KIND[sp][i] = "d3gpu_in_s1"
        else:
            PRE[sp][i] = s; KIND[sp][i] = "overwritten_in_s1"
print("targets", {sp: len(T[sp]) for sp in T}, "kinds", {sp: pd.Series(KIND[sp]).value_counts().to_dict() for sp in KIND}, flush=True)

def rows_for(start, ids=None, order=None):
    out = {}
    for sp in ("test", "val"):
        ii = T[sp] if ids is None else ids.get(sp, [])
        if order is not None: ii = order(sp, list(ii))
        d = {}
        for i in ii:
            r = copy.deepcopy(ROWS0[sp][i]); r["s1"] = float(start[sp][i]); d[i] = r
        out[sp] = d
    return out

W9 = f"{H.W}/r9"
C2_ORDER = ["D_fusion", "F_textaudit", "E_kinematics", "K_D3", "L_S3", "M_S2", "R_wingspan"]
C2_SKIP = {"ag_r9_F_cls_caw", "ag_r9_L2b_mogefov_speed", "ag_r9_N1c_caw_mra_D2", "ag_r9_N2a_sim_unveto", "ag_r9_N3c_indep_majority",
           "ag_r9_S2_U1_s2plan_geo", "ag_r9_d3gnf_caw", "gkf_F2", "gkm_M2", "gkp_P2", "segbg_size_caw"}
MILD = [("GEO_MISSING", "ag_r10_GM_T_V2"), ("A_rul263", "ag_r9_d3gnf_caw"), ("F_textaudit", "ag_r9_F_cls_caw"),
        ("N_D2", "ag_r9_N2a_sim_unveto"), ("N_D2", "ag_r9_N3c_indep_majority"),
        ("MILD", "r14_U7_rul297"), ("MILD", "r14_rul106"), ("MILD", "r14_KINU_V1"), ("MILD", "r14_U8_vlb"),
        ("MILD", "r14_RADF_V1"), ("MILD", "r14_U2U11_dzlb")]
MODERATE = [("C_variants", "segbg_size_caw"), ("G_gkwide", "gkm_M2")]
def fam(k):
    p = f"{AG}/r14/MILD/rules.py" if k == "MILD" else f"{W9}/{k}/rules.py"
    return load(p, f"r14_{k}")
os.environ.setdefault("GMD", f"{AG}/r10/GEO_MISSING")
OLDX = f"{AG}/r16/OLDROUNDS"; NEWX = f"{AG}/r16/NEWROUNDS"
OLD_IDS = ["r16_c86blobpair", "r16_rul041", "r16_rul060", "r16_rul147", "r16_rul109", "r16_rul069", "r16_rul186",
           "r16_rul148", "r16_rul194", "r16_rul071", "r16_rul055", "r16_rul096", "r16_second_all", "r16_rul022",
           "r16_d11roadfam", "r16_hgt", "r16_rul105", "r16_rul313", "r16_rul048", "r16_c83vclr", "r16_rul093", "r16_rul079",
           "r16_rul227", "r16_rul428", "r16_rul304", "r16_rul323", "r16_rul332", "r16_rul315", "r16_rul051_all", "r16_c80u",
           "r16_c94", "r16_c83_all", "r16_rul319", "r16_c88", "r16_c84", "r16_d10", "r16_c58", "r16_c47bcut", "r16_rul412_all", "r16_rul191"]
C4S_SKIP = {"D_U4c", "r16_rul148", "r16_rul315"}
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
       ("D_U4c", f"{W9}/D_fusion/rules.py", "r16_D", "U4c"),
       ("I_U1_V1a", f"{NEWX}/py/rules_reimpl.py", "r16_reimpl", "I_U1_V1a"),
       ("K_U2_V3", f"{W9}/K_D3/rules.py", "r16_K", "U2_V3"),
       ("L2b_V1", f"{NEWX}/py/rules_reimpl.py", "r16_reimpl", "L2b_V1"),
       ("M_U3_V1", f"{W9}/M_S2/rules_u3.py", "r16_Mu3", "U3_V1"),
       ("MO_V1", f"{AG}/r10/MOTION/rules.py", "r16_MO", "ag_r10_MOTION_V1_jumpy")]
def wit_d(r, x):
    w = [v for v in (r.get("s1_g"), r.get("s1_cs")) if v is not None and np.isfinite(v) and v > 0]
    return min(abs(math.log(x / v)) for v in w) if w else None
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
U17 = [("REFIRE", "r17_d10_V2", "REFIRE/patches/r17_d10_V2"), ("REFIRE", "r17_rul191_V2", "REFIRE/patches/r17_rul191_V2"),
       ("REFIRE", "r17_kin_A", "REFIRE/patches/r17_kin_A"),
       ("D3DEEP", "U1_occ", "D3DEEP/patches/r17_U1_occ"), ("D3DEEP", "U2_short_guard", "D3DEEP/patches/r17_U2_short"),
       ("D3DEEP", "U3_tw", "D3DEEP/patches/r17_U3_tw"), ("D3DEEP", "U4_apx_V2", "D3DEEP/patches/r17_U4_apx_V2"),
       ("D3DEEP", "U5_rest", "D3DEEP/patches/r17_U5_rest"), ("D3DEEP", "U6_kch", "D3DEEP/patches/r17_U6_kch"),
       ("S2DEEP", "S_V1", "S2DEEP/patches/S2DEEP_S_V1")]
U18 = [("D3ROUTE", "r18_W3_cf_V1L", "D3ROUTE/patches/r18_W3_cf_V1L"),
       ("D2DEEP", "r18_W_V1_qwin", "D2DEEP/patches/r18_W_V1"), ("D2DEEP", "r18_F_V1_fallalt", "D2DEEP/patches/r18_F_V1"),
       ("S3DEEP", "r18_U1_rest_V2", "S3DEEP/patches/r18_U1_rest_V2"), ("S3DEEP", "r18_U2_facade", "S3DEEP/patches/r18_U2_facade"),
       ("S3DEEP", "r18_U3_rul428_own", "S3DEEP/patches/r18_U3_swap"), ("S3DEEP", "r18_U5_rul268_own", "S3DEEP/patches/r18_U5_disc")]
PAT = {}
for rd, U in (("r17", U17), ("r18", U18)):
    for _, nm, p in U:
        PAT[nm] = {sp: {int(k): v for k, v in json.load(open(f"{AG}/{rd}/{p}_{sp}.json")).items()} for sp in ("test", "val")}
PATCH_HITS = []
RFN = {}
def _r17fns():
    if RFN: return RFN
    RU16 = load(f"{OLDX}/py/rules.py", "r16_old_rules"); D3 = load(f"{AG}/r17/D3DEEP/py/rules.py", "r17_d3deep_rules")
    CDR = f"{AG}/r17/REFIRE/cache"
    def refire(nm, gate):
        f = RU16.make(nm, "fixed", gate)
        def g(r):
            old = os.environ.get("R16_CACHE"); os.environ["R16_CACHE"] = CDR
            try:
                return f(r)
            finally:
                if old is None: os.environ.pop("R16_CACHE", None)
                else: os.environ["R16_CACHE"] = old
        return g
    RFN.update({"r17_d10_V2": refire("r17_d10_V2", ("skip", 1 / 1.4, 1.4)), "r17_rul191_V2": refire("r17_rul191_V2", ("outside_m", 1.5, 2.3)),
                "r17_kin_A": refire("r17_kin_A", None), "U6_kch": D3.VARIANTS["U6_kch"]})
    return RFN
def mkpatch(nm, earlier):
    def g(r):
        e_ = PAT[nm][r["split"]].get(int(r["id"]))
        if nm in _r17fns():
            v_ = RFN[nm](r)
            if v_ is None:
                if e_ is not None: PATCH_HITS.append(dict(stage=nm, split=r["split"], id=int(r["id"]), before=e_.get("before"), cur=float(r["cur"]), value=None, why="rule does not fire"))
                return None
            v_, w_ = v_ if isinstance(v_, tuple) else (v_, "")
            if e_ is None: PATCH_HITS.append(dict(stage=nm, split=r["split"], id=int(r["id"]), before=None, cur=float(r["cur"]), value=float(v_), why="NEW fire: " + str(w_)[:150]))
            e_ = dict(value=float(v_), why=w_, before=(e_ or {}).get("before"))
        if e_ is None: return None
        v = float(e_["value"]); c = float(r["cur"])
        PATCH_HITS.append(dict(stage=nm, split=r["split"], id=int(r["id"]), before=e_.get("before"), cur=c, value=v, why=str(e_.get("why", ""))[:200]))
        if not np.isfinite(v) or v <= 0: return None
        prev = earlier[r["split"]].get(int(r["id"]))
        if prev is not None and abs(math.log(v / c)) > 1e-12:
            de, dn = wit_d(r, c), wit_d(r, v)
            if de is not None and dn > de + 1e-12: return None
        return v, str(e_.get("why", ""))
    g.__name__ = nm; return g
NX = load(f"{AG}/r19/NEXT/py/rules.py", "next_rules")
LS = load(f"{AG}/r21/L0_S/py/rules.py", "l0s_rules"); LD = load(f"{AG}/r21/L0_D/py/rules.py", "l0d_rules")
RB = load(f"{AG}/r23b/py/rules.py", "r23b_rules")

def ap(fn, cur):
    out = {}
    for sp in ("test", "val"):
        out[sp] = H.apply(fn, sp, cur)[0] if len(H.ROWS[sp]) else cur[sp].copy()
    return out
def seq_earlier(cur, fns, wrapper):
    E = {"test": {}, "val": {}}
    for nm, fn in fns:
        snap = {sp: dict(E[sp]) for sp in E}; n = ap(wrapper(nm, fn, snap), cur)
        for sp in ("test", "val"):
            for i in n[sp].index[np.abs(np.log(n[sp] / cur[sp].reindex(n[sp].index))) > 1e-12]: E[sp][int(i)] = nm
        cur = n
    return cur
def chain21(cur):
    units = [nm_fn for nm_fn in LS.UNITS] + [nm_fn for nm_fn in LD.ORDER]
    cur = {sp: cur[sp].copy() for sp in cur}; by = {sp: {} for sp in cur}
    for nm, fn in units:
        for sp in ("test", "val"):
            if not len(H.ROWS[sp]): continue
            n, w = H.apply(fn, sp, cur)
            for i in list(w):
                if i in by[sp]:
                    r = H.ROWS[sp][i]; dp, dn = wdist(r, float(cur[sp][i])), wdist(r, float(n[i]))
                    if dp is not None and dn is not None and dn > dp:
                        n[i] = cur[sp][i]; del w[i]; continue
                by[sp][i] = nm
            cur[sp] = n
    return cur

def stages_list():
    st = []
    c2 = [(nm, fn) for k in C2_ORDER for nm, fn in fam(k).ACCEPTED if nm not in C2_SKIP]
    st.append(("c2", lambda c: _plain(c, c2)))
    c3 = [(nm, dict(fam(k).ACCEPTED)[nm]) for k, nm in MILD + MODERATE]
    st.append(("c3b", lambda c: _plain(c, c3)))
    RU16 = load(f"{OLDX}/py/rules.py", "r16_old_rules"); OLDFN = dict(RU16.ACCEPTED)
    cand = [(n, OLDFN[n]) for n in OLD_IDS] + [(c_, fget(load(p, mn), k)) for c_, p, mn, k in NEW]
    cand = [(n, f) for n, f in cand if n not in C4S_SKIP]
    st.append(("c4s", lambda c: seq_earlier(c, cand, wrap16)))
    st.append(("c5", lambda c: seq_earlier(c, [(nm, None) for _, nm, _ in U17], lambda nm, fn, e: mkpatch(nm, e))))
    st.append(("c6", lambda c: seq_earlier(c, [(nm, None) for _, nm, _ in U18], lambda nm, fn, e: mkpatch(nm, e))))
    st.append(("c7", lambda c: _plain(c, [("r19_U1_V1", NX.U1_V1)])))
    st.append(("c8", chain21))
    st.append(("c9b", lambda c: _plain(c, [(f"r23b_{k}", fn) for k, fn in enumerate(RB.stages())])))
    return st
def _plain(cur, fns):
    for nm, fn in fns: cur = ap(fn, cur)
    return cur
STAGES = stages_list()

def run(start, ids=None, order=None, others=None):
    H.ROWS = rows_for(start, ids, order)
    try:
        cur = {sp: pd.Series({i: float(start[sp][i]) for i in H.ROWS[sp]}, dtype=float) for sp in ("test", "val")}
        if others is not None:
            for sp in ("test", "val"): cur[sp] = pd.concat([cur[sp], others[sp].drop(cur[sp].index, errors="ignore")])
        out = {"s1": {sp: cur[sp].copy() for sp in cur}}
        for nm, f in STAGES:
            cur = f(cur); out[nm] = {sp: cur[sp].copy() for sp in cur}
        return out
    finally:
        H.ROWS = ROWS0

LG = lambda a, b: abs(math.log(float(a) / float(b)))

PATCH_HITS.clear(); ctl = run(S1); ctl_hits = list(PATCH_HITS)
PATCH_HITS.clear(); cf = run(PRE); cf_hits = list(PATCH_HITS)
rows = []
for sp in ("test", "val"):
    for i in T[sp]:
        r = ROWS0[sp][i]
        d = dict(split=sp, id=i, sub_id=i + 1, video_id=r["video_id"], cat=r["cat"], q=r["question"][:120], kind=KIND[sp][i],
                 d3gpu_why=str(PT[sp][i].get("why", ""))[:160])
        for nm in ctl: d[f"ctl_{nm}"] = float(ctl[nm][sp][i]); d[f"cf_{nm}"] = float(cf[nm][sp][i])
        d["changed_final"] = LG(cf["c9b"][sp][i], ctl["c9b"][sp][i]) > 1e-12
        tr = [nm for a, nm in zip(list(cf), list(cf)[1:]) if LG(cf[a][sp][i], cf[nm][sp][i]) > 1e-12]
        tc = [nm for a, nm in zip(list(ctl), list(ctl)[1:]) if LG(ctl[a][sp][i], ctl[nm][sp][i]) > 1e-12]
        d["cf_writers"] = "|".join(tr); d["ctl_writers"] = "|".join(tc)
        rows.append(d)
df = pd.DataFrame(rows)
for sp in ("test", "val"):
    s = df[(df.split == sp) & df.changed_final]
    json.dump({str(int(r.id)): dict(value=float(r.cf_c9b), before=float(r.ctl_c9b),
                                    why=f"r37 NOUNI: D3GPU focal gm(MoGe-2,UniDepthV2) -> MoGe-2 only; chain s1->c9b replayed (s1 {r.ctl_s1:.6g} -> {r.cf_s1:.6g}; writers {r.cf_writers or '-'})")
               for r in s.itertuples()}, open(f"{OUT}/cf_{sp}.json", "w"), indent=0)


CT = {int(k): v for k, v in json.load(open(f"{OUT}/cf_test.json")).items()}
BASECSV = f"{OD}/pred_t127_id1.csv"; OUTCSV = f"{OD}/pred_t128_id1.csv"
a = pd.read_csv(BASECSV, dtype=str, keep_default_na=False); b = a.copy(); pos = {int(x) - 1: j for j, x in enumerate(a.id)}
for i, e in CT.items():
    j = pos[i]
    b.loc[j, "parsed_value"] = repr(float(e["value"]))
b.to_csv(OUTCSV, index=False)
