import os, sys, json, re
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)) + "/../in"
S = sys.argv[1].rstrip("/")
rd = lambda p: pd.read_csv(p, float_precision="round_trip")
f_y1s, f_z1, f_y1 = f"{S}/pred_t144_id1.csv", f"{S}/pred_t145_id1.csv", f"{S}/pred_t143_id1.csv"
Y1S = rd(f_y1s); y1s = Y1S.set_index("id").parsed_value; z1 = rd(f_z1).set_index("id").parsed_value; y1 = rd(f_y1).set_index("id").parsed_value
PA = rd(H + "/z1s_A_rows.csv").set_index("id"); PB = rd(H + "/z1s_B_rows.csv").set_index("id")
K2 = rd(H + "/k2.csv").set_index("id1"); K3 = rd(H + "/k3.csv").set_index("id1"); K5 = rd(H + "/k5.csv").set_index("id1")
s_ = open(H + "/rul343.txt", encoding="utf-8").read(); parts = re.split(r"@@(r1_\w+)\.json\n", s_)[1:]
R1J = {parts[i]: json.loads(parts[i + 1]) for i in range(0, len(parts), 2)}
OWN_C = {"crop", "crop112", "full", "full112", "tshift30"}
R2_DROP = {"rul413", "rul131", "disc", "rul153", "rul281", "rul062"}
R2_PIX = {"rul267", "rul057", "rul237", "rul019", "rul145", "rul210", "rul082", "rul112",
          "rul011", "rul069", "rul079", "rul256", "orbit"}
SETS = ("m", "v", "mu", "vu")


def own_c(i):
    c = str(K3.c_src[i]); return c == "nan" or bool(set(c.split("|")) & OWN_C)


def meas_ok(i):
    k = K5.loc[i]
    return all(bool(k[f"{s}_E"]) for s in SETS) and all(str(k[f"{s}_A_eq_old"]) in ("True", "nan", "None") for s in SETS)


def chain_ok(i):
    k = K2.loc[i]
    if str(k.q_answer_source) not in ("meas", "vlm", "h1") or str(k.u_answer_source) not in ("meas", "vlm", "h1"): return False
    for a_, m_ in ((k.q_answer_source, k.q_merge_src), (k.u_answer_source, k.u_merge_src)):
        if str(a_) == "meas" and str(m_) not in ("d", "v", "xg_v"): return False
    return not any(x in str(k.mode48) for x in ("scene", "fovdefault", "calibrejected"))


def gfin(i, grp):
    g = K2.g46[i] if grp == "R5" else K2.g48[i]
    return bool(np.isfinite(pd.to_numeric(g, errors="coerce")))


def P2(i, grp, unit):
    if not isinstance(K2.folds.get(i), str): return False, "no fold"
    if grp == "R4" and unit == "sruler_geo": return False, "scene ruler"
    if grp == "R1":
        e = R1J.get(f"r1_{K2.folds[i]}", {}).get(unit, {}).get(str(i - 1))
        if e is None or not any(np.isclose(round(e[0], d_), z1[i], rtol=1e-12, atol=1e-12) for d_ in (6, 7)):
            return False, "R1 rerun differs"
    if not own_c(i): return False, "CaW not own"
    if not chain_ok(i): return False, "ruler/scene/focal chain"
    if not meas_ok(i): return False, "meas not reproduced own-only"
    if grp == "R2" and unit in R2_PIX and int(K2.n_other_q[i]) != 0: return False, "pixel rule, other queries in file"
    return True, "P2"


def verdict_A(i):
    tag = PA.tag[i]; grp, unit = tag.split(":", 1)
    if grp == "R2" and unit in R2_DROP: return False, f"text:{unit}"
    if bool(PA.taint[i]): return False, "mode taint"
    p1 = bool(PA.g_own[i]) and bool(PA.c_own[i]) and (grp in ("R3", "R4", "R5") or bool(PA.own_files[i])) and gfin(i, grp)
    p2, w2 = P2(i, grp, unit)
    if p1 and p2: return True, "P1+P2"
    if p1: return True, "P1"
    if p2: return True, "P2"
    return False, w2


def verdict_B(i):
    rule = str(PB.rule[i])
    if rule == "rul062": return False, "text:rul062"
    if bool(PB.taint[i]): return False, "mode taint"
    if rule in ("rul271", "rul073"):
        if not isinstance(K2.folds.get(i), str): return False, "no fold"
        if own_c(i) and chain_ok(i) and meas_ok(i): return True, "C1-free own fusion, raw geo P2"
        return False, "raw geo not proven"
    ok = bool(PB.own_files[i]) and bool(PB.g_own[i]) and bool(PB.c_own[i]) and bool(np.isfinite(pd.to_numeric(K2.g48.get(i), errors="coerce")))
    return (True, "own_files gate == own gate") if ok else (False, "gate read other queries / g,c")


VA = {i: verdict_A(i) for i in PA.index}; VB = {i: verdict_B(i) for i in PB.index}
new = y1s.copy()
for i, (k, _) in VA.items():
    if k: new[i] = z1[i]
for i, (k, _) in VB.items():
    if k: new[i] = y1[i]
ids = list(PA.index) + list(PB.index); sh = ids[:]; np.random.RandomState(68).shuffle(sh)
out = f"{S}/pred_t146_id1.csv"
O = Y1S.copy(); O["parsed_value"] = new.loc[O.id].values; O.to_csv(out, index=False, lineterminator="\r\n")
