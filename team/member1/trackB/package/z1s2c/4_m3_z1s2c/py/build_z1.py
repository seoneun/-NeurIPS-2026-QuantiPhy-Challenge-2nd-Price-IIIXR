import sys, os, json
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)) + "/../in"
SUBS = sys.argv[1].rstrip("/")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fusion_calib as FC
rd = lambda n: pd.read_csv(n, float_precision="round_trip")

f_y1s = f"{SUBS}/pred_t144_id1.csv"; f_y1 = f"{SUBS}/pred_t143_id1.csv"
f_m3 = sys.argv[2]
Y1S = rd(f_y1s); base = Y1S.set_index("id").parsed_value
M3 = rd(f_m3).set_index("id").parsed_value; Y1 = rd(f_y1).set_index("id").parsed_value
X = pd.read_csv(H + "/diff_attrib_r67.csv").set_index("id")
D48 = pd.read_csv(H + "/pred_t002_detail.csv"); D48.index = D48["Unnamed: 0"] + 1
D46 = pd.read_csv(H + "/pred_t113_detail.csv"); D46.index = D46["Unnamed: 0"] + 1
N48 = rd(H + "/pred_t002_id1.csv").set_index("id").parsed_value
CG = pd.read_csv(H + "/pred_t059_change_groups.csv").set_index("id1")
AUD = pd.read_csv(H + "/audit_test.csv").set_index("id1")
PAR = FC.from_json(json.load(open(H + "/fusion_params.json"))["default"])
R3_CATS = ("D3",)
SEL = {"agree_gm", "agree_wgm", "dis_geo", "dis_caw_routed"}
NOGEO = {"caw_nogeo", "rul263_caw", "caw_c1"}
GEO_RULES = {"rul220"}
R66_IMPLAUSIBLE = {306, 3146}


def tainted(mode, src):
    m = str(mode)
    return ("fovdefault" in m or "calibrejected" in m) and str(src) not in NOGEO


def rec(i):
    x = X.loc[i] if i in X.index else None
    d48 = D48.loc[i]; d46 = D46.loc[i]
    return dict(id=int(i), base=float(base[i]), b17=(float(x.b17) if x is not None else None),
                verdict=(x.verdict if x is not None else None), ugrp=(x.ugrp if x is not None else None),
                unit=(x.unit if x is not None else None), in_r66=(x.in_r66 if x is not None and isinstance(x.in_r66, str) else ""),
                cat=str(d48["cat"]), n48=float(N48[i]), src48=str(d48.answer_source), mode48=d48["mode"], mode46=d46["mode"],
                src46=str(d46.answer_source),
                cf=dict(g=d48.g, cs=d48.cs, mode=d48["mode"], cat=d48["cat"], src=d48["src"], kind=d48["kind"]),
                pred_t059=bool(CG.pred_t059_changed.get(i, False)), cfull=float(AUD.cfull.get(i, np.nan)))


def decide_one(cf):
    df = pd.DataFrame([cf])
    return float(FC.decide(PAR, df)[0])


def rule(r):
    v, tag = r["base"], ""
    if r["id"] in R66_IMPLAUSIBLE: return v, tag
    vd, ug, un = r["verdict"], r["ugrp"], r["unit"]
    if vd == "LEGAL" and ug == "meas_rule":
        geo_val = not any(np.isfinite(c) and np.isclose(r["b17"], c, rtol=1e-3) for c in (float(r["cf"]["cs"]), r["cfull"]))
        if not (un in GEO_RULES and geo_val and tainted(r["mode48"], r["src48"])): v, tag = r["b17"], "R1:" + un
    elif vd == "LEGAL" and ug == "48a_origin":
        if not r["in_r66"] and not tainted(r["mode48"], r["src48"]): v, tag = r["b17"], "R2:" + un.replace("48a:patch:", "")
    elif vd == "LEGAL_PIPELINE":
        if not tainted(r["mode48"], r["src48"]): v, tag = r["b17"], "R4:" + un.replace("48a:", "")
    elif vd == "LEGAL_IF_DISCLOSED":
        if not tainted(r["mode46"], r["src46"]): v, tag = r["b17"], "R5:pred_t113_S3"
    if r["cat"] in R3_CATS and r["pred_t059"] and r["src48"] in SEL and not tainted(r["mode48"], r["src48"]):
        carries48 = np.isclose(v, r["n48"], rtol=1e-6)
        if (vd == "LEGAL" and ug == "calib_fusion") or carries48:
            c = decide_one(r["cf"])
            if np.isfinite(c) and c > 0: v, tag = c, "R3:pred_t059" + ("" if vd == "LEGAL" and ug == "calib_fusion" else "_whole")
    return v, tag


def build(order):
    out = {}
    for i in order: out[i] = rule(rec(i))
    return out


ids = list(base.index)
REF = build(ids)
z1 = pd.Series({i: REF[i][0] for i in ids})
d = Y1S.copy(); d["parsed_value"] = [float(z1[i]) for i in d.id]
d.to_csv(f"{SUBS}/pred_t145_id1.csv", index=False, lineterminator="\r\n")
