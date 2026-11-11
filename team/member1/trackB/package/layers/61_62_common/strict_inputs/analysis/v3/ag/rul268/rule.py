import os, sys, re, json, glob
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
SUB = sys.argv[1] if len(sys.argv) > 1 else "pred_t090"
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
PATCHES = ("dim,lane,kin,hgt,rul264,jump,second,rul001,rul439,wing,twin,roll,rul280,rul265,rul285,rul298,rul413,rul006,wlen,"
           "rul267,rul412,rul303,rul005,rul428,rul424,orbit,rul305,rul282,disc,rul269,rul323,rul432,rul324,p1pf,rul436,"
           "rul438,rul301,rul297").split(",")
pd.set_option('display.width', 400); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 60)

SPH = r"\b(ball|balls|basketball|football|volleyball|baseball|globe|sphere|orb|marble|bubble|balloon|planet|moon|jupiter|io|europa|ganymede|callisto|orange|apple|egg|bead|pearl)\b"
CYL = r"\b(bottle|neck|tube|cup|mug|glass|jar|can|pipe|pole|trunk|cylinder|handle|vase|pot|bucket|column|candle|log|container|planter|plant|table|rim|stem|leg|post|pillar|tower|chimney|barrel|drum|pocket|pockets|cone)\b"
WHEEL = r"\b(wheel|wheels|tire|tires|tyre|tyres)\b"
DISC = (r"\b(lampshade|lampshades|shade|plate|plates|dish|saucer|coin|coins|disc|discs|disk|disks|cd|frisbee|pizza|pepperoni|pie|"
        r"pancake|cookie|cookies|biscuit|clock|hoop|manhole|lid|tray|carpet|rug|cell|cells|donut|doughnut|bagel|button|medal|"
        r"mirror|target|umbrella|trampoline|tape|cymbal|gear|pulley|fan)\b")
FLAT = (r"\b(?:chinese\s+chess|xiangqi|(?:red|black)\s+(?:general|chariot|horse|elephant|advisor|cannon|soldier)\s+chess|"
        r"checkers?|draughts?|pucks?|go\s+stones?|poker\s+chips?|tokens?|coasters?)\b")
STRONG = ("2d_ratio", "3d_calib")
LG2 = np.log10(2.0)
MINF, MAXF, MINPX = 1.10, 2.6, 15.0
XG_TOL = np.log(1.5)
REPRO_TOL = np.log(1.05)
UNIT = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "km": 0.001, "ft": 1 / 0.3048, "in": 1 / 0.0254, "inch": 1 / 0.0254, "inches": 1 / 0.0254}


def dclass(p):
    p = str(p).lower()
    if not p or p == "nan" or re.search(SPH, p) or (re.search(CYL, p) and not re.search(WHEEL, p)):
        return None
    if re.search(WHEEL, p) and "steering" not in p:
        return "wheel"
    if re.search(DISC, p):
        return "disc"
    if re.search(FLAT, p) and not re.search(CYL, p):
        return "flat"
    return None


def fuse(row, gnew):
    src = str(row.answer_source); cs = row.cs; mode = str(row["mode"])
    strong = mode in STRONG
    size = row["kind"] == "size"
    have_c = np.isfinite(cs) and cs > 0
    agree = have_c and abs(np.log10(gnew / cs)) <= LG2
    if src in ("geo_cawinvalid", "geo_only_mode", "agree_geo_kind"):
        return gnew, src
    if src in ("agree_gm", "agree_wgm", "dis_geo", "dis_size_gm"):
        w = 0.65 if (strong and size) else 0.5
        if not have_c:
            return gnew, "geo_nocaw"
        if agree:
            return float(gnew ** w * cs ** (1 - w)), f"agree_w{w:.2f}"
        if strong:
            return gnew, "dis_geo"
        if size:
            return float(np.sqrt(gnew * cs)), "dis_size_gm"
        return None, "dis_weak_motion"
    return None, "unhandled:" + src






