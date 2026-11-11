import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ["RA_Q"]); WO = os.path.abspath(os.environ["RA_OUT"])

pd.set_option("display.width", 280); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 500)
SUB = sys.argv[1]
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
TAG = "rul291"

WINGED = r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft)\b"
FLAT = (r"paper|sheet|letter|envelope|tray|mat|placemat|rug|carpet|towel|napkin|cloth|tablecloth|book|notebook|note book|"
        r"magazine|newspaper|card|cutting board|chopping board|blanket|doormat")
FLAT_RX = re.compile(r"\b(?:" + FLAT + r")s?$")
SIDE_RX = re.compile(r"longer side|shorter side|thickness")
SRC_OK = {"dis_geo", "dis_size_gm", "agree_gm", "agree_wgm", "rul263_dimgm", "d3lab_gm", "rul263_unstable_gm"}
ASPECT_RR, ASPECT_BOX, MIN_LONG, MAX_PRIOR_CUT = 2.5, 2.3, 15.0, 0.5
W_GEO = 0.7


def parse(split):
    d = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); d[int(j["id"])] = j
    return d


_TJ = {}


def track_json(split, vid, mt):
    k = (split, vid, mt)
    if k not in _TJ:
        if mt == "v1":
            cands = [f"{R}/geo/out/tracks_{split}/{vid}.json"]
        else:
            cands = [f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json"]
        p = next((c for c in cands if os.path.exists(c)), None)
        _TJ[k] = (json.load(open(p)), p) if p else (None, None)
    return _TJ[k]


def ent_keys(ent):
    if not ent or not ent.get("phrase"):
        return []
    rel = ent.get("rel")
    base = ent["phrase"] + "|" + ",".join(ent.get("where") or [])
    return ([base + "|" + rel[0] + ":" + rel[1]] if rel else []) + [base]


def frames_of(vj, ent):
    if vj is None:
        return None, None
    qs = {q["key"]: q for q in vj["queries"]}
    for k in ent_keys(ent):
        if k in qs:
            q = qs[k]
            if vj.get("version") == "v2":
                if q.get("obj") is None:
                    return None, k
                return [f.get("st") if f else None for f in vj["objs"][str(q["obj"])]], k
            return [f.get("st") if f else None for f in q["frames"]], k
    return None, None


def ext_frames(st):
    val = np.array([s is not None for s in st])
    area = np.array([s["area"] if s else 0 for s in st], float)
    med = np.median(area[val]) if val.any() else 0
    ok = val & (area > 0.4 * med) & (area < 2.5 * med) if med > 0 else val
    border = np.array([s["border"] if s else 9 for s in st])
    okc = np.where(ok)[0]
    oke = np.where(ok & (border == 0))[0]
    fb = False
    if len(oke) < max(2, 0.3 * len(okc)):
        oke = okc; fb = True
    return okc, oke, fb


def geom(st):
    okc, oke, fb = ext_frames(st)
    if len(oke) == 0:
        return None
    rr = np.array([st[j]["rr"][:2] for j in oke], float)
    bx = np.array([st[j]["box"] for j in oke], float)
    bw, bh = bx[:, 2] - bx[:, 0], np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    L_, S_ = float(np.median(rr[:, 0])), float(np.median(rr[:, 1]))
    return dict(n_oke=len(oke), fb=fb, rrL=L_, rrS=S_, asp=L_ / max(S_, 1e-6), asp_pf=float(np.median(rr[:, 0] / np.maximum(rr[:, 1], 1e-6))),
                boxasp=float(np.median(bw / bh)))


def norm_ph(s):
    s = re.sub(r"[^a-z ]+", " ", str(s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:the|a|an) ", "", s)
    return s


def ph_forms(s):
    s = norm_ph(s)
    if not s:
        return set()
    out = {s, s[:-1] if s.endswith("s") and len(s) > 3 else s}
    return out | {x.replace(" ", "") for x in out}


def moving_target(j):
    tf = ph_forms((j.get("target") or {}).get("phrase"))
    pri = [p for p in (j.get("priors") or []) if p.get("kind") in ("speed", "accel") and ph_forms(p.get("obj")) & tf]
    dk = set(j.get("target_dkeys") or [])
    ts = {round(float(d["t"]), 4) for d in (j.get("depth") or []) if d.get("key") in dk and d.get("t") is not None}
    why = []
    if pri:
        why.append("prior " + pri[0]["kind"])
    if len(ts) >= 2:
        why.append(f"depth@{len(ts)}t")
    return "+".join(why)


def cut_frac(st):
    if st is None:
        return np.nan
    v = [s for s in st if s is not None]
    if not v:
        return np.nan
    return float(np.mean([int(s.get("border", 0)) != 0 for s in v]))


rows = []
for split, suf in (("test", "detail"),):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    RW = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0,
                     usecols=lambda c: c in ("Unnamed: 0", "question", "meas_tracks", "meas_mode", "meas_why", "meas_mpp"))
    PP = parse(split)
    for i in D.index:
        j = PP.get(int(i))
        if not j:
            continue
        q = j.get("q") or {}
        if q.get("kind") != "size" or q.get("dim") != "width":
            continue
        ques = str(j.get("question", "")).strip()
        if SIDE_RX.search(ques.lower()):
            continue
        ph = re.sub(r"[^a-z ]+$", "", str((j.get("target") or {}).get("phrase") or "").strip().lower()).strip()
        if not FLAT_RX.search(ph):
            continue
        r = D.loc[i]
        src = str(r.answer_source)
        mt = str(RW.meas_tracks.get(i, "")) if i in RW.index else ""
        vj, path = track_json(split, str(j["video_id"]), mt) if mt in ("v1", "v2") else (None, None)
        st, key = frames_of(vj, j.get("target"))
        gm_ = geom(st) if st is not None else None
        pst, pkey = frames_of(vj, j.get("prior_obj")) if (j.get("prior_obj") or {}).get("phrase") else (None, None)
        g = float(r.g) if pd.notna(r.g) else np.nan
        cs = float(r.cs) if pd.notna(r.cs) else np.nan
        row = dict(split=split, id=int(i), vid=str(j["video_id"]), cat=r["cat"], ph=ph, q=ques[:60], qfull=ques, src=src, ans=float(r.answer), g=g, cs=cs,
                   mode=str(RW.meas_mode.get(i, "")), mt=mt, tdir=(path or "").split("/")[-2] if path else "", key=key,
                   winged=bool(re.search(WINGED, ph)), pcut=cut_frac(pst), pkey=pkey,
                   moving=moving_target(j))
        if gm_:
            row.update(gm_)
        rows.append(row)
X = pd.DataFrame(rows).set_index(["split", "id"])
for c in ("rrL", "rrS", "asp", "asp_pf", "boxasp", "n_oke"):
    if c not in X:
        X[c] = np.nan
X["gp"] = X.g * X.rrL / X.rrS
with np.errstate(divide="ignore", invalid="ignore"):
    X["d0"] = np.abs(np.log(X.g / X.cs)); X["d1"] = np.abs(np.log(X.gp / X.cs))


def why_out(r):
    if r.moving: return f"moving target {r.moving}"
    if r.winged: return "winged"
    if pd.isna(r.rrL): return "no track"
    if not (r.asp >= ASPECT_RR): return f"aspect {r.asp:.2f}"
    if not (r.boxasp >= ASPECT_BOX): return f"box w/h {r.boxasp:.2f}"
    if not (r.rrL >= MIN_LONG): return f"rr_long {r.rrL:.1f}px"
    if r.pcut > MAX_PRIOR_CUT: return f"prior cut {r.pcut:.2f}"
    if not (np.isfinite(r.g) and np.isfinite(r.cs) and r.g > 0 and r.cs > 0): return "no g/cs"
    if r.src not in SRC_OK: return f"src {r.src}"
    if not (r.d1 < r.d0): return 't160'
    return ""


X["out"] = X.apply(why_out, axis=1)
X["fire"] = X.out == ""
ratio = np.maximum(X.gp, X.cs) / np.minimum(X.gp, X.cs)
X["new"] = np.where(ratio <= 2, np.exp(W_GEO * np.log(X.gp) + (1 - W_GEO) * np.log(X.cs)), X.gp)
X["how"] = np.where(ratio <= 2, "wgm", "geo'")
X.loc[~X.fire, "new"] = np.nan

X["pooled"] = X.new

print('')
F = X[X.fire]
out = {}
if "test" in F.index.get_level_values(0):
    for i, r in F.loc["test"].iterrows():
        out[str(int(i))] = dict(value=float(r.pooled),
                                why=f"{TAG} {r.ph}: rr {r.rrL:.0f}/{r.rrS:.0f}px asp {r.asp:.2f} box {r.boxasp:.2f}; g {r.g:.4g}->g' {r.gp:.4g}, "
                                    f"cs {r.cs:.4g} -> {r.how} {r.new:.4g}" + (f" pooled {r.pooled:.4g}" if abs(r.pooled - r.new) > 1e-9 else ""))
json.dump(out, open(f"{WO}/flatw2own_test.json", "w"), indent=1)
print('')
