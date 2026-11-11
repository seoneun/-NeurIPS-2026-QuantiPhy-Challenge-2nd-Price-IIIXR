import sys, os, json, re, math
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
TAG = "rul056"
W = {"dis_geo": 1.0, "geo_cawinvalid": 1.0, "agree_wgm": 0.7, "agree_gm": 0.5, "dis_size_gm": 0.5, "rul263_dimgm": 0.5,
     "rul263_unstable_gm": 0.5, "d3lab_gm": 0.5}
ASP, TH_LO, TH_HI_2D, TH_HI_3D, R_MAX = 2.0, 4.0, 30.0, 15.0, 0.97
INCL = re.compile(r"includ|with its|with the")
PER = re.compile(r"\b(?:person|people|man|men|woman|women|human|pedestrian|player|boy|girl|child|kid|lady|guy|walker|runner|jogger|"
                 r"athlete|skater|dancer|worker|swimmer|diver|baby|astronaut|robot|doll|statue|mannequin)s?\b")
WINGED = re.compile(r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft)\b")
SCENE = re.compile(r"\b(lanes?|road|street|court|tiles?|floor|ceiling|room|wall|field|track|pool|river|sidewalk|crosswalk|ground|sky|water|ruler|calibre|window frame|markings?|gap|opening|surface|deck|lattice|frame of)\b")
FLAT = re.compile(r"\b(?:paper|sheet|letter|envelope|tray|mat|placemat|rug|carpet|towel|napkin|cloth|tablecloth|book|notebook|note book|"
                  r"magazine|newspaper|card|cutting board|chopping board|blanket|doormat|phone|tablet|laptop)s?\b")
ASP_MAX = 8.0
LEG = re.compile(r"\b(?:cat|dog|puppy|kitten|horse|pony|cow|bull|ox|elephant|raccoon|deer|giraffe|lion|tiger|bear|panda|fox|wolf|"
                 r"rabbit|bunny|sheep|goat|pig|camel|zebra|kangaroo|monkey|gorilla|squirrel|mouse|rat|duck|chicken|hen|goose|ostrich|"
                 r"penguin|frog|lizard|dinosaur|animal|pet|ant|spider|crab|beetle|insect)s?\b")


def parse(split):
    d = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); d[int(j["id"])] = j
    return d


_TJ = {}


def track_json(split, vid, mt):
    k = (split, vid, mt)
    if k not in _TJ:
        cands = [f"{R}/geo/out/tracks_{split}/{vid}.json"] if mt == "v1" else [f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json"]
        p = next((c for c in cands if os.path.exists(c)), None)
        _TJ[k] = json.load(open(p)) if p else None
    return _TJ[k]


def ent_keys(ent):
    if not ent or not ent.get("phrase"):
        return []
    rel = ent.get("rel")
    base = ent["phrase"] + "|" + ",".join(ent.get("where") or [])
    return ([base + "|" + rel[0] + ":" + rel[1]] if rel else []) + [base]


def frames_of(vj, ent):
    if vj is None:
        return None
    qs = {q["key"]: q for q in vj["queries"]}
    for k in ent_keys(ent):
        if k in qs:
            q = qs[k]
            if vj.get("version") == "v2":
                return None if q.get("obj") is None else [f.get("st") if f else None for f in vj["objs"][str(q["obj"])]]
            return [f.get("st") if f else None for f in q["frames"]]
    return None


def geom(st):
    val = np.array([s is not None for s in st])
    area = np.array([s["area"] if s else 0 for s in st], float)
    med = np.median(area[val]) if val.any() else 0
    ok = val & (area > 0.4 * med) & (area < 2.5 * med) if med > 0 else val
    border = np.array([s["border"] if s else 9 for s in st])
    okc = np.where(ok)[0]; oke = np.where(ok & (border == 0))[0]
    if len(oke) < max(2, 0.3 * len(okc)):
        oke = okc
    if len(oke) < 3:
        return None
    rr = np.array([st[j]["rr"] for j in oke], float)
    bx = np.array([st[j]["box"] for j in oke], float)
    L, S, a = rr[:, 0], rr[:, 1], np.deg2rad(rr[:, 2])
    bw, bh = bx[:, 2] - bx[:, 0], np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    th = []
    for k in range(len(L)):
        best = None
        for t_ in (a[k], a[k] + np.pi / 2):
            e = abs(L[k] * abs(math.cos(t_)) + S[k] * abs(math.sin(t_)) - bw[k]) + abs(L[k] * abs(math.sin(t_)) + S[k] * abs(math.cos(t_)) - bh[k])
            if best is None or e < best[0]:
                best = (e, t_ % np.pi)
        th.append(best[1])
    th = np.array(th)
    thm = np.degrees((0.5 * math.atan2(np.median(np.sin(2 * th)), np.median(np.cos(2 * th)))) % np.pi)
    return dict(n=len(oke), rrL=float(np.median(L)), rrS=float(np.median(S)), bh=float(np.median(bh)),
                asp=float(np.median(L) / max(np.median(S), 1e-6)), theta=float(min(thm, 180 - thm)))


rows = []
for split, suf in (("test", "detail"),):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    RW = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0, usecols=lambda c: c in ("Unnamed: 0", "meas_tracks", "meas_mode", "meas_mpp"))
    PP = parse(split)
    for i in D.index:
        j = PP.get(int(i))
        if not j or (j.get("q") or {}).get("kind") != "size" or j["q"].get("dim") != "height":
            continue
        r = D.loc[i]
        ques = str(j.get("question", "")).lower()
        tph = str((j.get("target") or {}).get("phrase") or j["q"].get("target") or "").lower()
        why = ""
        src = str(r.answer_source)
        w = W.get(src)
        if INCL.search(ques): why = "including"
        elif PER.search(tph) or LEG.search(tph): why = "person/legged"
        elif WINGED.search(tph) or SCENE.search(tph) or FLAT.search(tph): why = "winged/scene/flat noun"
        elif w is None: why = f"src {src}"
        g, cs, ans = float(r.g) if pd.notna(r.g) else np.nan, float(r.cs) if pd.notna(r.cs) else np.nan, float(r.answer)
        mode = str(RW.meas_mode.get(i, "")); mt = str(RW.meas_tracks.get(i, "")); mpp = RW.meas_mpp.get(i, np.nan)
        G = None
        if not why:
            vj = track_json(split, str(j["video_id"]), mt) if mt in ("v1", "v2") else None
            st = frames_of(vj, j.get("target"))
            G = geom(st) if st is not None else None
            three = str(r["cat"])[1] == "3"
            if G is None: why = "no track"
            elif not (np.isfinite(g) and g > 0): why = "no geo"
            elif not (ASP <= G["asp"] <= ASP_MAX): why = f"aspect {G['asp']:.2f}"
            elif not (TH_LO <= G["theta"] <= (TH_HI_3D if three else TH_HI_2D)): why = f"theta {G['theta']:.1f}"
            elif mode.startswith("2d_ratio") and not (pd.notna(mpp) and abs(np.log(g / (mpp * G["bh"]))) < 0.03): why = 't155'
            elif G["rrS"] / G["bh"] >= R_MAX: why = f"r {G['rrS'] / G['bh']:.3f}"
        row = dict(split=split, id=int(i), vid=str(j["video_id"]), cat=r["cat"], tph=tph, q=j["question"], src=src, w=w, mode=mode,
                   g=g, cs=cs, ans=ans, why=why)
        if G:
            row.update(G); row["r"] = G["rrS"] / G["bh"]
        rows.append(row)
X = pd.DataFrame(rows)
X["fire"] = X.why == ""
X["new"] = np.where(X.fire, X.ans * X.r.fillna(1.0) ** X.w.fillna(0.0), np.nan)
X["gp"] = X.g * X.r
F = X[X.fire]
for sp in ("test",):
    out = {str(int(r.id)): dict(value=float(r.new), why=f"{TAG} {r.tph}: rr {r.rrL:.0f}/{r.rrS:.0f}px asp {r.asp:.2f} tilt {r.theta:.1f}deg, "
                                                         f"bh {r.bh:.0f}px; r {r.r:.3f}; g {r.g:.4g}->{r.gp:.4g}, cs {r.cs:.4g}, {r.src} w{r.w}; was {r.ans:.4g}")
           for _, r in F[F.split == sp].iterrows()}
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
    print(f"[{sp}] {TAG}: {len(out)} rows")
