"""analysis/v3/ag/rul291/rul291.py - rul291: flatw + the regress verifier's sibling-length filter.

(copied from analysis/v3/ag/flatw/flatw.py; flatw: the width of a flat object lying with its long axis horizontal, cycle-17 audit)

geo measures "width" as the SHORT side of the target mask's rotated rectangle (measure.Track.ext_px: rr[1] unless winged).
For a flat object (paper, tray, book, mat, ...) seen with a clearly elongated, horizontal footprint, the question author's
"width" is the horizontal (long) extent, so the short-side reading under-states it by the rr aspect.

INPUTS : parse q (kind, dim, target phrase, question); geo raw (meas_tracks, meas_mode); the track file chosen by meas_tracks
         (v1 tracks_{split}; v2 -> tracks_{split}_v2sc, else tracks_{split}_v2, exactly as measure_v3 --tracks v2sc resolves);
         target obj per-frame st.rr = [long, short, angle], st.box, st.border; detail g, cs, answer_source.
GATES  : 1 kind size, dim width; question lacks longer side / shorter side / thickness.
         2 target phrase ENDS with a flat noun (optionally plural); not winged.
         3 over the extent frames (measure.Track.oke: area-consistent, border == 0, fallback to all usable frames):
           median(rr_long) / median(rr_short) >= 2.5 and median(box w / box h) >= 2.3 (long axis horizontal).
         4 median rr_long >= 15 px; the prior (ruler) track is border-cut in <= 50 % of its tracked frames (no prior track: pass).
         5 g > 0, cs > 0 finite; answer_source in SRC_OK (not patch:*, caw_nogeo, d3size_caw).
         6 (rul291) NO length sibling: the same video_id has no size question with dim "length" on the same normalised target
           phrase, and no question whose normalised text equals this question with "width" -> "length". When the author asks
           both, "width" is the shorter cover side (length = long side), not the horizontal extent, so geo's short side is right.
         7 (rul291, optional) target is not the moving object: the normalised target phrase is not the obj of a speed/accel
           prior, and the target_dkeys do not have depth entries at >= 2 distinct times.
COMPUTE: g' = g * median(rr_long) / median(rr_short); accept only if |ln(g'/cs)| < |ln(g/cs)|.
         max(g',cs)/min(g',cs) <= 2 -> exp(0.7 ln g' + 0.3 ln cs) (agree_wgm), else g'.
         lab copies (captured_*, same event, same question): log-mean of the per-copy values (as --lab_pool event).
Labels only EVALUATE (val + all-237); they never set an answer.
usage: python rul291.py [SUB] [--nowrite] [--no_moving]"""
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
import analyze_caw as A
from lab_rows import labelled
from variant_keys import split_event, split_variant

pd.set_option("display.width", 280); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 500)
args = [x for x in sys.argv[1:] if not x.startswith("--")]
SUB = args[0] if args else "sub_xt_Nzk"
WRITE = "--nowrite" not in sys.argv
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
TAG = "rul291"
MOVING_GATE = "--no_moving" not in sys.argv

WINGED = r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft)\b"   # geo/measure.py
FLAT = (r"paper|sheet|letter|envelope|tray|mat|placemat|rug|carpet|towel|napkin|cloth|tablecloth|book|notebook|note book|"
        r"magazine|newspaper|card|cutting board|chopping board|blanket|doormat")
FLAT_RX = re.compile(r"\b(?:" + FLAT + r")s?$")
SIDE_RX = re.compile(r"longer side|shorter side|thickness")
SRC_OK = {"dis_geo", "dis_size_gm", "agree_gm", "agree_wgm", "d3size_dimgm", "d3lab_gm", "d3size_unstable_gm"}
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
    """per-frame list of st (or None) for the entity's query, as measure_v3.Video resolves it (cfg e: queries)."""
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
    """measure.Track: ok = tracked & area within 0.4-2.5x median; oke = ok & border == 0 (fallback ok when too few)."""
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
    """normalised phrase + its singular (trailing s dropped) + space-free form (note book == notebook)."""
    s = norm_ph(s)
    if not s:
        return set()
    out = {s, s[:-1] if s.endswith("s") and len(s) > 3 else s}
    return out | {x.replace(" ", "") for x in out}


def norm_q(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())).strip()


def sibling_index(PP):
    """per video_id: target-phrase forms and normalised question texts of its size/length questions."""
    idx = {}
    for i, j in PP.items():
        q = j.get("q") or {}
        if q.get("kind") == "size" and q.get("dim") == "length":
            e = idx.setdefault(str(j["video_id"]), {"ph": set(), "q": set(), "ids": []})
            e["ph"] |= ph_forms((j.get("target") or {}).get("phrase") or q.get("target"))
            e["q"].add(norm_q(j.get("question")))
            e["ids"].append(int(i))
    return idx


def length_sibling(j, idx):
    e = idx.get(str(j["video_id"]))
    if not e:
        return ""
    hit_ph = ph_forms((j.get("target") or {}).get("phrase") or (j.get("q") or {}).get("target")) & e["ph"]
    hit_q = re.sub(r"\bwidth\b", "length", norm_q(j.get("question"))) in e["q"]
    if hit_ph or hit_q:
        return ("ph" if hit_ph else "") + ("+q" if hit_q else "") + ":" + ",".join(map(str, sorted(e["ids"])))
    return ""


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
for split, suf in (("test", "detail"), ("val", "valdetail")):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    RW = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0,
                     usecols=lambda c: c in ("Unnamed: 0", "question", "meas_tracks", "meas_mode", "meas_why", "meas_mpp"))
    PP = parse(split)
    SIB = sibling_index(PP)
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
                   lsib=length_sibling(j, SIB), moving=moving_target(j))
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
    if r.lsib: return f"length sibling {r.lsib}"
    if MOVING_GATE and r.moving: return f"moving target {r.moving}"
    if r.winged: return "winged"
    if pd.isna(r.rrL): return "no track"
    if not (r.asp >= ASPECT_RR): return f"aspect {r.asp:.2f}"
    if not (r.boxasp >= ASPECT_BOX): return f"box w/h {r.boxasp:.2f}"
    if not (r.rrL >= MIN_LONG): return f"rr_long {r.rrL:.1f}px"
    if r.pcut > MAX_PRIOR_CUT: return f"prior cut {r.pcut:.2f}"
    if not (np.isfinite(r.g) and np.isfinite(r.cs) and r.g > 0 and r.cs > 0): return "no g/cs"
    if r.src not in SRC_OK: return f"src {r.src}"
    if not (r.d1 < r.d0): return f"g' {r.gp:.3g} moves away from CaW {r.cs:.3g}"
    return ""


X["out"] = X.apply(why_out, axis=1)
X["fire"] = X.out == ""
ratio = np.maximum(X.gp, X.cs) / np.minimum(X.gp, X.cs)
X["new"] = np.where(ratio <= 2, np.exp(W_GEO * np.log(X.gp) + (1 - W_GEO) * np.log(X.cs)), X.gp)
X["how"] = np.where(ratio <= 2, "wgm", "geo'")
X.loc[~X.fire, "new"] = np.nan

# lab copies: log-mean over the event's copies asking the same question (lab_pool event); only copies the rule fires on
X["ev"] = [split_event(v)[0] for v in X.vid]
X["qk"] = X.qfull.str.strip().str.lower()
X["pooled"] = X.new
labm = X.vid.str.startswith("captured_") & X.fire
for (sp, ev, qk), G in X[labm].reset_index().groupby(["split", "ev", "qk"]):
    if len(G) >= 2:
        pv = float(np.exp(np.mean(np.log(G.new))))
        for i in G.id:
            X.loc[(sp, i), "pooled"] = pv
# mixed groups (a copy of the same lab question that does not fire) are reported
mixed = []
for (sp, ev, qk), G in X[X.vid.str.startswith("captured_")].reset_index().groupby(["split", "ev", "qk"]):
    if G.fire.any() and not G.fire.all():
        mixed.append((sp, ev, qk, G[["id", "vid", "out"]].values.tolist()))

print(f"=== {SUB}: width rows with a flat head noun: {len(X)}; fire {int(X.fire.sum())}")
cols = ["vid", "cat", "ph", "lsib", "moving", "src", "mode", "mt", "tdir", "ans", "g", "cs", "rrL", "rrS", "asp", "asp_pf", "boxasp", "n_oke", "pcut", "gp", "new", "pooled", "how", "out"]
print(X[cols].round(3).to_string())
print("mixed lab groups:", mixed)

# ---------- label evaluation (labels only evaluate) ----------
L = labelled(SUB)
Ly = {(sp, int(i)): float(y) for i, sp, y in zip(L.index, L.split, L.y)}
X["y"] = [Ly.get(k, np.nan) for k in X.index]
lab = X[X.y.notna()].copy()
if len(lab):
    lab["m0"] = A.row_mra(lab.ans.values, lab.y.values)
    lab["mg"] = A.row_mra(lab.g.values, lab.y.values)
    lab["mgp"] = A.row_mra(lab.gp.values, lab.y.values)
    lab["m1"] = np.where(lab.fire, A.row_mra(lab.pooled.fillna(1).values, lab.y.values), lab.m0)
    print("\nlabelled candidate rows (fired and gated):")
    print(lab[["vid", "ph", "src", "ans", "g", "gp", "cs", "y", "pooled", "m0", "mg", "mgp", "m1", "out"]].round(3).to_string())
    f = lab[lab.fire]
    print(f"\nlabelled rows changed: {len(f)}; mean MRA {f.m0.mean():.3f} -> {f.m1.mean():.3f}")
    print("by split:", f.groupby(level=0).apply(lambda d: f"{len(d)} rows {d.m0.mean():.3f}->{d.m1.mean():.3f}").to_dict())

F = X[X.fire]
print("\ntest rows changed by category:", F.loc["test"].cat.value_counts().to_dict() if "test" in F.index.get_level_values(0) else {})
print("val rows changed by category:", F.loc["val"].cat.value_counts().to_dict() if "val" in F.index.get_level_values(0) else {})
if WRITE:
    for sp in ("test", "val"):
        out = {}
        if sp in F.index.get_level_values(0):
            for i, r in F.loc[sp].iterrows():
                out[str(int(i))] = dict(value=float(r.pooled),
                                        why=f"{TAG} {r.ph}: rr {r.rrL:.0f}/{r.rrS:.0f}px asp {r.asp:.2f} box {r.boxasp:.2f}; g {r.g:.4g}->g' {r.gp:.4g}, "
                                            f"cs {r.cs:.4g} -> {r.how} {r.new:.4g}" + (f" pooled {r.pooled:.4g}" if abs(r.pooled - r.new) > 1e-9 else ""))
        fn = f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json"
        json.dump(out, open(fn, "w"), indent=1)
        print(f"wrote {fn}: {len(out)} rows")

# ---------- diff vs the flatw (v1) patch, for the record ----------
for sp in ("test", "val"):
    fn0 = f"{R}/cache/tmp/ag_flatw_patch_{sp}.json"
    if not os.path.exists(fn0):
        continue
    p0 = set(json.load(open(fn0)))
    p1 = set(str(int(i)) for i in F.loc[sp].index) if sp in F.index.get_level_values(0) else set()
    drop = sorted(p0 - p1, key=int); add = sorted(p1 - p0, key=int)
    print(f"{sp}: flatw {len(p0)} rows -> rul291 {len(p1)} rows; dropped {drop} added {add}")
    for i in drop:
        k = (sp, int(i))
        if k in X.index:
            print(f"   {i} {X.loc[k, 'vid']}: {X.loc[k, 'out']}")
gm = X[(X.lsib != "") | (X.moving != "")]
print("\nrows touched by gate 6/7 (any candidate):")
print(gm[["vid", "ph", "q", "lsib", "moving", "out"]].to_string())
