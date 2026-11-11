#!/usr/bin/env python
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ["RA_Q"]); WO = os.path.abspath(os.environ["RA_OUT"])
BASE = sys.argv[1]; GT = sys.argv[2]
TAG = "rul023"
LAYERS = {"v1": [""], "v2": ["v2scSIL", "v2sc", "v2"]}
WINGED = r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft)\b"
FLAT = (r"paper|sheet|letter|envelope|tray|mat|placemat|rug|carpet|towel|napkin|cloth|tablecloth|book|notebook|note book|"
        r"magazine|newspaper|card|cutting board|chopping board|blanket|doormat|"
        r"phone|iphone|smartphone|cellphone|cell phone|mobile phone|tablet")
PHONE_RX = re.compile(r"\b(?:phone|iphone|smartphone|cellphone|cell phone|mobile phone|tablet)s?$")
REL_RX = re.compile(r"\s+(?:(?:which|that) (?:is|are) |(?:is|are) )?(?:placed|located|lying|laid|sitting|standing|resting|positioned|"
                    r"next to|to the (?:left|right) of|on (?:the|top of|a)|near|beside|behind|in front of|under|above|below|"
                    r"between|by the|at the|in the)\b.*$")
FLAT_RX = re.compile(r"\b(?:" + FLAT + r")s?$")
SIDE_RX = re.compile(r"longer side|shorter side|thickness")
SRC_OK = {"dis_geo", "dis_size_gm", "agree_gm", "agree_wgm", "rul263_dimgm", "d3lab_gm", "rul263_unstable_gm"}
ASPECT_RR, ASPECT_BOX, MIN_LONG, MAX_PRIOR_CUT = 2.5, 2.3, 15.0, 0.5
W_GEO = 0.7
_TJ = {}


def track_json(split, vid, mt):
    k = (split, vid, mt)
    if k not in _TJ:
        cands = [f"{R}/geo/out/tracks_{split}{'_' + l if l else ''}/{vid}.json" for l in LAYERS.get(mt, [])]
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
            if vj.get("version") == "v2" or "objs" in vj:
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
    okc = np.where(ok)[0]; oke = np.where(ok & (border == 0))[0]
    if len(oke) < max(2, 0.3 * len(okc)):
        oke = okc
    return oke


def geom(st):
    oke = ext_frames(st)
    if len(oke) == 0:
        return None
    rr = np.array([st[j]["rr"][:2] for j in oke], float)
    bx = np.array([st[j]["box"] for j in oke], float)
    bw, bh = bx[:, 2] - bx[:, 0], np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    L_, S_ = float(np.median(rr[:, 0])), float(np.median(rr[:, 1]))
    return dict(rrL=L_, rrS=S_, asp=L_ / max(S_, 1e-6), boxasp=float(np.median(bw / bh)))


def norm_ph(s):
    s = re.sub(r"[^a-z ]+", " ", str(s or "").lower()); s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(?:the|a|an) ", "", s)


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
    return ("prior " + pri[0]["kind"] if pri else "") + (f" depth@{len(ts)}t" if len(ts) >= 2 else "")


def cut_frac(st):
    v = [s for s in (st or []) if s is not None]
    return float(np.mean([int(s.get("border", 0)) != 0 for s in v])) if v else np.nan


for split in ("test",):
    D = pd.read_csv(f"{R}/submissions/{BASE}_detail.csv", index_col=0)
    RW = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv", index_col=0)
    PP = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); PP[int(j["id"])] = j
    out = {}
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
        ph = REL_RX.sub("", ph).strip()
        if not FLAT_RX.search(ph):
            continue
        src = str(D.answer_source[i])
        mt = str(RW.meas_tracks.get(i, ""))
        vj, path = track_json(split, str(j["video_id"]), mt) if mt in ("v1", "v2") else (None, None)
        st, key = frames_of(vj, j.get("target"))
        gm_ = geom(st) if st is not None else None
        pst, _ = frames_of(vj, j.get("prior_obj")) if (j.get("prior_obj") or {}).get("phrase") else (None, None)
        mwhy = str(RW.meas_why.get(i, "")); mpp = float(pd.to_numeric(RW.meas_mpp.get(i), errors="coerce"))
        mval = float(pd.to_numeric(RW.meas_value.get(i), errors="coerce"))
        cn2 = bool(src == "caw_nogeo" and mwhy == "2-decade sanity rule" and np.isfinite(mpp) and mpp > 0)
        g = float(D.g[i]) if pd.notna(D.g[i]) else (mval if cn2 else np.nan)
        cs = float(D.cs[i]) if pd.notna(D.cs[i]) else np.nan
        mov = moving_target(j); pcut = cut_frac(pst)
        why = ""
        if mov:
            why = f"moving target {mov}"
        elif re.search(WINGED, ph):
            why = "winged"
        elif gm_ is None:
            why = "no track"
        elif not gm_["asp"] >= ASPECT_RR:
            why = f"aspect {gm_['asp']:.2f}"
        elif not gm_["boxasp"] >= ASPECT_BOX:
            why = f"box w/h {gm_['boxasp']:.2f}"
        elif not gm_["rrL"] >= MIN_LONG:
            why = f"rr_long {gm_['rrL']:.1f}px"
        elif pcut > MAX_PRIOR_CUT:
            why = f"prior cut {pcut:.2f}"
        elif not (np.isfinite(g) and np.isfinite(cs) and g > 0 and cs > 0):
            why = "no g/cs"
        elif src not in SRC_OK and not cn2:
            why = f"src {src}" + (f" ({mwhy})" if src == "caw_nogeo" else "")
        if why:
            print(split, i, j["video_id"], ph, "|", why, "|", os.path.relpath(path, R) if path else None); continue
        gp = gm_["rrL"] * mpp if cn2 else g * gm_["rrL"] / gm_["rrS"]
        if not abs(np.log(gp / cs)) < abs(np.log(g / cs)):
            print(split, i, j["video_id"], ph, f"| g' {gp:.3g} moves away from CaW {cs:.3g}"); continue
        ratio = max(gp, cs) / min(gp, cs)
        new = float(np.exp(W_GEO * np.log(gp) + (1 - W_GEO) * np.log(cs))) if ratio <= 2 else float(gp)
        how = "wgm" if ratio <= 2 else "geo'"
        out[str(int(i))] = dict(value=new, why='t165')
        print(split, i, j["video_id"], D.cat[i], ph, "|", src, round(float(D.answer[i]), 4), "->", round(new, 4), "|", os.path.relpath(path, R),
              f"rrS*mpp {gm_['rrS'] * mpp if np.isfinite(mpp) else np.nan:.4g} vs own meas_value {mval:.4g}")
    json.dump(out, open(f"{WO}/ag_{TAG}_patch_{split}.json", "w"), indent=0)
    print(split, "rows", len(out))
