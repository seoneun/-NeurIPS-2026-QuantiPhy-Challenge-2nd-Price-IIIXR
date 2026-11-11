"""rul426: per-object depth twins. A simulation video V takes the time-less speed / acceleration prior that another video U states for
object K when V's depth_info lines for K (the 't=..s, distance_<K>_camera = d' lines, d to 4 decimals) are identical to U's (>= 2 lines),
even though the rest of the two depth tables differ (simtwin / rul420 need the whole table identical). Same renders reuse the same
scripted object path; the identical K depth lines say the same K trajectory sits in both clips.
  donor U   : exactly one prior, a time-less dynamic prior 'speed|acceleration of (the) K = x' (no t/t1/t2, no time word), K in U's lines;
  receiver V: base(V) != base(U) (strip _segmented); V's K-lines (>= 2) identical to U's; V's own prior (static or dynamic) is not about K;
  gate      : (a) rul420.motion_gate(V target, U prior object) passes unchanged, or
              (b) position identity: >= 24 common tracked frames (time = frame / CSV fps), median |c_V - c_U| <= 0.02 W, and the
                  early-motion fractions |c(0.5 s) - c(f0)| / |c(1 s) - c(f0)| (f0 = first tracked frame within 0.1 s) differ by <= 0.1
                  when both are finite;
  rows      : V's rows whose target is K (target_dkeys == [K] or target head noun == K) asking speed@t / avgspeed / accel@t;
              skip answer_source kin / scene_prior / patch:simtwin* and rows already in a simtwin / rul420 patch;
  output    : speed v -> speed@t, avgspeed = v (accel rows skipped); accel a -> speed@t = a t, avgspeed[t1,t2] = a (t1+t2)/2
              (default [0, clip end]), accel@t = a;
  guards    : all donors for (K, lines) (plain / _segmented / test / val copies) agree within 2% and have one kind; physical:
              |dd|/dt between consecutive K lines <= 1.05 x kinematic mean speed; no labelled row may change; val simulation_0300
              is a donor only.
--px adds the flagged variant subtwin_px: (V states a static prior on K and passes (a) or (b), or gate (a) fails only on coverage
and (b) fails) and the median pixel-speed ratio (centre, +-0.25 s,
all boxes incl. cut) in [0.9, 1.1] over >= 20 common times, and the receiver's current answer > 2x the donor value.
Writes cache/tmp/ag_subtwin{_px}_patch_{test,val}.json (final answers in the question's unit)."""
import json, re, sys, collections, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3")
_src = open(f"{R}/analysis/v3/ag/rul420/rul420.py").read()
ST = {"__file__": f"{R}/analysis/v3/ag/rul420/rul420.py"}   # the head's root line falls back to __file__ without QP_ROOT
exec(compile(_src.split("_meta = {}")[0], "simtwin2_head", "exec"), ST)   # motion_gate, track, ent_key, CSV_FPS, P, DET (read-only)
motion_gate, track, ent_key, CSV_FPS = ST["motion_gate"], ST["track"], ST["ent_key"], ST["CSV_FPS"]
PX = "--px" in sys.argv
SPLITS = ("test", "val")
P = {sp: [json.loads(l) for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl")] for sp in SPLITS}
DET = {"test": pd.read_csv(f"{R}/submissions/sub_xt_Nzk_detail.csv").set_index("Unnamed: 0"),
       "val": pd.read_csv(f"{R}/submissions/sub_xt_Nzk_valdetail.csv").set_index("Unnamed: 0")}
RAW = {sp: pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", usecols=["Unnamed: 0", "video_id", "depth_info", "fps"]).set_index("Unnamed: 0")
       for sp in SPLITS}
TIMEWORD = re.compile(r"\b(before|after|until|from|between|at|during|when|first|last)\b")
U_SPEED = {"m/s": 1.0, "cm/s": 100.0, "km/h": 3.6, "mm/s": 1000.0}
U_ACC = {"m/s^2": 1.0, "m/s2": 1.0, "cm/s^2": 100.0, "m/s²": 1.0}
LINE = re.compile(r"t\s*=\s*([0-9.]+)\s*s\s*,\s*distance_(.+?)_camera\s*=\s*([0-9.]+)", re.I)
PRIOR = re.compile(r"^\s*(?:the\s+)?(speed|acceleration)\s+of\s+(?:the\s+)?(.+?)\s*(?:=|\bis\b)\s*(-?[0-9.]+)", re.I)
# simtwin / rul420 rows already handled (later patches): never touch them
_done = set()
for tg in ("rul420",):              # simtwin (v1) is superseded and not in the patch list
    for sp in SPLITS:
        f = f"{R}/cache/tmp/ag_{tg}_patch_{sp}.json"
        if os.path.exists(f):
            _done |= {(sp, int(i)) for i in json.load(open(f))}

def norm(k):
    return re.sub(r"\s+", "_", str(k).strip().lower())

def base(v):
    return str(v).replace("_segmented", "")

def klines(txt):
    out = collections.defaultdict(set)
    for t, k, d in LINE.findall(str(txt)):
        out[norm(k)].add((round(float(t), 4), round(float(d), 4)))
    return {k: frozenset(v) for k, v in out.items()}

# per video: rows, K-lines (must be the same on every row of the video)
vids = collections.defaultdict(list)
for sp in SPLITS:
    for p in P[sp]:
        if str(p["video_id"]).startswith("simulation_"):   # simulation clips and their _segmented copies (source 'segmentation')
            vids[(sp, p["video_id"])].append(p)
KL = {}
for (sp, v), rows in vids.items():
    ks = {tuple(sorted((k, tuple(sorted(s))) for k, s in klines(RAW[sp].depth_info.get(r["id"], "")).items())) for r in rows}
    if len(ks) == 1:
        KL[(sp, v)] = klines(RAW[sp].depth_info.get(rows[0]["id"], ""))
    else:
        print("inconsistent depth_info across rows of", sp, v)

def dyn_prior_keys(rows):
    """normalized objects of the video's dynamic priors (text K and parse prior_dkeys)"""
    ks = set()
    for pr in rows[0]["priors"]:
        if pr["kind"] in ("speed", "accel"):
            m = PRIOR.match(str(pr.get("raw") or rows[0]["prior_text"]))
            if m:
                ks.add(norm(m.group(2)))
            if pr.get("obj"):
                ks.add(norm(pr["obj"]))
            for r in rows:
                ks |= {norm(x) for x in (r.get("prior_dkeys") or [])}
    return ks

def any_prior_keys(rows):
    """normalized objects of all the video's priors (static or dynamic)"""
    ks = set()
    for pr in rows[0]["priors"]:
        m = re.match(r"^\s*(?:the\s+)?\w+\s+of\s+(?:the\s+)?(.+?)\s*(?:=|\bis\b)", str(pr.get("raw") or ""), re.I)
        if m:
            ks.add(norm(m.group(1)))
        if pr.get("obj"):
            ks.add(norm(pr["obj"]))
    for r in rows:
        ks |= {norm(x) for x in (r.get("prior_dkeys") or [])}
    return ks

donors = collections.defaultdict(list)       # (K, lines) -> [(sp, U, kind, value_si, prior text, prior entity)]
for (sp, v), rows in vids.items():
    if (sp, v) not in KL:
        continue
    pri = rows[0]["priors"]
    if len(pri) != 1 or pri[0]["kind"] not in ("speed", "accel"):
        continue
    pr = pri[0]
    txt = str(rows[0]["prior_text"])
    if any(pr.get(x) is not None for x in ("t", "t1", "t2")) or TIMEWORD.search(txt.lower()):
        continue
    m = PRIOR.match(txt)
    if not m:
        print("  donor text not matched", sp, v, repr(txt)); continue
    K = norm(m.group(2))
    kind = "accel" if m.group(1).lower().startswith("acc") else "speed"
    if kind != pr["kind"]:
        print("  donor kind mismatch", sp, v, txt, pr["kind"]); continue
    if K not in KL[(sp, v)] or len(KL[(sp, v)][K]) < 2:
        continue
    donors[(K, KL[(sp, v)][K])].append((sp, v, kind, float(pr["value_si"]), txt, rows[0].get("prior_obj")))
print("donor keys", len(donors), "donor videos", sum(len(x) for x in donors.values()))

_tj = {}
def trk_all(sp, v, ent):
    """all-box centres {frame: c}, W of the frame, track dir"""
    k = ent_key(ent)
    if (sp, v, k) in _tj:
        return _tj[(sp, v, k)]
    res = (None, None, "no tracks")
    for d in ("v2sc", "v3nt"):
        p = f"{R}/geo/out/tracks_{sp}_{d}/{v}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        q = next((q for q in (j.get("queries_d") or []) + (j.get("queries") or []) if q.get("key") == k and q.get("obj") is not None), None)
        if q is None:
            res = (None, None, f"no query {k} in {d}"); break
        c = {f: np.array(e["st"]["c"], float) for f, e in zip(j["fidx"], j["objs"][str(q["obj"])]) if e and e.get("st")}
        res = (c, float(j.get("W") or 0), d); break
    _tj[(sp, v, k)] = res
    return res

def early0(c, fps):
    f0 = next((f for f in sorted(c) if f / fps <= 0.1 + 1e-9), None)
    f5, f1 = int(round(0.5 * fps)), int(round(1.0 * fps))
    if f0 is None or f5 not in c or f1 not in c:
        return np.nan
    d1 = float(np.hypot(*(c[f1] - c[f0])))
    return float(np.hypot(*(c[f5] - c[f0]))) / d1 if d1 > 1e-6 else np.nan

def pos_gate(rsp, rv, rent, dsp, dv, dent):
    """(b) position identity of receiver target and donor prior object"""
    cr, Wr, dr = trk_all(rsp, rv, rent); cd, Wd, dd = trk_all(dsp, dv, dent)
    if cr is None or cd is None:
        return False, f"pos: receiver {dr} / donor {dd}", {}
    fr, fd = CSV_FPS.get((rsp, rv)), CSV_FPS.get((dsp, dv))
    if not fr or not fd or not Wr or Wr != Wd:
        return False, f"pos: fps {fr}/{fd} W {Wr}/{Wd}", {}
    dist = []
    for f in sorted(cr):
        g = int(round(f / fr * fd))
        if g in cd and abs(g / fd - f / fr) < 0.5 / max(fr, fd):
            dist.append(float(np.hypot(*(cr[f] - cd[g]))))
    er, ed = early0(cr, fr), early0(cd, fd)
    med = float(np.median(dist)) if dist else np.nan
    info = {"n": len(dist), "med": med, "medW": med / Wr if dist else np.nan, "early_r": er, "early_d": ed, "tracks": (dr, dd)}
    s = f"pos: n={len(dist)} median |dc| {med:.2f}px = {info['medW']:.4f} W, early {er:.3f} vs {ed:.3f} [{dr}/{dd}]"
    if len(dist) < 24:
        return False, s + " -> < 24 common frames", info
    if not med <= 0.02 * Wr:
        return False, s + " -> median > 0.02 W", info
    if np.isfinite(er) and np.isfinite(ed) and abs(er - ed) > 0.1:
        return False, s + " -> early fractions differ > 0.1", info
    return True, s, info

def px_ratio(rsp, rv, rent, dsp, dv, dent):
    """median receiver/donor centre pixel speed (+-0.25 s, all boxes) over common times"""
    cr, _, _ = trk_all(rsp, rv, rent); cd, _, _ = trk_all(dsp, dv, dent)
    if cr is None or cd is None:
        return np.nan, 0
    fr, fd = CSV_FPS.get((rsp, rv)), CSV_FPS.get((dsp, dv))
    hr, hd = max(1, int(round(0.25 * fr))), max(1, int(round(0.25 * fd)))
    rat = []
    for f in sorted(cr):
        g = int(round(f / fr * fd))
        if all(x in cr for x in (f - hr, f + hr)) and all(x in cd for x in (g - hd, g + hd)):
            sr = float(np.hypot(*(cr[f + hr] - cr[f - hr]))); sd = float(np.hypot(*(cd[g + hd] - cd[g - hd])))
            if sd > 1e-6 and sr > 1e-6:
                rat.append((sr / (2 * hr / fr)) / (sd / (2 * hd / fd)))
    return (float(np.median(rat)) if rat else np.nan), len(rat)

def phys(lines, kind, val):
    ts = sorted(lines)
    for (t1, d1), (t2, d2) in zip(ts, ts[1:]):
        if t2 <= t1:
            return False, "bad times"
        path = 0.5 * val * (t2 ** 2 - t1 ** 2) if kind == "accel" else val * (t2 - t1)
        if abs(d2 - d1) > 1.05 * path + 1e-9:
            return False, f"|dd|={abs(d2 - d1):.2f} > 1.05*path={1.05 * path:.2f} ({t1}->{t2})"
    return True, "ok"

_nfr = {}
def clip_end(sp, v):
    if (sp, v) not in _nfr:
        import cv2
        vp = next((x for x in (f"{R}/data/{sp}_480p/{v}.mp4", f"{R}/data/test_480p/{v}.mp4", f"{R}/data/val_480p/{v}.mp4") if os.path.exists(x)), "")
        cap = cv2.VideoCapture(vp); n = 0
        while cap.grab():
            n += 1
        _nfr[(sp, v)] = n
    return (_nfr[(sp, v)] - 1) / CSV_FPS[(sp, v)]

def answer(q, kind, val, sp, v):
    unit = str(q.get("unit") or "").lower().replace(" ", "")
    if kind == "accel":
        if q["kind"] == "accel" and q["t"] is not None and unit in U_ACC:
            return val * U_ACC[unit]
        if q["kind"] == "speed" and q["t"] is not None and unit in U_SPEED:
            return val * float(q["t"]) * U_SPEED[unit]
        if q["kind"] == "avgspeed" and unit in U_SPEED:
            t1 = float(q["t1"]) if q.get("t1") is not None else 0.0
            t2 = float(q["t2"]) if q.get("t2") is not None else clip_end(sp, v)
            return val * (t1 + t2) / 2 * U_SPEED[unit]
    else:
        if q["kind"] == "speed" and q["t"] is not None and unit in U_SPEED:
            return val * U_SPEED[unit]
        if q["kind"] == "avgspeed" and unit in U_SPEED:
            return val * U_SPEED[unit]
    return None

def is_target(r, K):
    if [norm(x) for x in (r.get("target_dkeys") or [])] == [K]:
        return True
    ph = norm((r.get("target") or {}).get("phrase") or r["q"].get("target") or "")
    return ph == K or ph.split("_")[-1] == K

out = {sp: {} for sp in SPLITS}
outpx = {sp: {} for sp in SPLITS}
log = []
for (sp, v), rows in sorted(vids.items()):
    if (sp, v) not in KL:
        continue
    own = dyn_prior_keys(rows)
    own_any = any_prior_keys(rows)
    for K, lines in KL[(sp, v)].items():
        if len(lines) < 2 or (K, lines) not in donors:
            continue
        D = [d for d in donors[(K, lines)] if base(d[1]) != base(v)]
        if not D:
            continue
        if K in own:
            mine = [d for d in donors[(K, lines)] if d[1] == v and d[0] == sp]
            log.append((sp, v, K, "receiver has its own dynamic prior on K", "own", [(d[2], d[3]) for d in mine], "others", sorted({(d[1], d[2], d[3]) for d in D})))
            continue
        static_on_K = K in own_any          # V's own (static) prior is about K: main rule skips, flagged variant may take it
        kinds = {d[2] for d in D}; vals = [d[3] for d in D]
        if len(kinds) != 1 or max(vals) / min(vals) > 1.02:
            log.append((sp, v, K, "donors disagree", [(d[1], d[2], d[3]) for d in D])); continue
        kind = kinds.pop(); val = float(np.median(vals))
        ok, why = phys(lines, kind, val)
        if not ok:
            log.append((sp, v, K, "physical guard", why)); continue
        for r in rows:
            if not is_target(r, K):
                continue
            q = r["q"]
            if q["kind"] not in ("speed", "avgspeed", "accel") or (q["kind"] in ("speed", "accel") and q["t"] is None):
                continue
            src = str(DET[sp].answer_source.get(r["id"], ""))
            if src in ("kin", "scene_prior") or src.startswith("patch:simtwin") or (sp, r["id"]) in _done:
                log.append((sp, r["id"], v, K, "skip source", src)); continue
            ans = answer(q, kind, val, sp, v)
            if ans is None or not np.isfinite(ans) or ans <= 0:
                log.append((sp, r["id"], v, K, "no output", q["kind"], q["t"], q.get("unit"))); continue
            gates = []
            for (dsp, dv, dk, dval, dtxt, dent) in D:
                ga, wa, ia = motion_gate(sp, v, r.get("target"), dsp, dv, dent)
                gb, wb, ib = pos_gate(sp, v, r.get("target"), dsp, dv, dent)
                gates.append((dsp, dv, ga, wa, gb, wb, ia, dent))
            old = float(DET[sp].answer.get(r["id"], np.nan))
            rec = {"value": float(ans), "old": old, "src": src, "vid": v, "q": r["question"], "cat": DET[sp]["cat"].get(r["id"]),
                   "K": K, "gates": gates, "donors": D}
            passers = [g for g in gates if g[2] or g[4]]
            if passers and not static_on_K:
                g = passers[0]
                rec["why"] = (f"rul426: {K} depth lines identical to {g[1]} ({D[0][4]}); gate " + ("(a) rul420 motion" if g[2] else "(b) position identity"))
                out[sp][r["id"]] = rec
            elif PX:
                # flagged variant: (a) fails only on coverage
                for (dsp, dv, ga, wa, gb, wb, ia, dent) in gates:
                    if not (ga or gb):          # gate failed: only a coverage failure of (a) is tolerated
                        if not ia or "too few uncut" not in wa:
                            continue
                        if not (ia["spread"] <= 1.2 or not np.isfinite(ia["spread"])):
                            continue
                        if np.isfinite(ia["early_r"]) and np.isfinite(ia["early_d"]) and abs(ia["early_r"] - ia["early_d"]) > 0.1:
                            continue
                    mr, nr = px_ratio(sp, v, r.get("target"), dsp, dv, dent)
                    if nr >= 20 and 0.9 <= mr <= 1.1 and old > 2 * ans:
                        rec["why"] = (f"subtwin_px: {K} depth lines identical to {dv} ({D[0][4]}); pixel-speed ratio {mr:.3f} (n={nr}), old answer > 2x"
                                      + ("; receiver states a static prior on K" if static_on_K else "; gate (a) short on coverage"))
                        outpx[sp][r["id"]] = rec; break
                else:
                    log.append((sp, r["id"], v, K, "gate fail", [(g[1], g[3], g[5]) for g in gates]))
            else:
                log.append((sp, r["id"], v, K, "gate fail", [(g[1], g[3], g[5]) for g in gates]))
for x in log:
    print("  log", x)
# labelled-row guard
from lab_rows import labelled
L = labelled("sub_xt_Nzk")
lab = {(s, int(i)) for s, i in zip(L.split, L.index)}
for nm, O in (("rul426", out), ("subtwin_px", outpx)):
    for sp in SPLITS:
        for i in list(O[sp]):
            if (sp, i) in lab:
                print(f"  LABELLED row {sp} {i} would change -> dropped ({O[sp][i]['old']:.3f} -> {O[sp][i]['value']:.3f}, y={L[(L.split == sp) & (L.index == i)].y.iloc[0]})")
                O[sp].pop(i)
        o = {str(i): {"value": e["value"], "why": e["why"]} for i, e in O[sp].items()}
        if nm == "subtwin_px":
            o = {**{str(i): {"value": e["value"], "why": e["why"]} for i, e in out[sp].items()}, **o}
        json.dump(o, open(f"{R}/cache/tmp/ag_{nm}_patch_{sp}.json", "w"), indent=1)
        print(f"[{nm} {sp}] {len(O[sp])} own rows, {len(o)} in file")
        for i, e in sorted(O[sp].items()):
            print(f"  {sp} {i} {e['vid']} [{e['cat']}] {e['q'][:64]!r} {e['src']} {e['old']:.3f} -> {e['value']:.3f} | {e['why']}")
            for g in e["gates"]:
                print(f"        donor {g[0]} {g[1]}: (a) {'PASS' if g[2] else 'fail'} {g[3]} | (b) {'PASS' if g[4] else 'fail'} {g[5]}")
