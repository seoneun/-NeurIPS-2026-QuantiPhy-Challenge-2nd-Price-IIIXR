"""rul254: rul426 donor for receivers whose own prior is a STATIC size prior on K (the case rul426 skips / subtwin_px
sends to pixel identity). Copied from d10s3sim_twinkey_kf.py (rul426 machinery), TAG renamed.
  receiver V: simulation row asking speed@t / accel@t / avgspeed of K; V's only prior is a static size prior on K.
  donor U   : base(U) != base(V); exactly one prior, a time-less dynamic prior on K; U's K depth lines (>= 2, d to 4 dp) == V's;
              all donor copies agree within 2% and have one kind.
  gates     : (a) foreign-key guard: every distance_<key>_camera key in V's and in U's depth table names an object found in that
                  video's own questions / prior text / target(2)_dkeys / prior_dkeys;
              (b) physical: consecutive K lines, 0.5 x mean kinematic speed <= |dd|/dt <= 1.05 x mean kinematic speed;
              (c) own witness: some V geo or CaW value on a K kinematic row within 1.25x of the donor-implied value for that row;
              (d) skip answer_source kin / scene_prior / simtwin* / rul426 / rul081.
  output    : accel a -> speed@t = a t, avgspeed[t1,t2] = a (t1+t2)/2 (only with explicit t2), accel@t = a; speed v -> v.
Writes cache/tmp/ag_d21fam2_dkeyacc_s3st_patch_{test,val}.json."""
import json, re, sys, collections, os
import numpy as np, pandas as pd
TAG = "rul254"
BUILD = "sub_xt_Nzz41"
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/cache/tmp"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3")
SPLITS = ("test", "val")
P = {sp: [json.loads(l) for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl")] for sp in SPLITS}
DET = {"test": pd.read_csv(f"{R}/submissions/{BUILD}_detail.csv").set_index("Unnamed: 0"),
       "val": pd.read_csv(f"{R}/submissions/{BUILD}_valdetail.csv").set_index("Unnamed: 0")}
RAW = {sp: pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", usecols=["Unnamed: 0", "video_id", "depth_info"]).set_index("Unnamed: 0")
       for sp in SPLITS}
TIMEWORD = re.compile(r"\b(before|after|until|from|between|at|during|when|first|last)\b")
U_SPEED = {"m/s": 1.0, "cm/s": 100.0, "km/h": 3.6, "mm/s": 1000.0}
U_ACC = {"m/s^2": 1.0, "m/s2": 1.0, "cm/s^2": 100.0, "m/s²": 1.0}
LINE = re.compile(r"t\s*=\s*([0-9.]+)\s*s\s*,\s*distance_(.+?)_camera\s*=\s*([0-9.]+)", re.I)
ANYKEY = re.compile(r"distance_(.+?)_camera", re.I)
PRIOR = re.compile(r"^\s*(?:the\s+)?(speed|acceleration)\s+of\s+(?:the\s+)?(.+?)\s*(?:=|\bis\b)\s*(-?[0-9.]+)", re.I)
SKIP = ("kin", "scene_prior")
SKIP_PAT = ("simtwin", "rul426", "rul081")


def norm(k):
    return re.sub(r"\s+", "_", str(k).strip().lower())


def base(v):
    return str(v).replace("_segmented", "")


def klines(txt):
    out = collections.defaultdict(set)
    for t, k, d in LINE.findall(str(txt)):
        out[norm(k)].add((round(float(t), 4), round(float(d), 4)))
    return {k: frozenset(v) for k, v in out.items()}


vids = collections.defaultdict(list)
for sp in SPLITS:
    for p in P[sp]:
        if str(p["video_id"]).startswith("simulation_"):
            vids[(sp, p["video_id"])].append(p)
KL, KEYS = {}, {}
for (sp, v), rows in vids.items():
    ks = {tuple(sorted((k, tuple(sorted(s))) for k, s in klines(RAW[sp].depth_info.get(r["id"], "")).items())) for r in rows}
    if len(ks) == 1:
        di = RAW[sp].depth_info.get(rows[0]["id"], "")
        KL[(sp, v)] = klines(di)
        KEYS[(sp, v)] = {norm(k) for k in ANYKEY.findall(str(di))}


def own_objects_ok(sp, v):
    """gate (a): every depth key names an object of the video's own questions / prior / dkeys"""
    rows = vids[(sp, v)]
    dk = set()
    for r in rows:
        for f in ("target_dkeys", "target2_dkeys", "prior_dkeys"):
            dk |= {norm(x) for x in (r.get(f) or [])}
    text = re.sub(r"[^a-z]", "", " ".join(str(r.get("question", "")) + " " + str(r.get("prior_text", "")) for r in rows).lower())
    bad = [k for k in KEYS.get((sp, v), ()) if k not in dk and re.sub(r"[^a-z]", "", k) not in text]
    return (not bad), bad


donors = collections.defaultdict(list)
for (sp, v), rows in vids.items():
    if (sp, v) not in KL:
        continue
    pri = rows[0]["priors"]
    if len(pri) != 1 or pri[0]["kind"] not in ("speed", "accel"):
        continue
    pr = pri[0]
    txt = str(rows[0]["prior_text"])
    if any(pr.get(x) is not None for x in ("t", "t1", "t2")) or TIMEWORD.search(txt.lower().split("=")[0]):
        continue
    m = PRIOR.match(txt)
    if not m:
        continue
    K = norm(m.group(2))
    kind = "accel" if m.group(1).lower().startswith("acc") else "speed"
    if kind != pr["kind"] or K not in KL[(sp, v)] or len(KL[(sp, v)][K]) < 2:
        continue
    donors[(K, KL[(sp, v)][K])].append((sp, v, kind, float(pr["value_si"]), txt))
print("donor keys", len(donors))


def phys(lines, kind, val):
    ts = sorted(lines)
    for (t1, d1), (t2, d2) in zip(ts, ts[1:]):
        if t2 <= t1:
            return False, "bad times"
        mean_v = val * (t1 + t2) / 2 if kind == "accel" else val
        r = abs(d2 - d1) / (t2 - t1) / mean_v
        if not (0.5 <= r <= 1.05):
            return False, f"|dd|/dt / mean speed = {r:.3f} ({t1}->{t2})"
    return True, "ok"


def answer(q, kind, val):
    unit = str(q.get("unit") or "").lower().replace(" ", "")
    if kind == "accel":
        if q["kind"] == "accel" and q["t"] is not None and unit in U_ACC:
            return val * U_ACC[unit]
        if q["kind"] == "speed" and q["t"] is not None and unit in U_SPEED:
            return val * float(q["t"]) * U_SPEED[unit]
        if q["kind"] == "avgspeed" and unit in U_SPEED and q.get("t2") is not None:
            t1 = float(q["t1"]) if q.get("t1") is not None else 0.0
            return val * (t1 + float(q["t2"])) / 2 * U_SPEED[unit]
    else:
        if (q["kind"] == "avgspeed" or (q["kind"] == "speed" and q["t"] is not None)) and unit in U_SPEED:
            return val * U_SPEED[unit]
    return None


def is_target(r, K):
    if [norm(x) for x in (r.get("target_dkeys") or [])] == [K]:
        return True
    ph = norm((r.get("target") or {}).get("phrase") or r["q"].get("target") or "")
    return ph == K or ph.split("_")[-1] == K


def static_on_K(rows, K):
    pri = rows[0]["priors"]
    if len(pri) != 1 or pri[0]["kind"] != "size":
        return False
    o = norm(pri[0].get("obj") or "")
    pk = {norm(x) for r in rows for x in (r.get("prior_dkeys") or [])}
    return o == K or o.split("_")[-1] == K or pk == {K}


out = {sp: {} for sp in SPLITS}
log = []
for (sp, v), rows in sorted(vids.items()):
    if (sp, v) not in KL:
        continue
    for K, lines in KL[(sp, v)].items():
        if len(lines) < 2 or (K, lines) not in donors:
            continue
        D = [d for d in donors[(K, lines)] if base(d[1]) != base(v)]
        if not D:
            continue
        if not static_on_K(rows, K):
            log.append((sp, v, K, "receiver prior not static on K", rows[0]["prior_text"], [d[1] for d in D]))
            continue
        kinds = {d[2] for d in D}
        vals = [d[3] for d in D]
        if len(kinds) != 1 or max(vals) / min(vals) > 1.02:
            log.append((sp, v, K, "donors disagree", D))
            continue
        kind = kinds.pop()
        val = float(np.median(vals))
        okV, badV = own_objects_ok(sp, v)
        badU = [(d[1], own_objects_ok(d[0], d[1])[1]) for d in D if not own_objects_ok(d[0], d[1])[0]]
        if not okV or badU:
            log.append((sp, v, K, "foreign key", badV, badU))
            continue
        ok, why = phys(lines, kind, val)
        if not ok:
            log.append((sp, v, K, "physical", why))
            continue
        cand, wit = [], []
        for r in rows:
            q = r["q"]
            if not is_target(r, K) or q["kind"] not in ("speed", "avgspeed", "accel"):
                continue
            if q["kind"] in ("speed", "accel") and q["t"] is None:
                continue
            ans = answer(q, kind, val)
            if ans is None or not np.isfinite(ans) or ans <= 0:
                continue
            g, c = DET[sp].g.get(r["id"], np.nan), DET[sp].cs.get(r["id"], np.nan)
            for nm, x in (("geo", g), ("CaW", c)):
                if pd.notna(x) and x > 0 and max(x / ans, ans / x) <= 1.25:
                    wit.append((r["id"], nm, round(float(x), 3), round(ans, 3)))
            cand.append((r, ans))
        if not wit:
            log.append((sp, v, K, "no own witness", [(r["id"], DET[sp].g.get(r["id"]), DET[sp].cs.get(r["id"]), a) for r, a in cand]))
            continue
        for r, ans in cand:
            src = str(DET[sp].answer_source.get(r["id"], ""))
            s0 = src.replace("patch:", "")
            if s0 in SKIP or any(s0.startswith(x) for x in SKIP_PAT):
                log.append((sp, r["id"], "skip source", src))
                continue
            out[sp][r["id"]] = {"value": float(ans), "old": float(DET[sp].answer.get(r["id"], np.nan)), "vid": v, "q": r["question"],
                                "cat": DET[sp]["cat"].get(r["id"]), "src": src,
                                "why": f"{TAG}: {K} depth lines identical to {D[0][1]} ({D[0][4]}); own witness {wit[0][1]} row {wit[0][0]} {wit[0][2]} vs {wit[0][3]}"}
for x in log:
    print("  log", x)
from lab_rows import labelled
import analyze_caw as A
L = labelled(BUILD)
for sp in SPLITS:
    lab = L[L.split == sp]
    for i, e in sorted(out[sp].items()):
        s = f"  {sp} {i} {e['vid']} [{e['cat']}] {e['q'][:60]!r} {e['src']} {e['old']:.3f} -> {e['value']:.3f}"
        if i in lab.index:
            y = float(lab.y.loc[i])
            s += f"  LABELLED y={y} mra {A.row_mra(np.array([e['old']]), np.array([y]))[0]:.2f}->{A.row_mra(np.array([e['value']]), np.array([y]))[0]:.2f} (dropped)"
        print(s)
        print("      ", e["why"])
    o = {str(i): {"value": e["value"], "why": e["why"]} for i, e in out[sp].items() if i not in lab.index}
    json.dump(o, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
    print(f"[{sp}] {len(o)} rows written")
