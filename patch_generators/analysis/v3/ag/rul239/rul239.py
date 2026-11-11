"""rul239: two coverage fixes of rul426 / rul081 (per-object depth twins), nothing else changed.
 (1) donor key: when the donor's prior-text noun K ('person') is not a literal depth key, use the parser's own prior_dkeys binding
     if it is a single key present in the donor's depth lines ('human') [subtwin_kf.py = rul426.py with this fallback].
 (2) static-on-K receivers (receiver states a static prior on K): position identity on measure_v3's own tracks (median centre
     distance <= 0.02 W over >= 24 common frames) + px-speed ratio in [0.9, 1.1] (n >= 20) + the current answer off the donor
     value by > 2x in EITHER direction (rul081 required old > 2x donor only). Same pixels + same K depth lines = same clip.
Skips rows in ag_subtwin_patch / ag_c47subtwv1_patch / rul420. Labels only evaluate. No ids in the logic."""
"""rul081: coverage fix of the accepted rul426 rule (and its flagged px variant).
rul426 / rul420 read tracks only from tracks_{sp}_v2sc, else _v3nt. measure_v3 (the geo that produced the answers)
reads tracks_{sp}_v2sc, else tracks_{sp}_v2, else tracks_{sp} (v1). For videos without a v2sc file the twin gates
therefore compare v3nt tracks that the measurement never used; in sim_0040 the v3nt 'eagle' sits ~170 px off the real
eagle, so the position-identity gate failed although the measurement's own v1 eagle track is frame-for-frame identical
to the donor sim_0042's (which states 'speed of the eagle = 11.84m/s').
Rule = rul426 unchanged (donors, identical K depth lines, physical guard, row selection, answer()), with gate (b)
position identity re-evaluated on the measurement's own track chain (v2sc -> v2 -> v1) for receiver AND donor, and
with the px variant's extra conditions when the receiver states a static prior on K: median pixel-speed ratio in
[0.9, 1.1] over >= 20 common times and current answer > 2x the donor value. Rows already in ag_subtwin_patch are skipped. Static-on-K receivers (rul426's flagged px variant, never applied) are
admitted only when the position identity holds on the measurement's own tracks (same pixels, not just the same speed profile).
Writes cache/tmp/ag_c47subtwv1_patch_{test,val}.json"""
import json, os, sys, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
src = open(f"{R}/analysis/v3/ag/rul239/d10s3sim_twinkey_kf.py").read()
head = src.split("out = {sp: {} for sp in SPLITS}")[0]
NS = {"__name__": "subtwin_head"}
sys.argv = sys.argv[:1]
exec(compile(head, "subtwin_head", "exec"), NS)
for k in ("vids", "KL", "donors", "dyn_prior_keys", "any_prior_keys", "base", "phys", "is_target", "answer", "ent_key", "CSV_FPS",
          "DET", "_done", "KEYFIX", "early0", "SPLITS", "motion_gate", "pos_gate", "px_ratio"):
    globals()[k] = NS[k]
CUR = {"test": pd.read_csv(f"{R}/submissions/sub_xt_Nzz38_detail.csv").set_index("Unnamed: 0"),
       "val": pd.read_csv(f"{R}/submissions/sub_xt_Nzz38_valdetail.csv").set_index("Unnamed: 0")}
done_st = {sp: set(json.load(open(f"{R}/cache/tmp/ag_subtwin_patch_{sp}.json"))) | set(json.load(open(f"{R}/cache/tmp/ag_c47subtwv1_patch_{sp}.json"))) for sp in SPLITS}
print("KEYFIX", KEYFIX)
_tm = {}
def trk_meas(sp, v, ent):
    """measure_v3's own track chain: v2sc -> v2 -> v1 (plain). Returns ({frame: c}, W, dir)"""
    k = ent_key(ent)
    if (sp, v, k) in _tm:
        return _tm[(sp, v, k)]
    res = (None, None, "no tracks")
    for d in ("_v2sc", "_v2", ""):
        p = f"{R}/geo/out/tracks_{sp}{d}/{v}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        W = float(j.get("W") or 0)
        if "objs" in j:
            q = next((q for q in (j.get("queries_d") or []) + (j.get("queries") or []) if isinstance(q, dict) and q.get("key") == k and q.get("obj") is not None), None)
            if q is None:
                res = (None, None, f"no query {k} in {d or 'v1'}"); break
            c = {f: np.array(e["st"]["c"], float) for f, e in zip(j["fidx"], j["objs"][str(q["obj"])]) if e and e.get("st")}
        else:
            q = next((q for q in (j.get("queries") or []) if q.get("key") == k and q.get("frames") is not None), None)
            if q is None:
                res = (None, None, f"no query {k} in {d or 'v1'}"); break
            c = {f: np.array(e["st"]["c"], float) for f, e in zip(j["fidx"], q["frames"]) if e and e.get("st")}
        res = (c, W, d or "v1"); break
    _tm[(sp, v, k)] = res
    return res

def pos_gate_m(rsp, rv, rent, dsp, dv, dent):
    cr, Wr, dr = trk_meas(rsp, rv, rent); cd, Wd, dd = trk_meas(dsp, dv, dent)
    if cr is None or cd is None:
        return False, f"posm: receiver {dr} / donor {dd}", {}
    fr, fd = CSV_FPS.get((rsp, rv)), CSV_FPS.get((dsp, dv))
    if not fr or not fd or not Wr or Wr != Wd:
        return False, f"posm: fps {fr}/{fd} W {Wr}/{Wd}", {}
    dist = []
    for f in sorted(cr):
        g = int(round(f / fr * fd))
        if g in cd and abs(g / fd - f / fr) < 0.5 / max(fr, fd):
            dist.append(float(np.hypot(*(cr[f] - cd[g]))))
    er, ed = early0(cr, fr), early0(cd, fd)
    med = float(np.median(dist)) if dist else np.nan
    s = f"posm: n={len(dist)} median |dc| {med:.2f}px, early {er:.3f} vs {ed:.3f} [{dr}/{dd}]"
    if len(dist) < 24: return False, s + " -> < 24 common frames", {}
    if not med <= 0.02 * Wr: return False, s + " -> median > 0.02 W", {}
    if np.isfinite(er) and np.isfinite(ed) and abs(er - ed) > 0.1: return False, s + " -> early fractions differ", {}
    return True, s, {"med": med, "n": len(dist), "tracks": (dr, dd)}

def pxr_m(rsp, rv, rent, dsp, dv, dent):
    cr, _, _ = trk_meas(rsp, rv, rent); cd, _, _ = trk_meas(dsp, dv, dent)
    if cr is None or cd is None: return np.nan, 0
    fr, fd = CSV_FPS.get((rsp, rv)), CSV_FPS.get((dsp, dv))
    hr, hd = max(1, int(round(0.25 * fr))), max(1, int(round(0.25 * fd)))
    rat = []
    for f in sorted(cr):
        g = int(round(f / fr * fd))
        if all(x in cr for x in (f - hr, f + hr)) and all(x in cd for x in (g - hd, g + hd)):
            sr = float(np.hypot(*(cr[f + hr] - cr[f - hr]))); sd = float(np.hypot(*(cd[g + hd] - cd[g - hd])))
            if sd > 1e-6 and sr > 1e-6: rat.append((sr / (2 * hr / fr)) / (sd / (2 * hd / fd)))
    return (float(np.median(rat)) if rat else np.nan), len(rat)

out = {sp: {} for sp in SPLITS}
for (sp, v), rows in sorted(vids.items()):
    if (sp, v) not in KL: continue
    own = dyn_prior_keys(rows); own_any = any_prior_keys(rows)
    for K, lines in KL[(sp, v)].items():
        if len(lines) < 2 or (K, lines) not in donors: continue
        D = [d for d in donors[(K, lines)] if base(d[1]) != base(v)]
        if not D or K in own: continue
        static_on_K = K in own_any
        kinds = {d[2] for d in D}; vals = [d[3] for d in D]
        if len(kinds) != 1 or max(vals) / min(vals) > 1.02: continue
        kind = kinds.pop(); val = float(np.median(vals))
        ok, why = phys(lines, kind, val)
        if not ok: continue
        for r in rows:
            if not is_target(r, K): continue
            q = r["q"]
            if q["kind"] not in ("speed", "avgspeed", "accel") or (q["kind"] in ("speed", "accel") and q["t"] is None): continue
            src0 = str(DET[sp].answer_source.get(r["id"], ""))
            if src0 in ("kin", "scene_prior") or src0.startswith("patch:simtwin") or (sp, r["id"]) in _done: continue
            if str(r["id"]) in done_st[sp]: continue
            ans = answer(q, kind, val, sp, v)
            if ans is None or not np.isfinite(ans) or ans <= 0: continue
            old = float(CUR[sp].answer.get(r["id"], np.nan)); src = str(CUR[sp].answer_source.get(r["id"], ""))
            for (dsp, dv, dk, dval, dtxt, dent) in D:
                ga, wa, ia = motion_gate(sp, v, r.get("target"), dsp, dv, dent)
                gb, wb, ib = pos_gate(sp, v, r.get("target"), dsp, dv, dent)
                gm, wm, im = pos_gate_m(sp, v, r.get("target"), dsp, dv, dent)
                mr, nr = pxr_m(sp, v, r.get("target"), dsp, dv, dent)
                print(f"{sp} {r['id']} {v} K={K} donor {dv} val {val} | old {old:.3f} ({src}) -> {ans:.3f} | static_on_K={static_on_K}"
                      f"\n    (a) {ga} {wa}\n    (b) {gb} {wb}\n    (b-meas) {gm} {wm}\n    pxratio {mr:.3f} n={nr}")
                if (ga or gb) and not static_on_K:
                    continue          # rul426's own rows (already applied)
                if gm and (not static_on_K or (nr >= 20 and 0.9 <= mr <= 1.1 and max(old / ans, ans / old) > 2)):
                    tr_ = im.get("tracks"); md_ = im.get("med"); nn_ = im.get("n")
                    why_ = "rul239: %s depth lines identical to %s (%s); position identity on measure_v3 own tracks %s (median dc %.2f px, n=%s), px-speed ratio %.3f; was %.4g (%s)" % (K, dv, dtxt, tr_, md_, nn_, mr, old, src)
                    out[sp][str(r["id"])] = {"value": float(ans), "why": why_}
                    break
# labelled rows are only reported (evaluation), never used to select or drop rows
sys.path.insert(0, R + "/cache/tmp")
from lab_rows import labelled
L = labelled("sub_xt_Nzz38")
lab = {(s_, int(i_)) for s_, i_ in zip(L.split, L.index)}
for sp in SPLITS:
    for i in list(out[sp]):
        if (sp, int(i)) in lab:
            print("  LABELLED row changed", sp, i, out[sp][i])
for sp in SPLITS:
    json.dump(out[sp], open(R + "/cache/tmp/ag_d10s3sim_twinkey_patch_%s.json" % sp, "w"), indent=1)
    print(sp, len(out[sp]), out[sp])
