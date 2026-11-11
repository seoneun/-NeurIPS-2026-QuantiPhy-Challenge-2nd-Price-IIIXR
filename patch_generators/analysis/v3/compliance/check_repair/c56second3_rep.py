"""c56second3_rep.py BASE: per-question version of analysis/v3/ag/rul099 (rule.py, run_fusion.py, mkpatch.py).
Rule (kept from rule.py, code copied; rul417/rule.py helpers imported read-only): a 2D dist2 row whose target and target2
collapse onto one query key, where rul417 gives up with 'too few det frames'; the other instance is searched in the
alternate track runs tracks_<split>{_v3,_v3nt32,_v3nt,_v2,_v3ntmv} (first object with the target's head noun that is
physically distinct, of comparable area and present around the asked time); distance = centroid distance at t x 2D
scale x rul417's identity hedge.
Changes vs production:
  - 2D scale: rule.py took the median meas_mpp of the OTHER rows of the video measured in 2D under the same prior (other
    questions' measurements); here the row's own scale: grpC_pre_ownmeas.py (owntracks/meas.py, own-text pool, var m) on
    the row itself with the asked quantity set to the speed of the row's own target at the asked time
    (GRPC_FIXJSON q.kind=speed, q.unit=m/s, target2 removed); the scale is that measurement's mpp (the measure_v3 2D scale
    from the row's own prior text and its own prior-object track; the asked distance itself is not measured there);
  - alternate-run objects: only queries in the row's own-text pool (owntracks/owntext.py own_keys, mode tok) are searched
    (rule.py took any query with the same head noun);
  - RAW = the clean main geo table (GEO_T), DET = BASE (gate: base answer_source not patch);
  - fusion: the clean strict fusion (grpC_strict.py) of the main table with the rows set to 2d_ratio (rule.py RAW2);
    mkpatch.py gate kept: main answer changes vs xt_<B>, merged BASE answer equals the main base answer (rtol 1e-9),
    base source not patch.
Patch -> cache/tmp/ag_c56second3_patch_{test,val}.json."""
import os, sys, json, math, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
sys.path.insert(0, R + "/geo")
spec = importlib.util.spec_from_file_location("s2rule", f"{R}/analysis/v3/ag/rul417/rule.py")
S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
sys.path.insert(0, f"{R}/analysis/v3/compliance/owntracks")
from owntext import own_keys  # noqa: E402
ALT = ["_v3", "_v3nt32", "_v3nt", "_v2", "_v3ntmv"]
UNIT = S2.UNIT
DIAG = {}


def trigger_rows(split, RAW, DET):
    """= rule.py trigger_rows (rul417's gates up to its 'too few det frames' exit)"""
    out = []
    for d in (json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")):
        i = d["id"]; vid = str(d["video_id"]); q = d["q"]; t2 = d.get("target2") or {}
        if q.get("kind") != "dist2" or not t2.get("phrase") or d["video_type"][1] != "2":
            continue
        if np.isfinite(RAW.geo_value.get(i, np.nan)) or str(DET.answer_source.get(i, "")).startswith("patch"):
            continue
        tf = f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json"
        if not os.path.exists(tf):
            continue
        vj = json.load(open(tf))
        qm = {qq["key"]: qq.get("obj") for qq in vj["queries"]}
        o1, o2 = qm.get(S2.key(d["target"])), qm.get(S2.key(t2))
        if o1 is None or str(o1) not in vj["objs"]:
            continue
        if not (o2 is None or str(o1) == str(o2) or (str(o2) in vj["objs"] and S2.same_phys(vj, o1, o2))):
            continue
        hd = S2.head(d["target"]["phrase"])
        sec = t2.get("where") == ["#second"]
        contr = (S2.head(d["target"].get("raw")) == S2.head(t2.get("raw")) == hd and d["target"].get("raw") != t2.get("raw")
                 and S2.key(d["target"]) == S2.key(t2))
        if not ((sec and vid.startswith("simulation")) or contr):
            continue
        s1 = S2.ostats(vj, o1)
        if s1 is None or len(s1["ok"]) < 0.5 * s1["n"]:
            continue
        if len(S2.other_from_tracks(vj, o1, s1)) != 0:
            continue
        got, msg = S2.other_from_dets(vj, o1, hd, s1)
        if got is None or len(got[0]) >= 5:
            continue
        out.append((d, vj, o1, s1, hd))
    return out


def alt_pts(aj, o, sx, sy):
    T = np.array(aj["fidx"], float) / float(aj["fps"])
    fr = aj["objs"][str(o)]
    ok = [j for j, f in enumerate(fr) if f and f.get("st")]
    P = np.array([fr[j]["st"]["c"] for j in ok], float).reshape(-1, 2) * np.array([sx, sy])
    A = np.array([fr[j]["st"]["area"] for j in ok], float) * sx * sy
    return T[ok], P, A, [aj["fidx"][j] for j in ok]


def own_scale(split, rows):
    """own 2D scale per row: the row measured as the speed of its own target at its own time (own prior, own tracks)"""
    if not rows:
        return {}
    fix = {split: {str(d["id"]): {"q": {"kind": "speed", "unit": "m/s"}, "target2": None} for d in rows}}
    fp = f"{R}/cache/tmp/ag_c56second3_rep_fix.json"
    json.dump(fix, open(fp, "w"))
    m, dg = G.ownmeas(split, "m", {str(d["video_id"]) for d in rows}, "_grpCc56_scale", {"GRPC_FIXJSON": fp})
    out = {}
    for d in rows:
        i = d["id"]
        out[i] = dict(mpp=float(m.mpp.get(i, np.nan)) if bool(m.ok.get(i, False)) and "2d" in str(m["mode"].get(i)) else np.nan,
                      mode=m["mode"].get(i), why=m.why.get(i), pool=dg.get(str(i), {}).get("pool", {}).get("why"))
    return out


def edit(sp, u, RAW):
    DET = G.detail(G.BASE, sp)
    T = trigger_rows(sp, RAW, DET)
    SC = own_scale(sp, [t[0] for t in T])
    ch = []
    for d, vj, o1, s1, hd in T:
        i = d["id"]; vid = str(d["video_id"]); q = d["q"]
        rec = dict(id=i, vid=vid, cat=d["cat"], old=DET.answer.get(i), old_src=DET.answer_source.get(i), val=np.nan, why="",
                   scale=SC.get(i))
        DIAG.setdefault(sp, {})[i] = rec
        t1_, P1, B1 = S2.track_pts(vj, o1)
        f1 = {f: p for f, p in zip([vj["fidx"][j] for j in s1["ok"]], P1)}
        t0 = q.get("t")
        if t0 is None:
            rec["why"] = "no asked time"; continue
        found = None
        for run in ALT:
            af = f"{R}/geo/out/tracks_{sp}{run}/{vid}.json"
            if not os.path.exists(af):
                continue
            aj = json.load(open(af))
            if abs(float(aj["fps"]) - float(vj["fps"])) > 1e-6:
                continue
            sx, sy = float(vj["W"]) / float(aj["W"]), float(vj["H"]) / float(aj["H"])
            okeys, _ = own_keys(d, list(aj.get("queries", []) or []) + list(aj.get("queries_d", []) or []), d, mode="tok")
            objs = []
            for cfg in ("queries", "queries_d"):
                for qq in aj.get(cfg, []) or []:
                    if qq.get("obj") is not None and S2.head(qq.get("phrase")) == hd and str(qq["obj"]) in aj["objs"] and str(qq["obj"]) not in objs:
                        objs.append(str(qq["obj"]))
                        if qq["key"] not in okeys:
                            rec.setdefault("not_own", []).append(f"{run}:{qq['obj']}:{qq['key']}")
                            objs.pop()
            for o in objs:
                ta, Pa, Aa, fa = alt_pts(aj, o, sx, sy)
                if len(ta) == 0:
                    continue
                sh = [np.linalg.norm(p - f1[f]) for f, p in zip(fa, Pa) if f in f1]
                if len(sh) < 3:
                    continue
                gA = float(np.median(sh)) > max(20.0, 0.5 * math.sqrt(float(np.median(Aa))))
                ar = float(np.median(Aa)) / max(s1["area"], 1.0)
                gB = 0.25 <= ar <= 4.0
                nt = int((np.abs(ta - t0) <= 0.1 + 1e-9).sum())
                gC = nt >= 3
                rec.setdefault("cands", []).append(f"{run}:{o} d{np.median(sh):.0f} ar{ar:.2f} n{nt} pass{int(gA and gB and gC)}")
                if gA and gB and gC:
                    found = (run, o, ta, Pa); break
            if found:
                break
        if not found:
            rec["why"] = "no alt object passes gates"; continue
        run, o, ta, Pa = found
        a, b = S2.loc_fit(t1_, P1, t0), S2.loc_fit(ta, Pa, t0)
        if a is None or b is None:
            rec["why"] = "no position at t"; continue
        px = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        mpp = SC.get(i, {}).get("mpp", np.nan)
        if not np.isfinite(mpp):
            rec["why"] = "no own 2D scale"; continue
        hedge = 1.0
        ppx1 = float(np.linalg.norm(S2.line_fit(t1_, P1)[0][1])); ppx2 = float(np.linalg.norm(S2.line_fit(ta, Pa)[0][1]))
        pk = S2.key(d.get("prior_obj"))
        if pk is not None and pk == S2.key(d["target"]) and ppx1 > 0 and ppx2 > 0:
            hedge = math.sqrt(ppx1 / ppx2)
        val = px * mpp * hedge * UNIT.get(q.get("unit") or "m", 1.0)
        rec.update(src=f"{run}:{o}", px=round(px, 1), mpp=mpp, ppx1=round(ppx1, 1), ppx2=round(ppx2, 1), hedge=round(hedge, 3), val=val,
                   why=f"rul099: rul417 other instance from alt run {run} obj {o}; centroids at t={t0}s {px:.0f}px x own mpp {mpp:.5f} x id-hedge {hedge:.3f}")
        RAW.loc[i, ["geo_value", "meas_value", "parsed_value"]] = val
        RAW.loc[i, "meas_valid"] = True
        RAW.loc[i, "meas_mode"] = "2d_ratio"
        RAW.loc[i, "meas_why"] = np.nan
        RAW.loc[i, "meas_mpp"] = mpp * hedge
        RAW.loc[i, "meas_gap"] = False
        RAW.loc[i, "answer_source"] = "meas"
        ch.append(i)
    for i, r in DIAG.get(sp, {}).items():
        print(sp, i, r)
    return ch


out = G.run("rul099", edit)
for sp, (n, b, bs, rows) in out.items():
    keep = [i for i in G.gate(n, b, bs, rows) if np.isclose(b.answer[i], bs.answer[i], rtol=1e-9)]
    P = {i: {"value": float(n.answer[i]),
             "why": f"{DIAG[sp][i]['why']}; geo {n.g[i]:.4g} -> strict fusion {n.answer_source[i]}; was {bs.answer_source[i]} {bs.answer[i]:.4g}"}
         for i in keep}
    for i in keep:
        print(sp, i, P[i])
    print(f"[rul099 {sp}] trigger rows {len(DIAG.get(sp, {}))}, geo rows set {len(rows)}, written {len(keep)}")
    G.write_patch("rul099", sp, P)
json.dump({sp: {str(k): v for k, v in d.items()} for sp, d in DIAG.items()}, open(f"{R}/cache/tmp/ag_c56second3_rep_diag.json", "w"),
          indent=1, default=str)
