"""rul106: revive evtparse_ff (free-flight acceleration = g at a non-event time t on a lab bouncing-ball video) with a
second, self-ruled gravity witness that needs no focal length or depth.
Rows: q.kind == 'accel' with q.t set and no t1/t2, lab video (captured_*), priors[0] = size/diameter (not gravity) on the
same head noun as q.target (the object is its own ruler).
G1 (evtparse_ff gate): image-track parabola around t has the event-arc curvature (ratio 0.7-1.4, >= 5 frames, no contact
    within 1.5 frames of t) -- needs an apex event row on the same video (evtparse measurement).
G2: f0 = round(t*fps) (CSV fps); y = box-centre y on frames f0-4..f0+1, all present; no local max of y and no sign flip of
    dy inside (no bounce / apex); all d2y one sign with CV < 0.3.
G3: px_diam = median min(box w, h) over the window; a_img = |mean d2y| * fps^2 * D / px_diam in [0.75 g, 1.3 g].
-> geo = 9.8 m/s^2 in the question's unit; then pipeline fusion + copy log-mean (evtparse finals)."""
import os, sys, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/analysis/v3"); sys.path.insert(0, R + "/geo")
sys.path.insert(0, R + "/analysis/v3/ag/rul265"); sys.path.insert(0, R + "/analysis/v3/ag/evtparse")
import evtparse_patch as EP
SUB = os.environ.get("C61_SUB", "sub_xt_Nzz11")
EP.SUB = SUB
G = 9.8
UNIT = {"m/s^2": 1.0, "cm/s^2": 100.0, "mm/s^2": 1000.0, "ft/s^2": 1 / 0.3048}
TDIRS = ("v2sc", "v3ntmv", "v3nt", "v2", "v2mv", "v3")


def head(p):
    w = str(p or "").lower().replace("-", " ").split()
    return w[-1] if w else ""


def load_boxes(split, vid, phrase):
    for tv in TDIRS:
        p = f"{R}/geo/out/tracks_{split}_{tv}/{vid}.json"
        if not os.path.exists(p):
            continue
        vj = json.load(open(p))
        for q in (vj.get("queries") or []):
            if str(q.get("key", "")).split("|")[0].strip().lower() == phrase:
                fr = vj["objs"].get(str(q.get("obj"))) or []
                B = {int(j): f["st"]["box"] for j, f in zip(vj["fidx"], fr) if f and f.get("st")}
                if len(B) >= 10:
                    return B, tv, float(vj["fps"])
    return None, None, None


def g23(split, vid, phrase, t, fps, D):
    B, tv, tfps = load_boxes(split, vid, phrase)
    if B is None:
        return False, False, "no track", np.nan
    f0 = int(round(t * fps))
    fr = list(range(f0 - 4, f0 + 2))
    if any(f not in B for f in fr):
        return False, False, f"[{tv}] window frames {fr[0]}-{fr[-1]} incomplete", np.nan
    bx = np.array([B[f] for f in fr], float)
    y = 0.5 * (bx[:, 1] + bx[:, 3])
    d1 = np.diff(y); d2 = np.diff(y, 2)
    locmax = any(y[k] >= y[k - 1] and y[k] >= y[k + 1] for k in range(1, len(y) - 1))
    flip = bool(np.any(d1[1:] * d1[:-1] < 0))  # strict sign change of dy (a zero step is a flat apex sample, still free flight)
    same = np.all(d2 > 0) or np.all(d2 < 0)
    cv = float(np.std(d2) / abs(np.mean(d2))) if np.mean(d2) != 0 else np.inf
    ok2 = (not locmax) and (not flip) and same and cv < 0.3
    pxd = float(np.median(np.minimum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1])))
    a_img = abs(float(np.mean(d2))) * fps ** 2 * D / pxd
    ok3 = 0.75 * G <= a_img <= 1.3 * G
    why = (f"[{tv}{'' if abs(tfps-fps)<1e-6 else ' trkfps %g' % tfps}] frames {fr[0]}-{fr[-1]} y={np.round(y,1).tolist()} "
           f"d2y={np.round(d2,2).tolist()} cv {cv:.2f} locmax {locmax} flip {bool(flip)}; px diam {pxd:.1f} -> a_img {a_img:.2f} m/s^2")
    return ok2, ok3, why, a_img


def ev_arcs(split):
    mp = f"{R}/cache/tmp/ag_evtparse_meas_{split}.csv"
    out = {}
    if not os.path.exists(mp) or os.path.getsize(mp) < 10:
        return out
    M = pd.read_csv(mp)
    for _, m in M[M.act == "apexacc"].iterrows():
        out.setdefault(str(m.video_id), (float(m.ev_t), EP.num(m.tb1), EP.num(m.tb2), m.ev))
    return out


def run(split):
    D = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    arcs = ev_arcs(split)
    cand, rep, full = {}, [], set()
    for i, r in P.items():
        q = r["q"]; vid = str(r["video_id"]).replace(".mp4", "")
        if q.get("kind") != "accel" or q.get("t") is None or q.get("t1") is not None or q.get("t2") is not None:
            continue
        if r.get("video_source") != "lab" or not vid.startswith("captured_"):
            continue
        pr = (r.get("priors") or [{}])[0]
        if pr.get("gravity") or pr.get("kind") != "size" or pr.get("dim") != "diameter" or not pr.get("value_si"):
            rep.append(dict(id=i, vid=vid, t=q["t"], gate="prior", why=str(pr.get("raw"))[:50])); continue
        phrase = str((r.get("target") or {}).get("phrase") or q.get("target") or "").lower()
        if head(pr.get("obj")) != head(phrase) or q.get("unit") not in UNIT:
            rep.append(dict(id=i, vid=vid, t=q["t"], gate="noun/unit", why=f"{pr.get('obj')} vs {phrase}")); continue
        fps = float(r["fps"]); t = float(q["t"])
        ok2, ok3, w23, a_img = g23(split, vid, phrase, t, fps, float(pr["value_si"]))
        if vid in arcs:
            te, tb1, tb2, lab = arcs[vid]
            ok1, w1 = EP.ff_check(split, vid, phrase, t, te, tb1, tb2)
            ok1 = bool(ok1)
        else:
            ok1, w1 = False, "no apex-event arc on this video"
        cur = D.loc[i, "answer"] if i in D.index else np.nan
        rep.append(dict(id=i, vid=vid, t=t, gate=f"G1 {ok1} G2 {ok2} G3 {ok3}", a_img=round(a_img, 2), cur=cur,
                        src=D.loc[i, "answer_source"] if i in D.index else "", why=f"G1: {w1} | G2/3: {w23}"))
        if ok1 and ok2 and ok3:
            full.add(i)
        if ok1 and i in D.index and not str(D.loc[i, "answer_source"]).startswith("patch"):
            cand[i] = (G * UNIT[q["unit"]], True, "accel",
                       f"rul106 t={t:g}s free flight: event-arc curvature ({w1.split(';')[0]}) and self-ruled image accel "
                       f"{a_img:.2f} m/s^2 (diam {pr['value_si']} m / px diam, no focal/depth) -> |a| = g")
    # copies of one recording (plain/s/x) take the same value: keep G1-passing rows whose recording+question group has
    # at least one member passing all of G1-G3
    gk = lambda i: EP.split_event(str(P[i]["video_id"]).replace(".mp4", ""))[0] + "||" + str(P[i]["question"]).strip().lower()
    fg = {gk(i) for i in full}
    cand = {i: c for i, c in cand.items() if gk(i) in fg}
    print(f"[{split}] full G1-G3 pass: {sorted(full)}; fired (incl. copies): {sorted(cand)}")
    pd.set_option("display.width", 400); pd.set_option("display.max_colwidth", 400); pd.set_option("display.max_rows", 300)
    Rp = pd.DataFrame(rep)
    print(f"[{split}] scanned rows:"); print(Rp.to_string(index=False) if len(Rp) else "(none)")
    patch, Rf = EP.finals(cand, D, "") if cand else ({}, pd.DataFrame())
    patch = {k: dict(value=v["value"], why=v["why"].replace("evtparse: ", "", 1)) for k, v in patch.items()}
    out = f"{R}/cache/tmp/ag_c61ffre_patch_{split}.json"
    json.dump(patch, open(out, "w"), indent=1)
    print(f"[{split}] rul106 patch: {len(patch)} rows -> {out}")
    if len(Rf):
        print(Rf.round(4).to_string(index=False))
    return patch


if __name__ == "__main__":
    for s in sys.argv[1:] or ["test", "val"]:
        run(s)
