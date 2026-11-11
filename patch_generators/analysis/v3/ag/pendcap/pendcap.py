"""R3 pendcap: oscillation speed cap for a pendulum swinging (mostly) along the line of sight.

Gate (spec: audit14_synthesis R3; implementation choices marked [impl]):
  * cat D3, kind speed (instantaneous, timed), row not already patched (answer_source patch*)
  * >= 3 timed depth keys of the target's own key
  * box-size timing: >= 3 alternating extrema, each with prominence >= 15% of the median size, spacing CV <= 0.2
    -> T = 2 x median spacing (time = frame index / dataset fps column, never the container fps)
      [impl] size = sqrt(mask area) with a 3-frame running mean: the box height of a ~25 px ball moves in 1 px
             steps (4% each) and its plateaus make the extremum frame ambiguous; the literal box-height count is
             logged as h_spec.
      [impl] the first / last extremum of the clip may be one-sided (the clip starts or ends within a quarter period
             of a turning point): it needs the 15% prominence on its inner side and a strict turn on its outer side
             (a move > 0.5% of the median, the plateau tolerance), so it is a real turning point, not the clip edge.
  * the own depth keys reverse direction at least once, and every reversal lines up (<= 2 frames) with a box extremum
    of the opposite sense (depth min <-> size max).
      [impl] the reversal time is the parabola vertex through the three keys around it (a key sampled every 0.3 s is up
             to 3.6 frames from the true turning point); the literal sampled-key distance is logged.
      [impl, extra safety] with >= 5 own keys, a free-period sinusoid fit of the keys must give a period within 20% of T.
  * box size is used for TIMING only, never as a depth value.
Cap (spec):
  Ar = (max key d - min key d) / 2
  Al = (2D centroid px range over one period around t) / 2 * mean d / f   (f: lab pooled 0.602W @853 / 0.642W @854,
       else the row's geo focal meas_f)
  A = hypot(Ar, Al); vmax = 2 pi A / T; answer = min(answer, 1.15 vmax)      (converted to the question's unit)
Phase variant (tag rul333): min(answer, 1.15 vmax |sin(2 pi (t - t_ext) / T)| + 0.1 vmax), t_ext = box extremum
  nearest t. [impl guard] only for a planar swing (the ball stops at both turning points): the centroid coordinate with
  the larger range must move in phase with the size signal (R^2 >= 0.7 of a linear fit on it); an elliptical / conical
  swing (lateral motion out of phase, speed never 0) keeps the plain cap.
usage: python pendcap.py [sub_name]   (default sub_xt_Nzi)
"""
import sys, os, json, math
import numpy as np, pandas as pd

R = "/data/iiixr/users/sehong/quantiphy"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzi"
ALLCAT = os.environ.get("PC_ALLCAT") == "1"        # diagnostics only: run the gate on every category (never written)
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
TRK = "v2sc"
PROM = 0.15
EDGE = 0.005
PLANAR_R2 = 0.7
CVMAX = 0.2
FR_TOL = 2.0
MARGIN = 1.15
UNIT = {"m/s": 1.0, "cm/s": 100.0, "mm/s": 1000.0, "km/h": 3.6, "km/s": 0.001}


def load_parse(split):
    out = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        out[int(j["id"])] = j
    return out


_TC = {}


def load_tracks(split, vid):
    k = (split, vid)
    if k in _TC:
        return _TC[k]
    vj = None
    for p in (f"{R}/geo/out/tracks_{split}_{TRK}/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json",
              f"{R}/geo/out/tracks_{split}/{vid}.json"):
        if os.path.exists(p):
            vj = json.load(open(p)); vj["_path"] = p.split("/")[-2]
            break
    _TC[k] = vj
    return vj


def ent_key(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def track_frames(vj, key):
    """per-fidx list of st dicts (or None) for the query key, mirroring measure_v3.Video"""
    if vj is None or key is None:
        return None
    if vj.get("version") == "v2" or "objs" in vj:
        for qs_name in ("queries", "queries_d"):
            for q in vj.get(qs_name) or []:
                if q.get("key") == key:
                    if q.get("obj") is None:
                        return None
                    return [f.get("st") if f else None for f in vj["objs"][str(q["obj"])]]
        return None
    for q in vj.get("queries") or []:
        if q.get("key") == key:
            return [f.get("st") if f else None for f in q["frames"]]
    return None


def _plateau_centre(s, i, tol):
    a = b = i
    while a - 1 >= 0 and abs(s[a - 1] - s[i]) <= tol:
        a -= 1
    while b + 1 < len(s) and abs(s[b + 1] - s[i]) <= tol:
        b += 1
    return 0.5 * (a + b)


def extrema(s, thr, edge=None, tol=0.0):
    """alternating extrema of s (zigzag with threshold thr). Interior extrema are confirmed by a move >= thr on both
    sides. With edge (a px amount), the first leg's start and the last leg's end also count when they are turning
    points with an outer-side move >= edge. Returns [(position (float, frame units), +1 max / -1 min, one_sided)]."""
    n = len(s)
    if n < 5:
        return []
    piv = []
    hi = lo = 0
    i0 = None
    for i in range(1, n):
        if s[i] > s[hi]: hi = i
        if s[i] < s[lo]: lo = i
        if s[hi] - s[lo] >= thr:
            i0 = i; break
    if i0 is None:
        return []
    if hi > lo:
        cdir, ci, start = +1, hi, lo
    else:
        cdir, ci, start = -1, lo, hi
    if edge is not None and start > 0:
        outer = (s[:start].max() - s[start]) if cdir == +1 else (s[start] - s[:start].min())
        if outer >= edge:
            piv.append((_plateau_centre(s, start, tol), -cdir, True))
    for i in range(i0 + 1, n):
        if cdir == +1:
            if s[i] > s[ci]:
                ci = i
            elif s[ci] - s[i] >= thr:
                piv.append((_plateau_centre(s, ci, tol), +1, False)); cdir, ci = -1, i
        else:
            if s[i] < s[ci]:
                ci = i
            elif s[i] - s[ci] >= thr:
                piv.append((_plateau_centre(s, ci, tol), -1, False)); cdir, ci = +1, i
    if edge is not None and ci < n - 1:
        outer = (s[ci] - s[ci + 1:].min()) if cdir == +1 else (s[ci + 1:].max() - s[ci])
        if outer >= edge:
            piv.append((_plateau_centre(s, ci, tol), cdir, True))
    return piv


def vertex_time(ts, ds, k):
    """parabola vertex through keys k-1, k, k+1 (clipped to [t_{k-1}, t_{k+1}])"""
    x = np.array(ts[k - 1:k + 2], float); y = np.array(ds[k - 1:k + 2], float)
    c = np.polyfit(x, y, 2)
    if abs(c[0]) < 1e-12:
        return float(ts[k])
    tv = -c[1] / (2 * c[0])
    return float(np.clip(tv, x[0], x[2]))


def key_sine_T(ts, ds):
    """free-period sinusoid fit d = m + a cos wt + b sin wt; returns (T, A, rms) of the best period in [0.4, 4] s"""
    ts = np.asarray(ts, float); ds = np.asarray(ds, float)
    best = None
    for T in np.arange(0.4, 4.0, 0.005):
        w = 2 * np.pi / T
        X = np.c_[np.ones(len(ts)), np.cos(w * ts), np.sin(w * ts)]
        c = np.linalg.lstsq(X, ds, rcond=None)[0]
        e = float(np.sqrt(np.mean((ds - X @ c) ** 2)))
        if best is None or e < best[2]:
            best = (float(T), float(np.hypot(c[1], c[2])), e)
    return best


def analyse(split, parse, det, raw):
    rows = []
    for i, r in det.iterrows():
        j = parse.get(int(i))
        if j is None:
            continue
        q = j.get("q") or {}
        if q.get("kind") != "speed" or q.get("t") is None:
            continue
        if j.get("cat") != "D3" and not ALLCAT:
            continue
        own = j.get("target_dkeys") or []
        dep = [d for d in (j.get("depth") or []) if d.get("key") in own and d.get("t") is not None]
        if not own or len(dep) < 3:
            continue
        kbest = max(own, key=lambda k_: sum(1 for d in dep if d["key"] == k_))
        dd = {}
        for d in dep:
            if d["key"] == kbest:
                dd.setdefault(float(d["t"]), float(d["d"]))
        ts = sorted(dd); ds = [dd[x] for x in ts]
        if len(ts) < 3:
            continue
        rec = dict(split=split, id=int(i), vid=j["video_id"], cat=j.get("cat"), q=j["question"].strip()[:60], t=float(q["t"]),
                   unit=q.get("unit"), src=str(r.get("answer_source")), ans=float(r["answer"]), g=r.get("g"), nkeys=len(ts),
                   dmin=min(ds), dmax=max(ds))
        fps = float(j.get("fps") or 0)                     # dataset fps column (never the container rate)
        vj = load_tracks(split, j["video_id"])
        fr = track_frames(vj, ent_key(j.get("target")))
        if fr is None or fps <= 0:
            rec["why"] = "no_track"; rows.append(rec); continue
        fidx = np.array(vj["fidx"], float)
        st = fr
        val = np.array([s is not None for s in st])
        area = np.array([s["area"] if s else 0 for s in st], float)
        med = np.median(area[val]) if val.any() else 0
        ok = val & (area > 0.4 * med) & (area < 2.5 * med) if med > 0 else val
        idx = np.where(ok)[0]
        if len(idx) < 8:
            rec["why"] = "short_track"; rows.append(rec); continue
        fi = fidx[idx]
        tt = fi / fps
        box = np.array([st[k]["box"] for k in idx], float)
        cc = np.array([st[k]["c"] for k in idx], float)
        # literal spec signal (logged only): raw box height, interior two-sided extrema
        hh = box[:, 3] - box[:, 1]
        ph_ = extrema(hh, PROM * float(np.median(hh)))
        rec["h_spec"] = ";".join(f"{'M' if s_ > 0 else 'm'}{fi[int(round(p))]:.0f}" for p, s_, _ in ph_)
        rec["h_spec_n"] = len(ph_)
        # [impl] timing signal: sqrt(mask area), 3-frame running mean
        sa = np.sqrt(np.array([st[k]["area"] for k in idx], float))
        ss = np.convolve(np.pad(sa, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        smed = float(np.median(ss))
        piv = extrema(ss, PROM * smed, edge=EDGE * smed, tol=0.005 * smed)
        pf = np.array([np.interp(p, np.arange(len(fi)), fi) for p, _, _ in piv])     # frame index (fidx units)
        rec.update(trk=vj.get("_path"), nfr=len(idx), npiv=len(piv),
                   piv=";".join(f"{'M' if s_ > 0 else 'm'}{f_:.1f}{'e' if e_ else ''}" for f_, (_, s_, e_) in zip(pf, piv)))
        if len(piv) < 3:
            rec["why"] = "lt3_extrema"; rows.append(rec); continue
        pt = pf / fps
        sp = np.diff(pt)
        cv = float(np.std(sp) / np.mean(sp))
        T = 2 * float(np.median(sp))
        rec.update(cv=cv, T=T)
        if cv > CVMAX:
            rec["why"] = "cv"; rows.append(rec); continue
        # depth-key reversals vs box extrema (opposite sense)
        dif = np.sign(np.diff(ds))
        rev = [k for k in range(1, len(ds) - 1) if dif[k - 1] != 0 and dif[k] != 0 and dif[k - 1] != dif[k]]
        if not rev:
            rec["why"] = "no_reversal"; rows.append(rec); continue
        okv, okl, msg, nchk = True, True, [], 0
        for k in rev:
            want = +1 if dif[k - 1] < 0 else -1          # depth min -> size max
            tv = vertex_time(ts, ds, k)
            cand = [pt[m] for m, (_, s_, _) in enumerate(piv) if s_ == want]
            if tv < tt[0] - FR_TOL / fps or tv > tt[-1] + FR_TOL / fps:
                msg.append(f"k{ts[k]}:outside"); continue
            dv = min([abs(c - tv) for c in cand]) * fps if cand else 99
            dl = min([abs(c - ts[k]) for c in cand]) * fps if cand else 99
            msg.append(f"k{ts[k]}:v{tv:.3f}/{dv:.1f}f,l{dl:.1f}f"); nchk += 1
            okv &= dv <= FR_TOL
            okl &= dl <= FR_TOL
        okv = okv and nchk > 0; okl = okl and nchk > 0
        rec.update(rev=" ".join(msg), rev_ok=okv, rev_lit=okl)
        if not okv:
            rec["why"] = "rev_mismatch"; rows.append(rec); continue
        if len(ts) >= 5:
            Tk, Ak, ek = key_sine_T(ts, ds)
            rec.update(T_keys=Tk, A_keys=Ak, rms_keys=ek)
            if abs(Tk / T - 1) > 0.2:
                rec["why"] = "keys_T_mismatch"; rows.append(rec); continue
        # cap
        Ar = (max(ds) - min(ds)) / 2
        W = int(vj.get("W") or 0)
        if j.get("video_source") == "lab" and W in (853, 854):
            f = (0.602 if W == 853 else 0.642) * W
        else:
            try:
                f = float(raw.loc[int(i), "meas_f"])
            except Exception:
                f = float("nan")
            if not np.isfinite(f) or f <= 0:
                rec["why"] = "no_focal"; rows.append(rec); continue
        t0 = rec["t"]
        win = (tt >= t0 - T / 2) & (tt <= t0 + T / 2)
        if win.sum() < 3:
            win = np.ones(len(tt), bool)
        rx = float(cc[win, 0].max() - cc[win, 0].min()); ry = float(cc[win, 1].max() - cc[win, 1].min())
        Al = 0.5 * math.hypot(rx, ry) * float(np.mean(ds)) / f
        A = math.hypot(Ar, Al)
        vmax = 2 * math.pi * A / T
        u = UNIT.get(str(q.get("unit")), None)
        if u is None:
            rec["why"] = "unit"; rows.append(rec); continue
        cap = MARGIN * vmax * u
        ci_ = 0 if rx >= ry else 1
        X_ = np.c_[np.ones(len(ss)), ss]
        res_ = cc[:, ci_] - X_ @ np.linalg.lstsq(X_, cc[:, ci_], rcond=None)[0]
        r2 = 1 - float(np.sum(res_ ** 2)) / max(float(np.sum((cc[:, ci_] - cc[:, ci_].mean()) ** 2)), 1e-9)
        text = float(pt[int(np.argmin(np.abs(pt - t0)))])
        sph = abs(math.sin(2 * math.pi * (t0 - text) / T))
        cap_ph = (MARGIN * vmax * sph + 0.1 * vmax) * u if r2 >= PLANAR_R2 else cap
        rec.update(planar_r2=r2, f=f, Ar=Ar, Al=Al, A=A, vmax=vmax, cap=cap, t_ext=text, sinph=sph, cap_ph=cap_ph, why="gate_ok",
                   new=min(rec["ans"], cap), new_ph=min(rec["ans"], cap_ph))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 50); pd.set_option("display.max_colwidth", 60)
    out = {}
    for split, suf in (("val", "valdetail"), ("test", "detail")):
        det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv").set_index("Unnamed: 0")
        raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
        D = analyse(split, load_parse(split), det, raw)
        out[split] = D
        print(f"== {split}: speed rows with >=3 own timed keys{' (all cats)' if ALLCAT else ' (D3)'}: {len(D)}")
        if len(D):
            print(D.why.value_counts().to_string())
            cols = [c for c in ["id", "vid", "cat", "t", "unit", "src", "ans", "g", "h_spec", "piv", "cv", "T", "T_keys", "rev",
                                "rev_lit", "planar_r2", "Ar", "Al", "vmax", "cap", "t_ext", "sinph", "cap_ph", "why"] if c in D.columns]
            print(D[cols].round(3).to_string())
        D.to_csv(f"{R}/cache/tmp/ag_pendcap_scan_{split}{'_allcat' if ALLCAT else ''}.csv", index=False)
    if ALLCAT:
        return
    for tag, col in (("pendcap", "new"), ("rul333", "new_ph")):
        for split in ("test", "val"):
            D = out[split]
            P = {}
            if len(D) and "new" in D.columns:
                for _, r in D[(D.why == "gate_ok")].iterrows():
                    if str(r["src"]).startswith("patch") or r["cat"] != "D3":
                        continue
                    if r[col] < r["ans"] * 0.999:
                        cap = r["cap"] if col == "new" else r["cap_ph"]
                        P[str(int(r["id"]))] = dict(value=float(r[col]),
                                                    why=f"{tag}: T={r['T']:.2f}s A={r['A']:.3f}m vmax={r['vmax']:.2f} cap={cap:.2f} (was {r['ans']:.3f})")
            json.dump(P, open(f"{R}/cache/tmp/ag_{tag}_patch_{split}.json", "w"), indent=1)
            print(f"wrote ag_{tag}_patch_{split}.json: {len(P)} rows")


if __name__ == "__main__":
    main()
