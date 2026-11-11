"""rul047 (label-free; labels only evaluate).

Scope: gravity-prior rows whose base geo failed the free-fall detector (raw meas_why == 'prior_no_freefall') and were then
filled by a CaW-size fallback ruler (raw meas_mode in {2d_ratio, 2d_vlmruler}); geo finite; answer_source not a patch.

Physics: the target itself falls. A quadratic fit of its v2sc box-centre y after the rest phase gives a_px, so
mpp = g_prior / a_px (planar drop, constant depth), independent of CaW's size guess. The fit vertex t_v is the effective
release time of the parabola.
  rest = median of the first max(3, n/8) samples; motion starts where cy - rest > 3 px; stop at the first upward step
  < -2 px; need >= 8 samples, R2 > 0.99, a_px > 0.

Timeline (the change against the synthesis spec): clips re-encoded 60 -> 50 fps drop one source frame in six, so every
5th step is doubled and k/fps is a sawtooth around real time. Candidates: the nominal timeline T = k/fps and the five
phases of the 6:5 pulldown T = (k + floor((k + p)/5)) / (1.2 fps) (CSV fps 25 or 50, consecutive frames only;
at 24 fps a 6:5 cadence is meaningless, and captured_0018's better 6:5 fit there is left alone). Each is fitted with up to
two trailing-outlier trims (last residual > max(4 px, 4 x median |res| of the others)). A pulldown phase is used only if
its RMS residual is < half the nominal one. Question times map to frames by k = t * fps (CSV fps) and then to T.
  internet_0029: nominal R2 0.9967, max res 15.7 px (a 4624, t_v 0.072) vs pulldown p=3, last frame trimmed:
  R2 0.99997, max res 1.3 px (a 3971 px/s^2 -> mpp 0.00247, t_v 0.045 s).

Re-measure with that mpp:
  speed at t (t_v < T(t) <= last fitted sample): v = mpp * hypot(vx_lin, a_px (T(t) - t_v)), vx_lin = slope of cx vs T
    over T >= t_v;
  disp t1 -> t2: centre displacement x mpp;
  gap 'distance of X from Y at t': (median y1 of Y's box - X's y2 at t) x mpp.
Keep only if |new / geo - 1| > 0.15. Final answer = the build's fusion with CaW: within 2x -> geo for --geo_kinds
(dist2, avgspeed), size g^0.65 c^0.35, else geometric mean; outside 2x -> geo (the gravity-calibrated 2D ratio is a
strong mode); no CaW -> geo.

usage: tools/run_limited.sh --io python analysis/v3/ag/rul047/rul047.py
 -> cache/tmp/ag_c32ngpropf_patch_{test,val}.json, cache/tmp/ag_c32ngpropf_diag_{split}.csv
"""
import json, os, re, sys, numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzt"
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
UNIT = {'m': 1.0, 'meters': 1.0, 'meter': 1.0, 'cm': 100.0, 'mm': 1000.0, 'm/s': 1.0, 'cm/s': 100.0}
GEO_KINDS = {"dist2", "avgspeed"}          # build flag --geo_kinds dist2,avgspeed
GATE = 0.15
SPEC = os.environ.get("C32NGPF_SPEC") == "1"   # reproduce the synthesis spec: nominal timeline, no trim, geo as final


def parse(split):
    d = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l)
        d[int(j['id'])] = j
    return d


def load_track(split, vid, phrase):
    try:
        T = json.load(open(f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json"))
    except Exception:
        return None
    phrase = re.sub(r"^(the )?(bottom|top|center|centre|base) of (the )?", "", str(phrase or ""))
    obs = [q.get('obj') for q in T['queries'] if q.get('phrase') == phrase and q.get('obj') is not None]
    if not obs:
        obs = [q.get('obj') for q in T['queries'] if phrase and phrase in str(q.get('phrase')) and q.get('obj') is not None]
    if not obs:
        return None
    A = [(f, *s['st']['box']) for f, s in zip(T['fidx'], T['objs'][str(obs[0])]) if s is not None and s.get('st')]
    return np.array(A, float) if len(A) else None


def timelines(fidx, fps):
    out = {"nominal": fidx / fps}
    if not SPEC and int(round(fps)) in (25, 50) and len(fidx) > 1 and np.all(np.diff(fidx) == 1):   # 30->25 / 60->50 re-encodes only
        for p in range(5):
            out[f"pd65p{p}"] = (fidx + np.floor((fidx + p) / 5)) / (1.2 * fps)
    return out


def quadfit(T, y, seg):
    idx = np.array(seg)
    for _ in range(2):
        c = np.polyfit(T[idx], y[idx], 2)
        res = y[idx] - np.polyval(c, T[idx])
        if not SPEC and len(idx) > 8 and abs(res[-1]) > max(4.0, 4 * np.median(np.abs(res[:-1]))):
            idx = idx[:-1]
            continue
        break
    c = np.polyfit(T[idx], y[idx], 2)
    res = y[idx] - np.polyval(c, T[idx])
    r2 = 1 - np.sum(res ** 2) / np.sum((y[idx] - y[idx].mean()) ** 2)
    return dict(a=2 * c[0], tv=-c[1] / (2 * c[0]), r2=r2, rms=float(np.sqrt(np.mean(res ** 2))), mx=float(np.abs(res).max()), idx=idx)


def ffit(A, fps):
    fidx = A[:, 0]
    cy = (A[:, 2] + A[:, 4]) / 2
    n = len(cy)
    rest = np.median(cy[:max(3, n // 8)])
    mv = np.where(cy - rest > 3)[0]
    if len(mv) < 8:
        return None
    seg = np.arange(mv[0], n)
    rev = np.where(np.diff(cy[seg]) < -2)[0]
    if len(rev):
        seg = seg[:rev[0] + 1]
    if len(seg) < 8:
        return None
    TL = timelines(fidx, fps)
    fits = {k: quadfit(T, cy, seg) for k, T in TL.items()}
    nom = fits["nominal"]
    pds = [k for k in fits if k != "nominal"]
    name = "nominal"
    if pds:
        b = min(pds, key=lambda k: fits[k]["rms"])
        if fits[b]["rms"] < 0.5 * nom["rms"]:
            name = b
    f = dict(fits[name])
    f.update(name=name, T=TL[name], nom=nom)
    return f


def gprior(txt):
    m = re.search(r"(\d+(?:\.\d+)?)\s*m/s", str(txt))
    g = float(m.group(1)) if m else 9.8
    return g if 9.0 <= g <= 10.5 else None


def fuse(new, cs, kind):
    if not (np.isfinite(cs) and cs > 0):
        return new, "geo (no CaW)"
    r = new / cs
    if 0.5 <= r <= 2.0:
        if kind in GEO_KINDS:
            return new, "agree_geo_kind"
        if kind == "size":
            return new ** 0.65 * cs ** 0.35, "agree_wgm"
        return float(np.sqrt(new * cs)), "agree_gm"
    return new, "dis_geo"


def run(split):
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    PP = parse(split)
    out, diag = {}, []
    for i, r in raw.iterrows():
        j = PP.get(int(i))
        if j is None or i not in det.index:
            continue
        pt = str(r.ground_truth_prior).lower()
        if not ('gravit' in pt or '9.8' in pt):
            continue
        g0 = gprior(pt)
        q = j['q']
        unit = UNIT.get(str(q.get('unit')).lower())
        if g0 is None or unit is None:
            continue
        src = str(det.loc[i, 'answer_source']).replace('mo_guard:', '')
        if src.startswith('patch:'):
            continue
        if not (str(r.meas_why) == 'prior_no_freefall' and str(r.meas_mode) in ('2d_ratio', '2d_vlmruler')):
            continue
        geo = float(r.geo_value)
        if not (np.isfinite(geo) and geo > 0):
            continue
        fps = float(r.fps)                      # CSV fps (never the container fps)
        A = load_track(split, r.video_id, q.get('target'))
        if A is None:
            continue
        f = ffit(A, fps)
        if f is None or f['r2'] <= 0.99 or f['a'] <= 0:
            diag.append(dict(id=i, vid=r.video_id, kind=q.get('kind'), skip='no_fit' if f is None else f"r2 {f['r2']:.4f} a {f['a']:.0f}"))
            continue
        a, tv, T = f['a'], f['tv'], f['T']
        mpp = g0 / a
        fidx = A[:, 0]
        cx = (A[:, 1] + A[:, 3]) / 2
        cy = (A[:, 2] + A[:, 4]) / 2
        I = lambda arr, tq: float(np.interp(tq * fps, fidx, arr))
        k = q.get('kind')
        new, how = None, ""
        if k == 'speed' and q.get('t') is not None:
            Tq = float(np.interp(float(q['t']) * fps, fidx, T))
            Tlast = float(T[f['idx'][-1]])
            if tv < Tq <= Tlast:
                sel = T >= tv
                vx = np.polyfit(T[sel], cx[sel], 1)[0] if sel.sum() >= 3 else 0.0
                new = float(np.hypot(vx, a * (Tq - tv)) * mpp)
                how = f"v=mpp*hypot(vx {vx:.0f}, a(T-tv)) at T {Tq:.4f}"
        elif k == 'disp' and q.get('t1') is not None and q.get('t2') is not None:
            px = np.hypot(I(cx, q['t2']) - I(cx, q['t1']), I(cy, q['t2']) - I(cy, q['t1']))
            new = float(px * mpp)
            how = f"centre disp {px:.1f}px"
        elif k == 'dist2' and q.get('t') is not None and q.get('target2'):
            B = load_track(split, r.video_id, q['target2'])
            if B is not None:
                top = float(np.median(B[:, 2]))
                px = top - I(A[:, 4], q['t'])
                new = float(px * mpp)
                how = f"gap {q['target2']} top {top:.0f} - y2(t) = {px:.1f}px"
        if new is None or not np.isfinite(new) or new <= 0:
            diag.append(dict(id=i, vid=r.video_id, kind=k, skip='no_measure'))
            continue
        new *= unit
        rel = new / geo - 1
        cs = float(det.loc[i, 'cs'])
        fin, fsrc = (new, "spec_geo") if SPEC else fuse(new, cs, k)
        dr = dict(id=i, vid=r.video_id, kind=k, tl=f['name'], a=round(a, 1), tv=round(tv, 4), r2=round(f['r2'], 6), mx=round(f['mx'], 2),
                  nom_a=round(f['nom']['a'], 1), nom_tv=round(f['nom']['tv'], 4), nom_mx=round(f['nom']['mx'], 2), mpp=mpp,
                  old_geo=geo, new_geo=new, rel=round(rel, 3), cs=cs, final=fin, fsrc=fsrc, old_ans=float(det.loc[i, 'answer']), src=src, how=how)
        if abs(rel) <= GATE:
            dr['skip'] = 'within_15pct_of_geo'
            diag.append(dr)
            continue
        diag.append(dr)
        out[str(i)] = {'value': float(fin), 'why': (
            f"rul047: fallback-ruled geo {geo:.4g} ({r.meas_mode}); own free fall on {f['name']} timeline a={a:.0f}px/s^2 "
            f"R2={f['r2']:.5f} t_v={tv:.3f}s -> mpp {mpp:.5f}; {how} -> geo {new:.4g}; CaW {cs:.4g} -> {fsrc} {fin:.4g}; "
            f"was {det.loc[i, 'answer']:.4g} {src}")}
    return out, pd.DataFrame(diag)


if __name__ == '__main__':
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80)
    for split in ['test', 'val']:
        o, D = run(split)
        nm = "c32ngpropfspec" if SPEC else "rul047"
        json.dump(o, open(f"{R}/cache/tmp/ag_{nm}_patch_{split}.json", 'w'), indent=0)
        D.to_csv(f"{R}/cache/tmp/ag_{nm}_diag_{split}.csv", index=False)
        print(split, 'rul047', len(o))
        if len(D):
            print(D.drop(columns=[c for c in ('how',) if c in D.columns]).to_string())
        for kk, v in o.items():
            print(' ', kk, round(v['value'], 4), v['why'])
