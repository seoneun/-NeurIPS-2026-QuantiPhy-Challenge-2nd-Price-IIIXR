import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 900); pd.set_option('display.max_colwidth', 60)
SPLIT = sys.argv[1]
TOL = 2.0
CHORD = 0.25
STOP = {"the", "a", "an", "of", "in", "on"}
POS = {"left": (0, -1), "leftmost": (0, -1), "right": (0, 1), "rightmost": (0, 1),
       "top": (1, -1), "upper": (1, -1), "bottom": (1, 1), "lower": (1, 1)}


def cut_flags(tr, W, H):
    out = []
    for j in tr.okc:
        b = tr.st[j]["box"]
        out.append(bool((b[0] <= TOL) or (b[1] <= TOL) or (b[2] >= W - 1 - TOL) or (b[3] >= H - 1 - TOL)))
    return np.array(out, bool)


def chord_speeds(t, P, h=CHORD):
    out = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + h - 1e-9)
        if j >= len(t):
            break
        out.append(float(np.linalg.norm(P[j] - P[i]) / (t[j] - t[i])))
    return np.array(out)


def theil_sen(t, y):
    n = len(t)
    if n < 3:
        return np.nan
    s = [(y[j] - y[i]) / (t[j] - t[i]) for i in range(n) for j in range(i + 1, n) if t[j] > t[i]]
    return float(np.median(s))


def rcv(x):
    if len(x) < 3:
        return np.nan
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)) / m) if m > 0 else np.nan


def runs(cut, t):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out, s = [], None
    for k in range(len(cut)):
        brk = k > 0 and (t[k] - t[k - 1]) > 3 * dt
        if cut[k] or brk:
            if s is not None:
                out.append((s, k)); s = None
        if not cut[k] and s is None:
            s = k
    if s is not None:
        out.append((s, len(cut)))
    return out


def sing(w):
    if w.endswith("es") and len(w) > 4 and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def words(p):
    return {sing(w) for w in re.findall(r"[a-z]+", (p or "").lower()) if w not in STOP}


def head(p):
    ws = [w for w in re.findall(r"[a-z]+", (p or "").lower()) if w not in STOP]
    return sing(ws[-1]) if ws else ""


def diff_named(qa, qb):
    wa, wb = words(qa.get("phrase")), words(qb.get("phrase"))
    if wa and wb and not (wa <= wb or wb <= wa):
        return True
    ga, gb = set(qa.get("where") or []), set(qb.get("where") or [])
    return bool(ga and gb and ga != gb)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def common(ta, tb):
    n = min(len(ta.st), len(tb.st))
    return [j for j in range(n) if ta.st[j] and tb.st[j] and ta.st[j].get("box") is not None and tb.st[j].get("box") is not None]


def dup(ta, tb):
    com = common(ta, tb)
    if len(com) < 5:
        return False
    io = np.array([iou(ta.st[j]["box"], tb.st[j]["box"]) for j in com])
    return bool(np.median(io) >= 0.8 and (io >= 0.8).mean() >= 0.8)


def shared_info(V, vj, tr, own=None):
    pk, pobj = tr.q.get("key"), tr.q.get("obj")
    own = set(own or [])
    sh_id, sh_dup = [], []
    for k, t2 in V.tr.items():
        if k == pk or not diff_named(tr.q, t2.q) or k not in own:
            continue
        if t2 is tr or (pobj is not None and t2.q.get("obj") == pobj):
            sh_id.append(k)
        elif dup(tr, t2):
            sh_dup.append(k)
    sh_d = []
    qd = vj.get("queries_d") or []
    pd_ = next((q.get("obj") for q in qd if q.get("key") == pk), None)
    if pd_ is not None:
        sh_d = [q["key"] for q in qd if q.get("key") != pk and q.get("obj") == pd_ and diff_named(tr.q, q) and q.get("key") in own]
    pos_ok, pos_why = False, "no_pos_qualifier"
    pw = [w for w in (tr.q.get("where") or []) if w in POS]
    if pw:
        ax, sg = POS[pw[0]]
        h0 = head(tr.q.get("phrase"))
        oth, seen = [], set()
        for k, t2 in V.tr.items():
            o2 = t2.q.get("obj")
            if k == pk or t2 is tr or head(t2.q.get("phrase")) != h0 or (pobj is not None and o2 == pobj) or o2 in seen:
                continue
            if dup(tr, t2):
                continue
            seen.add(o2); oth.append((k, t2))
        if not oth:
            pos_why = "no_other_distinct_instance"
        else:
            pos_ok, msgs = True, []
            for k, t2 in oth:
                com = common(tr, t2)
                if len(com) < 5:
                    pos_ok = False; msgs.append(f"{k}:few_common"); continue
                d = np.array([sg * (tr.st[j]["c"][ax] - t2.st[j]["c"][ax]) for j in com], float)
                ext = float(np.median([tr.st[j]["box"][ax + 2] - tr.st[j]["box"][ax] for j in com]))
                ok = bool(np.median(d) > 0.25 * ext and (d > 0).mean() >= 0.8)
                pos_ok &= ok
                msgs.append(f"{k}:{np.median(d):.0f}px/{(d > 0).mean():.2f}")
            pos_why = ";".join(msgs)
    return dict(pobj=pobj, sh_id=",".join(sh_id), sh_dup=",".join(sh_dup), sh_d=",".join(sh_d), pos_ok=pos_ok, pos_why=pos_why)


def box_at(tr, t, W, H):
    if tr is None:
        return "miss", np.nan
    ta = np.asarray(tr.t_all, float)
    dt = float(np.median(np.diff(ta))) if len(ta) > 1 else 1.0
    j = int(np.argmin(np.abs(ta - t)))
    if abs(ta[j] - t) > 1.5 * dt or not tr.st[j] or tr.st[j].get("box") is None:
        return "miss", np.nan
    b = tr.st[j]["box"]
    m = float(min(b[0], b[1], W - 1 - b[2], H - 1 - b[3]))
    return ("cut" if m <= TOL else "ok"), round(m, 1)


raw = pd.read_csv(f"{R}/preds/{SPLIT}_geo_v4ntqdir3srngsr_d_vw_raw.csv", index_col=0)
MV.TRACK_VER = "v2sc"
byv, vjs = MV.load_split_tracks(SPLIT, "e")
MV.TRACK_VER = "v3ntmv"
_, vjs3 = MV.load_split_tracks(SPLIT, "e")
rows = []
for v, recs in byv.items():
    cand = [r for r in recs if r.get("priors") and r["priors"][0].get("kind") == "speed" and not r["priors"][0].get("gravity")
            and r["priors"][0].get("t") is None and r["priors"][0].get("t1") is None and str(r["video_type"])[1] == "2"]
    if not cand:
        continue
    MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=True)
    VV = {}
    for tv_, src_ in (("v2", vjs), ("v1", vjs3)):
        if v in src_:
            VV[tv_] = (src_[v], MV.Video(src_[v], "e"))
    for r in cand:
        rr = raw.loc[r["id"]] if r["id"] in raw.index else None
        pr = r["priors"][0]
        d = dict(id=r["id"], vid=v, cat=r["cat"], qk=r["q"]["kind"], qt=r["q"].get("t"), praw=pr.get("raw"), vsi=pr.get("value_si"),
                 pobj_ph=(r.get("prior_obj") or {}).get("phrase"), tgt=(r.get("target") or {}).get("phrase"))
        if rr is None:
            d["note"] = "no_raw"; rows.append(d); continue
        d.update(mode=str(rr.meas_mode), mtr=rr.meas_tracks, msrc=rr.merge_src, mpp=rr.meas_mpp, geo=rr.geo_value, meas=rr.meas_value,
                 grav=rr.meas_grav)
        mt_ = rr.meas_tracks if rr.meas_tracks in VV else "v2"
        if mt_ not in VV:
            d["note"] = "no_track_file"; rows.append(d); continue
        vj, V = VV[mt_]
        d["fps"] = vj.get("fps")
        W, H = float(vj["W"]), float(vj["H"])
        tr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
        if tr is None or tr.n == 0:
            d["note"] = "no_prior_track"; rows.append(d); continue
        old = MV.prior_kin(tr, pr)
        df = MV.drift_factor(tr)
        d.update(pkey=tr.q.get("key"), trn=tr.n, old_px=old, drift=df)
        d.update(shared_info(V, vj, tr, own=[MV.Video.key(r.get(role)) for role in ("target", "target2") if r.get(role)]))
        if r["q"]["kind"] == "dist2" and r["q"].get("t") is not None:
            tg = V.get(r["target"])
            tg2 = V.get(r["target2"]) if r.get("target2") else None
            if str(rr.meas_grav) == "second_fix" and tg is not None:
                tg2 = MV.second_instance(V, vj, tg)
            d["tc1"], d["tm1"] = box_at(tg, float(r["q"]["t"]), W, H)
            d["tc2"], d["tm2"] = box_at(tg2, float(r["q"]["t"]), W, H)
            d["tk1"] = tg.q.get("key") if tg is not None else None
            d["tk2"] = tg2.q.get("key") if tg2 is not None else None
        if old is not None and old > 0 and np.isfinite(rr.meas_mpp):
            d["mpp_rep"] = pr["value_si"] / old
            d["mpp_err"] = d["mpp_rep"] / rr.meas_mpp - 1
        t, P = tr.t, tr.uv
        cut = cut_flags(tr, W, H)
        d["clip"] = round(float(cut.mean()), 3); d["nclip"] = int(cut.sum())
        bfield = np.array([(tr.st[j].get("border", 0) or 0) != 0 for j in tr.okc])
        d["clip_bf"] = round(float(bfield.mean()), 3)
        cs_full = chord_speeds(t, P)
        d["chord_full"] = float(np.median(cs_full)) if len(cs_full) else np.nan
        d["cv_full"] = rcv(cs_full)
        rs = runs(cut, t)
        if rs:
            i, j = max(rs, key=lambda x: (t[x[1] - 1] - t[x[0]], x[1] - x[0]))
            tt, PP = t[i:j], P[i:j]
            d.update(run_disp=round(float(np.linalg.norm(P[j - 1] - P[i])), 1), run_i=i, run_j=j, run_n=j - i, run_dur=round(float(tt[-1] - tt[0]), 3), run_t0=round(float(tt[0]), 2), run_t1=round(float(tt[-1]), 2))
            csr = chord_speeds(tt, PP)
            d["chord_run"] = float(np.median(csr)) if len(csr) else np.nan
            d["cv_run"] = rcv(csr)
            if len(tt) >= 3:
                d["ts_run"] = float(np.hypot(theil_sen(tt, PP[:, 0]), theil_sen(tt, PP[:, 1])))
                sg = MV.segment(tt, PP, tr.size_px)
                d["avg_run"] = MV.avg_speed_c(tt, PP, tt[0], tt[-1], sg)
        d["ts_full"] = float(np.hypot(theil_sen(t, P[:, 0]), theil_sen(t, P[:, 1])))
        d["size"] = round(tr.size_px, 1)
        rows.append(d)
D = pd.DataFrame(rows).set_index("id")
D.to_csv(f"{R}/cache/tmp/ag_uncut2S2_diag_{SPLIT}.csv")
