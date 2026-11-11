"""jump scan: prior-object tracks of 2D dynamic-prior rows (speed / non-gravity acceleration).
Flags per-frame jumps of the prior track (centroid step, rigid box-size step, IoU with another same-class query object)
and re-estimates the prior px quantity on the cleaned track with measure_v3.prior_kin."""
import os, sys, json, re, math, copy
os.environ.setdefault("QP_PRIOR_DRIFT", "1.5"); os.environ.setdefault("QP_PRIOR_ACCGUARD", "3")
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import numpy as np, pandas as pd
import measure_v3 as MV
import track_common as tc
MV.FIX = set("abcdef")

NONRIGID = (r"(person|man|woman|boy|girl|people|pedestrian|player|walker|runner|child|kid|lady|human|cat|dog|horse|bird|animal|cow|"
            r"sheep|deer|fish|snake|duck|goose|pigeon|eagle|butterfly|insect|bee|worm|swimmer|dancer|skater|cyclist|rider|monkey|"
            r"elephant|tiger|lion|bear|rabbit|chicken|phoenix|dragon|flag|cloth|rope|water|smoke|fire|balloon|jellyfish|octopus)s?$")


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def theil_sen_speed(t, P):
    if len(t) < 4:
        return np.nan
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]; m = dt > 1e-6
    sx = np.median((P[j, 0] - P[i, 0])[m] / dt[m]); sy = np.median((P[j, 1] - P[i, 1])[m] / dt[m])
    return float(math.hypot(sx, sy))


def flags(tr, V, fps, step_k=3.0, floor_f=0.15, size_r=1.4, iou_t=0.3):
    """per okc-frame flags: step = okc steps k->k+1 with a centroid jump, size = rigid box-size jumps, iou = okc indices
    where the box overlaps another same-class query object"""
    t, uv = tr.t, tr.uv
    js = tr.okc
    n = len(js)
    out = dict(step=[], size=[], iou=[], med=np.nan, thr=np.nan)
    if n < 6:
        return out
    nf = np.maximum(np.round(np.diff(t) * fps), 1.0)
    st = np.linalg.norm(np.diff(uv, axis=0), axis=1) / nf
    med = float(np.median(st))
    thr = step_k * med + floor_f * tr.size_px
    out["med"], out["thr"] = med, thr
    out["step"] = [int(k) for k in np.where(st > thr)[0]]
    out["maxstep"] = float(st.max() / max(thr, 1e-6))
    head = tc.head_noun(tr.q.get("phrase") or "")
    rigid = not re.search(NONRIGID, head or "") and not re.search(MV.WINGED, tr.q.get("phrase") or "")
    out["rigid"] = rigid
    if rigid:
        bx = np.array([tr.st[j]["box"] for j in js], float)
        w = np.maximum(bx[:, 2] - bx[:, 0], 1); h = np.maximum(bx[:, 3] - bx[:, 1], 1)
        rw = w[1:] / w[:-1]; rh = h[1:] / h[:-1]
        bd = np.array([(tr.st[j].get("border", 0) or 0) for j in js])
        ok = (bd[1:] == 0) & (bd[:-1] == 0)          # a box cut by the border changes size legitimately
        out["size"] = [int(k) for k in np.where(ok & ((np.maximum(rw, 1 / rw) > size_r) | (np.maximum(rh, 1 / rh) > size_r)))[0]]
    po = tr.q.get("obj")
    others = {}
    for k_, o_tr in V.tr.items():
        o = o_tr.q.get("obj") if o_tr is not None else None
        if o is None or o == po or o in others:
            continue
        if tc.head_noun(o_tr.q.get("phrase") or "") == head:
            others[o] = o_tr
    for kk, j in enumerate(js):
        b = tr.st[j]["box"]
        for o, o_tr in others.items():
            s2 = o_tr.st[j] if j < len(o_tr.st) else None
            if s2 and iou(b, s2["box"]) > iou_t:
                out["iou"].append(kk); break
    out["n_same"] = len(others)
    return out


def sub_track(tr, keep_idx):
    """a copy of Track tr restricted to okc positions keep_idx (indices into okc)"""
    c = copy.copy(tr)
    keep_idx = np.asarray(sorted(set(int(i) for i in keep_idx)), int)
    c.okc = tr.okc[keep_idx]
    c.t = tr.t[keep_idx]
    c.uv = np.asarray(tr.uv, float)[keep_idx]
    c.n = len(c.okc)
    c._segs = None
    return c


def pieces(n, cuts):
    out, s = [], 0
    for k in sorted(set(cuts)):
        out.append(list(range(s, k + 1))); s = k + 1
    out.append(list(range(s, n)))
    return [p for p in out if p]


def clean_variants(tr, fl, anchor_j):
    n = tr.n
    bad_iou = set(fl["iou"])
    cuts = sorted(set(fl["step"]) | set(fl["size"]))
    dropA = set(bad_iou)
    for k in cuts:
        dropA.add(k); dropA.add(k + 1)
    A = [i for i in range(n) if i not in dropA]
    ps = pieces(n, cuts)
    ai = int(np.argmin(np.abs(tr.okc - anchor_j))) if anchor_j is not None else None
    pa = next((p for p in ps if ai in p), max(ps, key=len)) if ai is not None else max(ps, key=len)
    B = [i for i in pa if i not in bad_iou]
    pl = max(ps, key=len)
    C = [i for i in pl if i not in bad_iou]
    return dict(A=A, B=B, C=C, npieces=len(ps))


def main(split, tracks="v2sc"):
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[int(r["id"])] = r
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    rows = []
    byv = {}
    for i, r in P.items():
        pr = (r.get("priors") or [None])[0]
        if r["cat"] != "D2" or pr is None or pr.get("kind") not in ("speed", "accel") or pr.get("gravity") or pr.get("value_si") is None:
            continue
        byv.setdefault(r["video_id"], []).append(r)
    for v, rs in sorted(byv.items()):
        p = f"{R}/geo/out/tracks_{split}_{tracks}/{v}.json"
        if not os.path.exists(p):
            p = f"{R}/geo/out/tracks_{split}_v2/{v}.json" if tracks.startswith("v2") else p
        if not os.path.exists(p):
            continue
        vj = json.load(open(p))
        V = MV.Video(vj, "e")
        fps = float(vj["fps"])
        cache = {}
        for r in rs:
            pr = r["priors"][0]
            tr = V.get(r["prior_obj"]) if r["prior_obj"].get("phrase") else None
            x = raw.loc[r["id"]]
            d = dict(id=r["id"], video_id=v, pkind=pr["kind"], pt=pr.get("t"), pt1=pr.get("t1"), prior=r["prior_text"][:40],
                     merge_src=x.merge_src, mode=x.meas_mode, mpp=x.meas_mpp, geo=x.geo_value, why=x.meas_why, value_si=pr["value_si"])
            if tr is None or tr.n < 6:
                d["status"] = "no_prior_track"; rows.append(d); continue
            key = (id(tr), pr["kind"], pr.get("t"), pr.get("t1"), pr.get("t2"))
            if key not in cache:
                fl = flags(tr, V, fps)
                anc = (tr.q.get("anchor") or [None])[0]
                old = MV.prior_kin(tr, pr)
                cv = clean_variants(tr, fl, anc)
                res = dict(n=tr.n, size=round(tr.size_px, 1), med_step=fl["med"], thr=fl["thr"], maxstep=fl.get("maxstep"), n_step=len(fl["step"]),
                           n_size=len(fl["size"]), n_iou=len(fl["iou"]), n_same=fl.get("n_same", 0), rigid=fl.get("rigid"),
                           npieces=cv["npieces"], old=old, ts_full=theil_sen_speed(tr.t, tr.uv),
                           steps=",".join(str(k) for k in fl["step"][:6]), sizes=",".join(str(k) for k in fl["size"][:6]),
                           ious=",".join(str(k) for k in fl["iou"][:8]), anchor=anc)
                for nm in ("A", "B", "C"):
                    idx = cv[nm]
                    if len(idx) >= 6:
                        c = sub_track(tr, idx)
                        res["new" + nm] = MV.prior_kin(c, pr)
                        res["n" + nm] = len(idx)
                        res["ts" + nm] = theil_sen_speed(c.t, c.uv)
                    else:
                        res["new" + nm] = np.nan; res["n" + nm] = len(idx); res["ts" + nm] = np.nan
                cache[key] = res
            d.update(cache[key]); d["status"] = "ok"
            d["obj"] = tr.q.get("obj")
            rows.append(d)
    df = pd.DataFrame(rows)
    df["implied"] = [vs / m if (pd.notna(m) and m > 0) else np.nan for vs, m in zip(df.value_si, df.mpp)]
    df.to_csv(f"{R}/cache/tmp/ag_jump_scan_{split}_{tracks}.csv", index=False)
    return df


if __name__ == "__main__":
    tracks = os.environ.get("JUMP_TRACKS", "v2sc")
    for s in sys.argv[1:]:
        df = main(s, tracks)
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
        ok = df[df.status == "ok"]
        print(s, tracks, "rows", len(df), "ok", len(ok), "videos", df.video_id.nunique())
        m = ok.implied.notna() & ok["mode"].astype(str).str.startswith("2d_ratio") & (ok.merge_src == "d")
        rr = (ok.old[m] / ok.implied[m])
        print("old/implied (2d_ratio, merge d): n", int(m.sum()), "within 1%", round(float((np.abs(np.log(rr)) < 0.01).mean()), 3),
              "quantiles", rr.quantile([0, .05, .5, .95, 1]).round(3).tolist())
        u = ok.drop_duplicates(["video_id", "obj", "pkind", "pt", "pt1"])
        print("unique prior tracks", len(u), "with step jumps", int((u.n_step > 0).sum()), "size jumps", int((u.n_size > 0).sum()), "iou", int((u.n_iou > 0).sum()))
        print("pkind/time:", u.groupby(["pkind", u.pt.notna(), u.pt1.notna()]).size().to_dict())
        f = u[(u.n_step > 0) | (u.n_size > 0) | (u.n_iou > 0)].copy()
        for c in ("newA", "newB", "newC"):
            f["r" + c[-1]] = f.old / f[c]
        f["rts"] = f.old / f.ts_full
        cols = ["video_id", "obj", "prior", "n", "size", "med_step", "maxstep", "n_step", "steps", "n_size", "sizes", "n_iou", "n_same", "rigid",
                "npieces", "anchor", "old", "rts", "rA", "rB", "rC", "nA", "nB", "nC"]
        print(f[cols].round(3).to_string())
