"""R5 SLOPESIDE: a size prior that names the object's side "parallel to the slope / incline / ramp / (direction of)
motion" is an extent ALONG the object's velocity direction.  measure.Track.ext_px reads "width" as the min-area
rectangle's SHORT side (rr[1]); rr is stored sorted (long, short, angle) so the side orientation is lost, and for a box
sliding down a 45-degree incline the short side is the ACROSS-slope extent (box height + oblique top face).

Rule (label-free, code only):
  1. rows whose prior raw text matches PAT, prior kind size, final source a plain fusion of a 2d_ratio geo
     (agree_gm / agree_wgm / dis_geo), not already patched.
  2. prior track = the track the pipeline used (meas_tracks v1 -> geo/out/tracks_{split}; else _v2sc), query = prior phrase.
  3. velocity direction u = principal axis of the centroid path (requires a straight path: s2/s1 < 0.15 and
     span >= 3 object sizes).
  4. orientation of the stored rr sides: a background-subtraction mask (temporal median of the static-camera video)
     inside the SAM box gives the span along u and across u in each border-free frame; if along > across the rr LONG
     side is the along-u side, else the SHORT side.
  5. along-u extent e_u(frame) = that rr side; motion blur and the visible end face (perspective) only ADD length
     along u, so the ruler is a low percentile (P10) of e_u over border-free frames.
  6. scale k = old prior px (prior value / meas_mpp) / new prior px; g' = g * k (2d_ratio: every answer is linear in
     m/px); fusion re-run with the pipeline's rule: |log10 g'/c| <= log10 2 -> gm (or the row's own wgm weight),
     else strong-mode geo.
usage: python analysis/v3/ag/rul423/rul423.py test|val
"""
import os, sys, re, json
import numpy as np, pandas as pd, cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo")
import track_common as tc

PAT = re.compile(r"\bparallel\s+to\s+(?:the\s+)?(?:direction\s+of\s+(?:motion|travel|movement)|slope|incline|ramp|motion|movement)\b", re.I)
SUB = os.environ.get("SS_SUB", "sub_xt_Nzg")
RAW = os.environ.get("SS_RAW", "geo_v4ntrj5sQ2srngsr_d_vw")
PCT = float(os.environ.get("SS_PCT", "10"))
split = sys.argv[1]
det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv", index_col=0)
P = {}
for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
    d = json.loads(l)
    P[d["id"]] = d

out, diag = {}, {}
cand = [i for i, d in P.items() if PAT.search(d.get("prior_text") or "")]
print(f"[{split}] prior text matches: {len(cand)} rows {cand[:20]}")
cache = {}


def ruler(vid, tv, phrase):
    key = (vid, tv, phrase)
    if key in cache:
        return cache[key]
    res = dict(ok=False)
    cache[key] = res
    tdir = f"{R}/geo/out/tracks_{split}" + ("" if tv == "v1" else "_v2sc")
    fn = f"{tdir}/{vid}.json"
    if not os.path.exists(fn):
        res["why"] = "no_track"; return res
    vj = json.load(open(fn))
    if "objs" in vj:
        qs = vj.get("queries_d") or vj["queries"]
        q = next((q for q in qs if q["key"].split("|")[0] == phrase), None)
        if q is None or q.get("obj") is None:
            res["why"] = "no_query"; return res
        st = [(f or {}).get("st") for f in vj["objs"][str(q["obj"])]]
    else:
        q = next((q for q in vj["queries"] if q["key"].split("|")[0] == phrase), None)
        if q is None:
            res["why"] = "no_query"; return res
        st = [f.get("st") for f in q["frames"]]
    fidx = vj["fidx"]
    area = np.array([s["area"] if s else 0 for s in st], float)
    val = area > 0
    med = np.median(area[val]) if val.any() else 0
    ok = val & (area > 0.4 * med) & (area < 2.5 * med)
    border = np.array([s["border"] if s else 9 for s in st])
    okc = np.where(ok)[0]
    oke = np.where(ok & (border == 0))[0]
    if len(okc) < 8 or len(oke) < 5:
        res["why"] = "few_frames"; return res
    uv = np.array([st[j]["c"] for j in okc], float)
    X = uv - uv.mean(0)
    _, s_, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]; n = np.array([-u[1], u[0]])
    span = np.ptp(X @ u)
    size = np.median([st[j]["rr"][0] for j in oke])
    lin = s_[1] / max(s_[0], 1e-9)
    if lin > 0.15 or span < 3 * size:
        res.update(why=f"path_not_straight(lin={lin:.2f},span/size={span / size:.1f})"); return res
    # background-subtraction mask for the side orientation
    vp = raw_vpath.get(vid) or f"{R}/data/{split}_480p/{vid}.mp4"
    F = np.array(tc.load_video(vp))
    bg = np.median(F, 0).astype(np.float32)
    al, ac, rl, rs = [], [], [], []
    for j in oke:
        fi = fidx[j]
        if fi >= len(F):
            continue
        x0, y0, x1, y1 = st[j]["box"]
        a, b, c, e = max(0, y0), min(F.shape[1], y1), max(0, x0), min(F.shape[2], x1)
        dif = np.abs(F[fi, a:b, c:e].astype(np.float32) - bg[a:b, c:e]).sum(2)
        m = (dif > 30).astype(np.uint8)
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if nl < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 0.3 * st[j]["area"]:
            continue
        pts = np.stack([xs + c, ys + a], 1).astype(float)
        pu, pn = pts @ u, pts @ n
        al.append(np.percentile(pu, 99) - np.percentile(pu, 1))
        ac.append(np.percentile(pn, 99) - np.percentile(pn, 1))
        rl.append(st[j]["rr"][0]); rs.append(st[j]["rr"][1])
    if len(al) < 5:
        res["why"] = "few_mask_frames"; return res
    al, ac, rl, rs = map(np.array, (al, ac, rl, rs))
    along_long = bool(np.median(al) > np.median(ac))
    e_u = rl if along_long else rs
    e_n = rs if along_long else rl
    res.update(ok=True, u=u.round(3).tolist(), lin=round(float(lin), 3), span_over_size=round(float(span / size), 1),
               along_long=along_long, px_new=float(np.percentile(e_u, PCT)), px_new_med=float(np.median(e_u)),
               px_across_med=float(np.median(e_n)), bg_along_med=float(np.median(al)), bg_across_med=float(np.median(ac)),
               bg_along_p10=float(np.percentile(al, PCT)), nfr=len(al),
               per_frame=[(int(fidx[j]), round(float(x), 1), round(float(y), 1), round(float(p), 1), round(float(q_), 1))
                          for j, x, y, p, q_ in zip(oke, rl, rs, al, ac)])
    return res


# video files of this checkout (tools/qp_common.load_split), not the video_path column of the geo table, which records
# where the files were when the table was made
sys.path.insert(0, R + "/tools")
from qp_common import load_split  # noqa: E402
_ls = load_split(split)
_vp = {}
for _v, _p in zip(_ls.video_id, _ls.video_path):
    _vp.setdefault(_v, _p); _vp.setdefault(str(_v).strip(), _p)
raw_vpath = {}
for i in cand:
    if i in raw.index and P[i]["video_id"] in _vp:
        raw_vpath[P[i]["video_id"]] = str(_vp[P[i]["video_id"]])

for i in cand:
    d = P[i]
    if i not in det.index:
        continue
    r = det.loc[i]
    src = str(r["answer_source"])
    pr = (d.get("priors") or [{}])[0]
    why0 = None
    if src.startswith("patch"):
        why0 = "already_patched"
    elif pr.get("kind") != "size":
        why0 = "prior_not_size"
    elif str(r["mode"]) != "2d_ratio" or not np.isfinite(r["g"]):
        why0 = "not_2d_ratio_geo"
    elif src not in ("agree_gm", "agree_wgm", "dis_geo"):
        why0 = f"source_{src}"
    if why0:
        diag[i] = why0; continue
    rw = raw.loc[i]
    mpp = float(rw["meas_mpp"])
    tv = str(rw["meas_tracks"]) if str(rw["meas_tracks"]) != "nan" else "v2sc"
    phrase = d["prior_obj"]["phrase"]
    rl_ = ruler(d["video_id"], tv, phrase)
    if not rl_["ok"]:
        diag[i] = rl_["why"]; continue
    px_old = pr["value_si"] / mpp
    k = px_old / rl_["px_new"]
    g, c = float(r["g"]), float(r["cs"])
    g2 = g * k
    if np.isfinite(c) and c > 0 and abs(np.log10(g2 / c)) <= np.log10(2):
        if src == "agree_wgm" and np.isfinite(r["answer"]):       # keep the row's own geo weight
            w = np.log(r["answer"] / c) / np.log(g / c) if abs(np.log(g / c)) > 1e-6 else 0.5
        else:
            w = 0.5
        v, how = g2 ** w * c ** (1 - w), f"agree w{w:.2f}"
    else:
        v, how = g2, "dis_geo"
    out[str(i)] = dict(value=float(v), why=f"rul423: prior side along motion, ruler {px_old:.1f}->{rl_['px_new']:.1f}px "
                                           f"(k={k:.3f}); geo {g:.3g}->{g2:.3g}; {how} with CaW {c:.3g}")
    diag[i] = dict(old=float(r["answer"]), new=float(v), g=g, g2=g2, c=c, src=src, k=k, px_old=px_old)

for key, rl_ in cache.items():
    print("ruler", key, {k_: v_ for k_, v_ in rl_.items() if k_ != "per_frame"})
    for pf in rl_.get("per_frame", []):
        print("   frame %d rr_long %.1f rr_short %.1f bg_along %.1f bg_across %.1f" % pf)
for i, dg in diag.items():
    print(i, dg)
fn = f"{R}/cache/tmp/ag_slopeside_patch_{split}.json"
json.dump(out, open(fn, "w"), indent=1)
print(f"[{split}] wrote {len(out)} rows -> {fn}")
