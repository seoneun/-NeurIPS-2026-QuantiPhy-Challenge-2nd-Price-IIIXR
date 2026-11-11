"""rul315: gravity-prior rows that geo left to CaW because no free-fall segment was found.

Scope: answer_source caw_nogeo (after a 'mo_guard:' prefix), geo raw meas_why prior_no_freefall / knownsize_no_freefall,
a gravity prior (9.0 <= g <= 10.5 m/s^2), not already a patch.

Physics: a body in free flight follows y(t) = y0 + vy0 t + a/2 t^2 in the image (y down); with the depth constant (planar
motion, the 2D category) a = g / mpp whatever the horizontal motion, so the vertical curvature alone sets the metres-per-pixel
scale, and the 2D ratio gives every other length in the plane. The pipeline's freefall() needs a changepoint piece
>= 0.15 s with <= 1.5 px residual: it misses short clean drops (a 60 fps droplet whose 0.1 s fall is split by a duplicated
frame) and drops in re-encoded clips with a dropped-frame cadence (internet_0029, 50 fps with every 5th step doubled).
fitlib.fall_scale fits a robust parabola (3-MAD clipping, <= 25 % outliers) to the box bottom, box top and centroid over
the longest clean window (R^2 >= 0.995, curvature t >= 8, travel >= 1.5 sizes and >= 30 px, positive second differences
>= 60 %, lateral acceleration <= 0.35 a, consistent halves, object size stable p90/p10 <= 1.35) and needs >= 2 signals
within 25 % (or one with R^2 >= 0.999 and >= 8 points). It reproduces the pipeline's scale where both work (58 tracks,
median ratio 1.000).

Rule conditions (all code on inputs):
  - 2D video (video_type[1] == '2'): the 2D ratio needs the target in the falling body's plane. 3D rows are skipped: the
    falling body has no depth entry and its depth changes (captured_0020 toy grows 24 -> 60 px in flight), or the motion
    is a pendulum (captured_0010), or the candidate signals disagree (simulation_0032_segmented).
  - falling body = the named prior object, else every moving track (track_ok dynamic); the longest clean window wins.
  - csv fps == track json fps (never the container fps); gravity_scale_check (frame height, class size ranges) passes.
  - target: size rows use ext_px/agg_ext like measure_v3, except 'horizontal ...' / 'vertical ...' questions, which take
    the axis-aligned box width / height (the rotated-rect long side of a vertical drop or vertical text is not its
    horizontal extent); motion rows use px_quantity.
Final answer: the pipeline fusion with CaW: within 2x -> geometric mean (size rows g^0.65 c^0.35, the build's --agree_w
size:0.65:strong); outside -> geo (gravity 2d_ratio is a strong mode). No lab copies are involved (2D only).

usage: tools/run_limited.sh --io python analysis/v3/ag/rul315/rul315.py [SUB]     (default sub_xt_Nzi)
 -> cache/tmp/ag_ngfall_patch_{test,val}.json, cache/tmp/ag_ngfall_diag_{split}.csv
"""
import json, os, sys, re
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for p_ in ("geo", "analysis/v3", "caw", "cache/tmp", "analysis/v3/ag/rul315"):
    sys.path.insert(0, f"{R}/{p_}")
import measure_v3 as MV
import fitlib as F

MV.FIX = set("abcdef")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzi"
RAWTAG = "geo_v4ntrj5sQ2srngsr_d_vw"
GROUPS = ("prior_no_freefall", "knownsize_no_freefall")
LEN_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "ft": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}
SPD_UNIT = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6}
ACC_UNIT = {"m/s^2": 1.0, "cm/s^2": 0.01}


def num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def grp_of(why):
    return str(why).split(":")[0].split("(")[0]


def load_vj(split, v):
    for sfx in ("_v2sc", "_v2", ""):
        p = f"{R}/geo/out/tracks_{split}{sfx}/{v}.json"
        if os.path.exists(p):
            return json.load(open(p)), sfx
    return None, None


def unit_si(kind, unit):
    if kind in ("size", "dist2", "disp", "path", "maxheight", "orbitdiam", "hfloor"):
        return LEN_UNIT.get(unit)
    if kind in ("speed", "avgspeed", "hvel"):
        return SPD_UNIT.get(unit)
    if kind == "accel":
        return ACC_UNIT.get(unit)
    return None


def target_px(tgt, r):
    q = r["q"]
    kind = q["kind"]
    if kind == "size":
        js = tgt.oke
        if len(js) == 0:
            return None, "no_extent"
        box = np.array([tgt.st[j]["box"] for j in js], float)
        if q.get("horizontal") and not q.get("vertical"):
            return float(np.median(box[:, 2] - box[:, 0])), "box_width"
        if q.get("vertical") and not q.get("horizontal"):
            return float(np.median(box[:, 3] - box[:, 1])), "box_height"
        e, _ = tgt.ext_px(q["dim"], r["target"]["phrase"])
        return float(MV.agg_ext(np.asarray(e, float), q["dim"])), f"ext_{q['dim']}"
    if kind in ("speed", "avgspeed", "hvel", "accel", "disp", "path", "maxheight"):
        v = MV.px_quantity(tgt, kind, q.get("t"), q.get("t1"), q.get("t2"), q.get("when"))
        return (None if v is None else float(v)), f"px_{kind}"
    return None, f"kind_{kind}"


def measure_row(r, vj, V, fps_csv):
    """-> (value in the question's unit, info dict) or (None, info)"""
    info = {}
    pr = r["priors"][0]
    g = float(pr["value_si"])
    fps = float(vj["fps"])
    if np.isfinite(fps_csv) and abs(fps_csv - fps) > 1e-3:
        return None, dict(why=f"fps_mismatch({fps_csv} vs {fps})")
    q = r["q"]
    us = unit_si(q["kind"], q.get("unit"))
    if us is None:
        return None, dict(why=f"unit({q['kind']},{q.get('unit')})")
    # ---- the falling body
    prior_tr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
    cands = [prior_tr] if (prior_tr is not None and prior_tr.n) else \
        list({id(t_): t_ for t_ in V.tr.values() if t_ is not None and t_.n and MV.track_ok(t_, True)[0]}.values())
    best = None
    for trk in cands:
        a, inf = F.fall_scale(trk, vj["H"], fps)
        if a is None:
            info.setdefault("fails", []).append(str(inf)[:160])
            continue
        if best is None or inf["dur"] > best[2]["dur"]:
            best = (trk, a, inf)
    if best is None:
        info["why"] = "no_clean_fall"
        return None, info
    ftr, a, inf = best
    mpp = g / a
    info.update(fall_obj=ftr.q.get("phrase"), a_px=round(a, 1), mpp=mpp, fit=str(inf)[:200])
    okg, wg = MV.gravity_scale_check(mpp, ftr, V)
    if not okg:
        info["why"] = wg
        return None, info
    # ---- the target
    tgt = V.get(r["target"])
    if tgt is None or not tgt.n:
        info["why"] = "no_target_track"
        return None, info
    okt, wt = MV.track_ok(tgt, q["kind"] != "size")
    if not okt:
        info["why"] = "target_" + wt
        return None, info
    v_px, how = target_px(tgt, r)
    if v_px is None or not np.isfinite(v_px) or v_px <= 0:
        info["why"] = "target_px_fail:" + how
        return None, info
    if q["kind"] == "accel":
        val_si = v_px * mpp
    else:
        val_si = v_px * mpp
    info.update(v_px=round(v_px, 2), how=how, why="ok")
    return val_si / us, info


def fuse(gv, cs, kind):
    """the build's fusion for a strong-mode (2d_ratio) geo: within 2x -> gm, size rows g^0.65 c^0.35 (submit_v3
    --agree_w size:0.65:strong, as in sub_xt_N*); outside 2x -> geo"""
    if not (np.isfinite(cs) and cs > 0):
        return gv, "geo_nocaw"
    r = gv / cs
    if 0.5 <= r <= 2.0:
        if kind == "size":
            return float(gv ** 0.65 * cs ** 0.35), "agree_wgm"
        return float(np.sqrt(gv * cs)), "agree_gm"
    return gv, "dis_geo(strong 2d_ratio)"


def run(split, scope=True, ids=None):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWTAG}_raw.csv")
    if "Unnamed: 0" in raw.columns:
        raw = raw.set_index("Unnamed: 0")
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    vids, rows, patch = {}, [], {}
    for r in recs:
        i = r["id"]
        if i not in det.index or i not in raw.index:
            continue
        if ids is not None and i not in ids:
            continue
        pr = (r.get("priors") or [None])[0]
        if not pr:
            continue
        gv = num(pr.get("value_si"))
        if not ((pr.get("gravity") or pr.get("kind") == "accel") and 9.0 <= gv <= 10.5):
            continue
        D = det.loc[i]
        src = re.sub(r"^mo_guard:", "", str(D.answer_source))
        grp = grp_of(raw.loc[i, "meas_why"])
        if scope and not (src == "caw_nogeo" and grp in GROUPS):
            continue
        rec = dict(id=i, vid=r["video_id"], cat=D["cat"], vtype=r.get("video_type"), q=r["question"].strip()[:70], grp=grp, src=src,
                   ans=num(D.answer), g=num(D.g), cs=num(D.cs), y=num(D.get("y", np.nan)))
        if src.startswith("patch"):
            rec["why"] = "skip_patched"; rows.append(rec); continue
        if str(r.get("video_type", "  "))[1:2] != "2":
            rec["why"] = "skip_3d"; rows.append(rec); continue
        if r["video_id"] not in vids:
            vj, sfx = load_vj(split, r["video_id"])
            vids[r["video_id"]] = (vj, None if vj is None else MV.Video(vj, "e"))
        vj, V = vids[r["video_id"]]
        if vj is None:
            rec["why"] = "no_tracks"; rows.append(rec); continue
        val, info = measure_row(r, vj, V, num(raw.loc[i, "fps"]))
        rec.update({k: info.get(k) for k in ("why", "fall_obj", "a_px", "mpp", "v_px", "how", "fit")})
        if val is not None:
            fin, how = fuse(val, rec["cs"], r["q"]["kind"])
            rec.update(new_g=val, new=fin, fuse=how)
            if scope and abs(fin / rec["ans"] - 1) > 0.005:
                patch[str(int(i))] = dict(value=float(fin), why=f"rul315: free fall of '{info['fall_obj']}' a={info['a_px']:.0f} px/s^2 "
                                          f"-> mpp=g/a={info['mpp']:.3g} m/px; target {info['how']} {info['v_px']:.1f} px -> geo {val:.4g} "
                                          f"vs CaW {rec['cs']:.4g} -> {how} {fin:.4g} (was {rec['ans']:.4g} {src})")
        rows.append(rec)
    return pd.DataFrame(rows), patch


def eval_labels(patches):
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    out = []
    for split, P in patches.items():
        Ls = L[L.split == split]
        for i in [int(k) for k in P if int(k) in Ls.index]:
            y = float(Ls.loc[i, "y"]); new = float(P[str(i)]["value"])
            out.append(dict(split=split, id=i, cat=Ls.loc[i, "cat"], y=y, old=float(Ls.loc[i, "answer"]), new=new,
                            m_old=float(Ls.loc[i, "m"]), m_new=float(A.row_mra(np.array([new]), np.array([y]))[0])))
    print("labelled rows changed by the patch:", len(out))
    if out:
        T = pd.DataFrame(out); print(T.to_string(index=False)); print("sum dMRA", round(float((T.m_new - T.m_old).sum()), 4))
    # method check: the same measurement on the labelled gravity rows (as if their geo were missing), D2 only by the rule
    M = []
    for split in ("val", "test"):
        Ls = L[L.split == split]
        X, _ = run(split, scope=False, ids=set(Ls.index))
        if not len(X):
            continue
        for _, x in X.iterrows():
            if x.get("why") != "ok":
                M.append(dict(split=split, id=x.id, vid=x.vid, q=x.q[:45], why=x.get("why"))); continue
            y = float(Ls.loc[x.id, "y"])
            M.append(dict(split=split, id=x.id, vid=x.vid, q=x.q[:45], why="ok", y=y, cur=float(Ls.loc[x.id, "answer"]), new_geo=x.new_g, new=x.new,
                          m_cur=float(Ls.loc[x.id, "m"]), m_geo=float(A.row_mra(np.array([x.new_g]), np.array([y]))[0]),
                          m_new=float(A.row_mra(np.array([x.new]), np.array([y]))[0])))
    if M:
        Mx = pd.DataFrame(M)
        print("method check on labelled gravity rows (rule measurement regardless of scope):")
        print(Mx.round(4).to_string(index=False))
        ok = Mx[Mx.why == "ok"]
        if len(ok):
            print(f"  n={len(ok)} mean MRA current {ok.m_cur.mean():.3f} | rul315 geo {ok.m_geo.mean():.3f} | rul315 fused {ok.m_new.mean():.3f}")


if __name__ == "__main__":
    pd.set_option("display.width", 330); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 200)
    patches = {}
    for split in ("test", "val"):
        X, patch = run(split)
        patches[split] = patch
        if len(X):
            X.to_csv(f"{R}/cache/tmp/ag_ngfall_diag_{split}.csv", index=False)
        json.dump(patch, open(f"{R}/cache/tmp/ag_ngfall_patch_{split}.json", "w"), indent=1)
        print(f"===== {split}: {len(X)} rows in scope; patch rows {len(patch)}")
        if len(X):
            cols = [c for c in ("id", "vid", "cat", "q", "grp", "ans", "g", "cs", "why", "fall_obj", "a_px", "mpp", "v_px", "how", "new_g", "new", "fuse") if c in X]
            print(X[cols].to_string(index=False))
            if "new" in X:
                ch = X[X.id.astype(str).isin(patch.keys())]
                print("changed by category:", ch["cat"].value_counts().to_dict())
        for k, v in patch.items():
            print(k, round(v["value"], 6), v["why"])
    eval_labels(patches)
