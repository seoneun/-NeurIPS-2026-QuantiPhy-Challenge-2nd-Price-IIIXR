"""QP second-instance rule outside simulation (rule 'second').
A two-instance distance question ("the distance between the two bottles / people / umbrellas on the left / closest
tall trees") is parsed as target + target2(#second). Outside simulation the pipeline's QP_SECOND_FIX does not run and
three failure modes remain:
  A collapse : target2 resolved to the same SAM2 object as target (or to no / a near-identical object) -> no geo.
  B qualifier: target2 dropped the target's positional qualifier (on the left) and took an instance that violates it.
  C closest  : "the two closest X" sets the surface-gap (minimum) flag although closest selects the pair.
The second instance is taken from the video's other tracked objects whose query phrase has the same head noun as the
target (bottle / person / umbrella ...), that are a different physical object (median centroid distance > max(20 px,
0.5 x size)), of comparable size (area ratio 0.25-4), cover >= 50 % of the frames and satisfy the target's positional
qualifiers. Exactly one such physical object must remain (near-identical duplicates from the two query configs count
once); otherwise the row is left alone. Distance = centroid distance (median over frames, or at the asked time)
 x the row's scale: 2D -> metres per px of the row (raw meas_mpp, else the video's 2D mpp under the same prior);
3D -> back-projection with each object's own camera distance (near/far keys assigned by apparent size) and the
video's focal f/W from the raw csv."""
import sys, json, re, os, math
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/cache/tmp")
import track_common as tc
RAWN = os.environ.get("AG_RAW", "geo_v4ntrj5sQ2srngsr_d_vw")
TRK = os.environ.get("AG_TRACKS", "_v2sc")
UNIT = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "km": 0.001}
POS = {"left": (0, -1), "right": (0, 1), "top": (1, -1), "upper": (1, -1), "bottom": (1, 1), "lower": (1, 1)}
IRR = {"people": "person", "men": "man", "women": "woman", "children": "child"}
SUB = os.environ.get("AG_SUB", "sub_xt_Nu")
# tall / thin objects: their separation is between their axes (trunk, post), never between crowns or lamp heads
TALL = {"tree", "palm", "pole", "post", "streetlight", "lamppost", "pillar", "column", "flagpole", "tower", "mast", "lamp"}


def head(p):
    s = (p or "").lower().strip()
    w = re.findall(r"[a-z]+", s)
    if w and len(w) == 1 and w[0] in IRR:
        return IRR[w[0]]
    h = tc.head_noun(s)
    return IRR.get(h, h)


def key(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def ostats(vj, o):
    fr = vj["objs"][str(o)]
    ok = [j for j, f in enumerate(fr) if f and f.get("st")]
    if not ok:
        return None
    C = np.array([fr[j]["st"]["c"] for j in ok], float)
    A = np.array([fr[j]["st"]["area"] for j in ok], float)
    return dict(ok=ok, n=len(fr), c=np.median(C, 0), area=float(np.median(A)), size=float(math.sqrt(np.median(A))))


def same_phys(vj, a, b):
    """near-identical tracks: median centroid distance < max(20 px, 0.5 x size)"""
    if a == b:
        return True
    A, B = vj["objs"][str(a)], vj["objs"][str(b)]
    ds = [np.hypot(*(np.array(x["st"]["c"]) - np.array(y["st"]["c"]))) for x, y in zip(A, B) if x and y and x.get("st") and y.get("st")]
    if len(ds) < 3:
        return False
    sa, sb = ostats(vj, a), ostats(vj, b)
    return float(np.median(ds)) < max(20.0, 0.5 * min(sa["size"], sb["size"]))


def pos_ok(st, where, W, H):
    for w in where or []:
        if w in POS:
            ax, sg = POS[w]
            mid = (W if ax == 0 else H) / 2.0
            if sg * (st["c"][ax] - mid) <= 0:
                return False
    return True


def frames_at(vj, o1, o2, q):
    T = np.array(vj["fidx"], float) / float(vj["fps"])        # track json fps (dataset rate), never the container's
    A, B = vj["objs"][str(o1)], vj["objs"][str(o2)]
    both = [j for j in range(len(T)) if A[j] and B[j] and A[j].get("st") and B[j].get("st")]
    if not both:
        return [], T
    t, when = q.get("t"), q.get("when")
    if when == "start":
        return [both[0]], T
    if when == "end":
        return [both[-1]], T
    if t is not None:
        j = min(both, key=lambda j: abs(T[j] - t))
        return ([j] if abs(T[j] - t) <= 0.25 else []), T
    return both, T


def gap_ratio(vj, o1, o2n, js_all, q, proj):
    """the pipeline's fix-f convention (measure_v3.dist2_px): median mask-to-mask surface gap / median centroid distance
    over the frames with both polygons (the asked frame only when a time is asked); proj maps (point, which) -> coords"""
    import measure_v3 as MV
    A, B = vj["objs"][str(o1)], vj["objs"][str(o2n)]
    T = np.array(vj["fidx"], float) / float(vj["fps"])
    both = [j for j in range(len(T)) if A[j] and B[j] and A[j].get("poly") and B[j].get("poly") and A[j].get("st") and B[j].get("st")]
    if not both:
        return None
    t, when = q.get("t"), q.get("when")
    if t is not None or when is not None:
        js = [j for j in js_all if j in both]
        if not js:
            j0 = js_all[0]
            jb = min(both, key=lambda j: abs(T[j] - T[j0]))
            if abs(T[jb] - T[j0]) > 0.25:
                return None
            js = [jb]
    else:
        js = both
    gaps, cens = [], []
    for j in js:
        g, pa, pb = MV._poly_dist(A[j]["poly"], B[j]["poly"])
        c1, c2 = proj(A[j]["st"]["c"], 1), proj(B[j]["st"]["c"], 2)
        cens.append(float(np.linalg.norm(c1 - c2)))
        gaps.append(float(np.linalg.norm(proj(pa, 1) - proj(pb, 2))) if pa is not None else 0.0)
    g, c = float(np.median(gaps)), float(np.median(cens))
    return g / c if c > 0 else None


def measure(d, vj, RAW, o1, o2n, axis=False):
    """distance between objects o1 and o2n in the question's unit, or (None, reason). Centroid distance, switched to the
    surface gap when the objects are close (gap < 0.5 x centroid distance; the pipeline's fix f) unless axis=True"""
    vid = str(d["video_id"])
    rw = RAW.loc[d["id"]]
    W, H = float(vj["W"]), float(vj["H"])
    js, T = frames_at(vj, o1, o2n, d["q"])
    if not js:
        return None, "no common frame at t", {}
    A, B = vj["objs"][str(o1)], vj["objs"][str(o2n)]
    unit = UNIT.get(d["q"].get("unit") or "m", 1.0)
    is3d = d["video_type"][1] == "3"
    if not is3d:
        mpp, src = rw.meas_mpp, "row"
        if not np.isfinite(mpp):
            vv = RAW[(RAW.video_id == vid) & (RAW.ground_truth_prior == rw.ground_truth_prior) & RAW.meas_mode.astype(str).str.startswith("2d")]
            mpp = float(np.nanmedian(vv.meas_mpp)) if vv.meas_mpp.notna().any() else np.nan
            src = "video"
        if not np.isfinite(mpp):
            return None, "no 2D scale", {}
        px = float(np.median([np.hypot(*(np.array(A[j]["st"]["c"]) - np.array(B[j]["st"]["c"]))) for j in js]))
        cen = px * mpp * unit
        gr = gap_ratio(vj, o1, o2n, js, d["q"], lambda u, k: np.asarray(u, float))
        ex = dict(px=round(px, 1), mpp=mpp, mpp_src=src, cen=cen, gr=gr)
        if gr is not None and 0 < gr < 0.5 and not axis:
            return cen * gr, f"surface gap {px * gr:.0f}px (centroid {px:.0f}px, gap<0.5 centroid) x mpp {mpp:.5f} ({src})", ex
        return cen, f"centroid {px:.0f}px x mpp {mpp:.5f} ({src})" + (" [axis]" if axis else ""), ex
    dk = list(d.get("target_dkeys") or [])
    ents = {k: [e for e in d["depth"] if e["key"] == k] for k in dk}
    if not dk or any(len(v) != 1 for v in ents.values()) or len(dk) > 2:
        return None, f"3D depth keys {dk} unsupported", {}
    s1, s2 = ostats(vj, o1), ostats(vj, o2n)
    if len(dk) == 2:
        ks = sorted(dk, key=lambda k: ents[k][0]["d"])          # near first
        near1 = s1["size"] >= s2["size"]                          # the bigger (apparent) object is the nearer one
        z1, z2 = (ents[ks[0]][0]["d"], ents[ks[1]][0]["d"]) if near1 else (ents[ks[1]][0]["d"], ents[ks[0]][0]["d"])
    else:
        z1 = z2 = ents[dk[0]][0]["d"]
    fw = RAW[(RAW.video_id == vid)].meas_f
    fw = float(np.nanmedian(fw)) if fw.notna().any() else np.nan
    if not np.isfinite(fw):
        return None, "no focal", {}
    f = fw * W; cx, cy = W / 2.0, H / 2.0
    def bp(u, z):
        r_ = np.array([(u[0] - cx) / f, (u[1] - cy) / f, 1.0]); return r_ / np.linalg.norm(r_) * z
    dd = [float(np.linalg.norm(bp(A[j]["st"]["c"], z1) - bp(B[j]["st"]["c"], z2))) for j in js]
    cen = float(np.median(dd)) * unit
    gr = gap_ratio(vj, o1, o2n, js, d["q"], lambda u, k: bp(u, z1 if k == 1 else z2))
    ex = dict(z=(z1, z2), fw=fw, cen=cen, gr=gr)
    if gr is not None and 0 < gr < 0.5 and not axis:
        return cen * gr, f"3D surface gap (gap/centroid {gr:.2f}) z={z1:.2f}/{z2:.2f} m f/W={fw:.3f}", ex
    return cen, f"3D centroids z={z1:.2f}/{z2:.2f} m f/W={fw:.3f} (gap/centroid {gr if gr is None else round(gr, 2)})", ex


def main(split):
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    RAW = pd.read_csv(f"{R}/preds/{split}_{RAWN}_raw.csv").set_index("Unnamed: 0")
    fn_ = f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv"
    DET = pd.read_csv(fn_).set_index("Unnamed: 0") if os.path.exists(fn_) else None
    out, diag = {}, []
    for d in P:
        vid = str(d["video_id"])
        if vid.startswith("simulation") or d["q"].get("kind") != "dist2":
            continue
        t2 = d.get("target2") or {}
        if t2.get("where") != ["#second"]:
            continue
        tf = f"{R}/geo/out/tracks_{split}{TRK}/{vid}.json"
        if not os.path.exists(tf):
            continue
        vj = json.load(open(tf))
        W, H = float(vj["W"]), float(vj["H"])
        qm = {q["key"]: q.get("obj") for q in vj["queries"]}
        o1, o2 = qm.get(key(d["target"])), qm.get(key(t2))
        i = d["id"]
        rec = dict(id=i, vid=vid, cat=d["cat"], q=d["question"].strip()[:70], o1=o1, o2=o2, case="", o2n=None, val=np.nan, why="")
        if o1 is None:
            rec["why"] = "no target track"; diag.append(rec); continue
        s1 = ostats(vj, o1)
        if s1 is None or len(s1["ok"]) < 0.5 * s1["n"]:
            rec["why"] = "target track short"; diag.append(rec); continue
        twhere = list(d["target"].get("where") or [])
        hd = head(d["target"]["phrase"])
        case = ""
        axis = hd in TALL
        if o2 is None or same_phys(vj, o1, o2):
            case = "A_collapse"
        elif not pos_ok(ostats(vj, o2), twhere, W, H):
            case = "B_qualifier"
        elif axis and str(RAW.loc[i].meas_gap) == "True":
            case = "C_tall_axis"        # two trees / poles: the surface gap between crowns is not their separation
        rec["case"] = case
        if case in ("A_collapse", "B_qualifier"):
            cands = set()
            for cfg in ("queries", "queries_d"):
                for q in vj.get(cfg, []):
                    if q.get("obj") is not None and head(q["phrase"]) == hd:
                        cands.add(int(q["obj"]))
            good = []
            for c in sorted(cands):
                sc = ostats(vj, c)
                if sc is None or same_phys(vj, o1, c) or len(sc["ok"]) < 0.5 * sc["n"]:
                    continue
                if not (0.25 <= sc["area"] / max(s1["area"], 1.0) <= 4.0):
                    continue
                if not pos_ok(sc, twhere, W, H):
                    continue
                if any(same_phys(vj, c, g) for g in good):
                    continue
                good.append(c)
            rec["cands"] = good
            if len(good) != 1:
                rec["why"] = f"{len(good)} distinct candidates"; diag.append(rec); continue
            o2n = good[0]
        else:
            o2n = o2
        val, why, ex = measure(d, vj, RAW, o1, o2n, axis=axis)
        rec.update(o2n=o2n, **ex)
        if val is None:
            rec["why"] = why; diag.append(rec); continue
        rec["val"] = val
        rec["why"] = why
        old = float(DET.answer.get(i, np.nan)) if DET is not None else np.nan
        rec["old"] = old
        diag.append(rec)
        # patch only a real change (> 10 %) of the current final answer
        if case and not (np.isfinite(old) and old > 0 and abs(math.log(val / old)) < math.log(1.1)):
            out[str(i)] = {"value": float(val), "why": f"second_out {case}: obj {o1}/{o2n} {why}"}
    return out, diag


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70)
    for sp in ("test", "val"):
        out, diag = main(sp)
        json.dump(out, open(f"{R}/cache/tmp/ag_second_patch_{sp}.json", "w"), indent=1)
        D = pd.DataFrame(diag)
        RAW = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv").set_index("Unnamed: 0")
        D["geo_raw"] = D.id.map(RAW.geo_value)
        print(f"==== {sp}: {len(out)} rows patched")
        print(D.drop(columns=["q"]).to_string())
