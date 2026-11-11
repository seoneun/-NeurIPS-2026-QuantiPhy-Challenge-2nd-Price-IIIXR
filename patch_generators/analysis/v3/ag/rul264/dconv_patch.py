"""rul264 patch: "distance between A and B" = centre-to-centre distance unless the question asks for the minimum distance
explicitly or one of the two objects is a person (non-rigid; the one labelled close pair, lab ball-to-person 0.198 m,
is a surface gap). The v3 measurement switches to the mask-to-mask surface gap whenever gap < 0.5 x centre distance
(measure_v3.dist2_px, FIX f); this patch puts the centre distance back on those rows when neither exception holds.
Writes R/cache/tmp/ag_dconv_patch_{split}.json and prints the changed rows."""
import sys, os, json, re
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/geo"); sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw")
import numpy as np, pandas as pd
import measure_v3 as MV
import measure as M1
MV.FIX = set("df"); MV.TRACK_VER = "v2sc"
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 200)

# explicit minimum: the superlative modifies the distance itself ("minimum distance", "closest distance", "shortest gap",
# "closest point"); "the two closest tall trees" / "the nearest player" select a pair and are not a surface convention
EXPLICIT_MIN = re.compile(r"\b(?:minimum|minimal|smallest|shortest|closest|nearest)\s+(?:\w+\s+)?(?:distance|gap|separation|clearance|points?)\b|\bgap\b|\bclearance\b")
PERSON = re.compile(r"\b(?:person|people|persons|man|men|woman|women|boy|girl|child|children|kid|baby|player|players|human|pedestrian|"
                    r"passer|receiver|referee|athlete|runner|walker|skater|cyclist|rider|swimmer|dancer|foot|feet|hand|hands)\b")
STRONG = ("2d_ratio", "3d_calib")


def gap_centre(tr1, tr2, t, when):
    """the frames / medians measure_v3.dist2_px uses (2D pixels): returns (g_px, c_px) or None"""
    fr1, fr2 = tr1.q.get("frames"), tr2.q.get("frames")
    if not fr1 or not fr2:
        return None
    T = tr1.t_all
    both = [j for j in range(len(T)) if fr1[j].get("poly") and fr2[j].get("poly") and fr1[j].get("st") and fr2[j].get("st")]
    if not both:
        return None
    if t is not None or when is not None:
        if when == "start":
            t = max(tr1.t.min(), tr2.t.min())
        elif when == "end":
            t = min(tr1.t.max(), tr2.t.max())
        js = [min(both, key=lambda j: abs(T[j] - t))]
        if abs(T[js[0]] - t) > 0.25:
            return None
    else:
        js = both
    gs, cs = [], []
    for j in js:
        gs.append(MV._poly_dist(fr1[j]["poly"], fr2[j]["poly"])[0])
        cs.append(float(np.linalg.norm(np.array(fr1[j]["st"]["c"]) - np.array(fr2[j]["st"]["c"]))))
    return float(np.median(gs)), float(np.median(cs))


def run(split):
    os.environ.setdefault("QP_NEAREST", "1")
    byv, vjs = MV.load_split_tracks(split, "d")
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv").set_index("Unnamed: 0")
    det = pd.read_csv(f"{R}/submissions/sub_xt_Ngd_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    recs = {r["id"]: r for rs in byv.values() for r in rs}
    gap_ids = raw.index[(raw.meas_kind == "dist2") & raw.meas_gap.astype(str).str.lower().eq("true")]
    out, patch = [], {}
    for i in gap_ids:
        r, x = recs[i], raw.loc[i]
        ql = r["question"].lower()
        d = dict(id=i, video_id=r["video_id"], src=r["video_source"], cat=r["cat"], question=r["question"],
                 meas=float(x.meas_value), geo=float(x.geo_value) if pd.notna(x.geo_value) else np.nan, merge=x.merge_src, mode=x.meas_mode,
                 mpp=float(x.meas_mpp) if pd.notna(x.meas_mpp) else np.nan,
                 ans=float(det.answer.get(i, np.nan)), src_ans=det.answer_source.get(i, ""), cs=float(det.cs.get(i, np.nan)) if pd.notna(det.cs.get(i, np.nan)) else np.nan,
                 exmin=bool(EXPLICIT_MIN.search(ql)), person=bool(PERSON.search(ql)))
        V = MV.Video(vjs[r["video_id"]], "d")
        a = V.get(r["target"]); b = V.get(r["target2"]) if r.get("target2") else None
        q = r["q"]
        cands = [b2 for b2 in MV.nearest_cands(V, r, b) if b2 is not None] if b is not None else []
        best = None
        for b2 in (cands if a is not None else []):   # 10-17 clean6: target track not found -> no candidate (was AttributeError)
            gc = gap_centre(a, b2, q["t"], q["when"])
            vpx = M1.dist2_px(a, b2, q["t"], q["when"])
            if gc is None or vpx is None or gc[0] <= 0:
                continue
            val = vpx * gc[0] / gc[1] * (d["mpp"] if np.isfinite(d["mpp"]) else np.nan)
            if best is None or val < best[0]:
                best = (val, gc[0], gc[1], vpx)
        if best is not None:
            d.update(repro=best[0], g_px=best[1], c_px=best[2], ratio=best[2] / best[1], centre=d["meas"] * best[2] / best[1])
        out.append(d)
    D = pd.DataFrame(out)
    if len(D) == 0:
        print(split, "no gap rows"); return D, {}
    D["repro_ok"] = (np.abs(np.log(D.repro / D.meas)) < 0.10) if "repro" in D else False
    # geo-derived answer: the answer equals the gap value (or a variant copy of a geo gap answer)
    D["ans_is_gap"] = np.abs(np.log(D.ans / D.meas)) < 0.03
    for _, d in D.iterrows():
        if d.exmin or d.person or not d.repro_ok or not d.ans_is_gap or d.merge not in ("d",) or not (d.ratio >= 1.5):
            continue
        cen = float(d.centre)
        cs = d.cs
        if np.isfinite(cs) and cs > 0 and max(cen / cs, cs / cen) <= 2:
            new, how = cen, "agree"                         # dist2 agreement rows take geo alone (agree_geo_kind)
        elif str(d["mode"]).startswith(STRONG):
            new, how = cen, "strong"
        else:
            new, how = cs, "caw"
        patch[int(d.id)] = dict(value=float(new), why=f"rul264 centre {cen:.3g} (gap {d.meas:.3g}, c/g {d.ratio:.2f}, caw {cs:.3g}, {how})")
    # variant copies (var_ans_plain etc.) of a patched plain row: the same convention on the variant's own measurement
    json.dump({str(k): v for k, v in patch.items()}, open(f"{R}/cache/tmp/ag_dconv_patch_{split}.json", "w"), indent=1)
    print(split, "gap rows", len(D), "patched", len(patch))
    print(D[[c for c in ["id", "video_id", "cat", "question", "ans", "src_ans", "cs", "meas", "geo", "merge", "mode", "repro", "ratio", "centre", "exmin", "person", "repro_ok", "ans_is_gap"] if c in D]].round(3).to_string())   # 10-17 clean6: print only; no row with repro -> no repro / ratio / centre columns
    for k, v in patch.items():
        print(k, D.set_index("id").loc[k, "video_id"], "|", v)
    return D, patch




def summary():
    """label-free check on the changed test rows: agreement with CaW (within 2x) before / after, by category"""
    det = pd.read_csv(f"{R}/submissions/sub_xt_Ngd_detail.csv").set_index("Unnamed: 0")
    P = json.load(open(f"{R}/cache/tmp/ag_dconv_patch_test.json"))
    rows = []
    for k, v in P.items():
        i = int(k); old = float(det.answer[i]); cs = float(det.cs[i]) if pd.notna(det.cs[i]) else np.nan
        rows.append(dict(id=i, cat=det.cat[i], old=old, new=v["value"], cs=cs,
                         agree_old=bool(max(old / cs, cs / old) <= 2) if cs > 0 else np.nan,
                         agree_new=bool(max(v["value"] / cs, cs / v["value"]) <= 2) if cs > 0 else np.nan))
    S = pd.DataFrame(rows)
    if not len(S):   # 10-17 clean6: empty test patch (print only; was AttributeError S.cat after the patches were written)
        print("changed test rows by cat {}"); return
    print("changed test rows by cat", S.cat.value_counts().to_dict())
    print("agreement with CaW within 2x: before", S.agree_old.mean().round(3), "after", S.agree_new.mean().round(3))
    print("median |log(ans/CaW)| before", np.median(np.abs(np.log(S.old / S.cs))).round(3), "after", np.median(np.abs(np.log(S.new / S.cs))).round(3))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["val", "test"]:
        run(s)
    summary()
