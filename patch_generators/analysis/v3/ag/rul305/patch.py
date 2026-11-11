#!/usr/bin/env python
"""ag rul305: patch the scene-ruler rows of basketball-key videos whose key failed the regulation check.

Input: cache/tmp/ag_keyruler_keys_<split>.json (key_check.py: old scene_ruler px, re-derived px, changed flag).
Scene-ruler geo is linear in the ruler (measure_v3: mpp = prior_m / (frac_w * W)), so a row measured with the scene ruler
(geo raw merge_src 'sruler', answered 'sruler_geo') becomes g * old_px / new_px. The final answer is recomputed as the
pipeline does for these rows: sruler_geo answers geo alone; the later bound rules of the recipe are re-checked
(phys_bounds_final class bound; disp_bound is scale-invariant because the row's speeds scale by the same factor).
Rows already patched (answer_source patch*) and rows of the rul424 patch are skipped.
Backstop diagnostic: tracked person box width (short side) in metres under the old and the new ruler (> 1.2 m = wrong).
usage: tools/run_limited.sh --io python analysis/v3/ag/rul305/patch.py
writes cache/tmp/ag_keyruler_patch_test.json, cache/tmp/ag_keyruler_patch_val.json"""
import sys, os, json, math
import numpy as np, pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for p in ("cache/tmp", "caw", "analysis/v3", "tools", "geo"):
    sys.path.insert(0, f"{R}/{p}")
SUB = "sub_xt_Nzc"
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
TRK = "_v2sc"
STALE = f"{R}/cache/tmp/ag_stale2_patch_test.json"


def person_width_m(split, vid, key_px, W_full, prior_m=4.88):
    fn = f"{R}/geo/out/tracks_{split}{TRK}/{vid}.json"
    if not os.path.exists(fn):
        return None
    T = json.load(open(fn))
    s = W_full / float(T["W"])                           # tracks are on the 480p frames (T["W"] wide)
    out = {}
    names = {}
    for qk in ("queries", "queries_d"):
        for q_ in (T.get(qk) or []):
            if isinstance(q_, dict) and q_.get("obj") is not None:
                names.setdefault(str(q_["obj"]), q_.get("key"))
    for oid, lst in (T.get("objs") or {}).items():
        ws = [min(b["st"]["box"][2] - b["st"]["box"][0], b["st"]["box"][3] - b["st"]["box"][1]) for b in lst if b and b.get("st")]
        if ws:
            out[names.get(str(oid), oid)] = round(float(np.median(ws)) * s * prior_m / key_px, 3)
    return out


def main():
    import analyze_caw as A
    import kinematic as K_
    K_.EXTRA_BOUNDS = K_.EXTRA2 = K_.EXTRA3 = True
    stale = set(json.load(open(STALE))) if os.path.exists(STALE) else set()
    res = {}
    for split in ("test", "val"):
        kfn = f"{R}/cache/tmp/ag_keyruler_keys_{split}.json"
        keys = json.load(open(kfn)) if os.path.exists(kfn) else {}
        det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
        raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
        df = A.load_test() if split == "test" else A.load_val()
        patch = {}
        for vid, k in keys.items():
            print(f"[{split}] {vid}: old px {k['old_px']} pass={k['old_pass']} new px {k.get('new_px')} use {k.get('use_px')} changed={k['changed']}")
            pw_old = person_width_m(split, vid, k["old_px"], k["W"])
            pw_new = person_width_m(split, vid, k["use_px"], k["W"]) if k.get("use_px") else None
            print(f"   person width m (backstop > 1.2): old ruler {pw_old} | checked ruler {pw_new}")
            if not k["changed"]:
                continue
            fac = k["old_px"] / k["use_px"]
            rows = det.index[det.video_id.astype(str) == vid]
            for i in rows:
                src = str(det.answer_source[i])
                if src.startswith("patch") or str(i) in stale:
                    print(f"   skip {i}: {src} / rul424"); continue
                if src != "sruler_geo" or str(raw.merge_src.get(i)) != "sruler":
                    print(f"   not a scene-ruler answer {i}: {src} (merge_src {raw.merge_src.get(i)})"); continue
                g = float(det.g[i]); new = g * fac
                q = str(det.question[i]); pri = str(df.loc[i, "ground_truth_prior"]) if i in df.index else ""
                b_ = K_.class_bound(q, pri, det.kind[i])
                if b_ is not None:
                    f_ = K_.q_si(q)
                    if not (b_[0] <= new * f_ <= b_[1]):
                        cs = float(det.cs[i]) if pd.notna(det.cs[i]) else np.nan
                        print(f"   {i}: new {new:.4g} outside class bound {b_}; cs {cs}")
                        if np.isfinite(cs) and b_[0] <= cs * f_ <= b_[1]:
                            new = cs
                patch[str(int(i))] = {"value": float(new), "why": f"rul305: basketball key failed the FT-circle/no-charge-arc check "
                                      f"({k['old_px']:.0f}px); re-derived key {k['use_px']:.0f}px (FT r/k {k.get('ft_ratio')}, NCA r/k "
                                      f"{k.get('nc_ratio')}): sruler geo {g:.4g} x {fac:.4f}"}
                print(f"   {i} {det.cat[i]} {det.kind[i]} | {q} | old {det.answer[i]:.4g} ({src}) -> new {new:.4g} | cs {det.cs[i]}")
        json.dump(patch, open(f"{R}/cache/tmp/ag_keyruler_patch_{split}.json", "w"), indent=1)
        res[split] = patch
        print(f"[{split}] patched rows: {len(patch)}", dict(det.loc[[int(x) for x in patch], "cat"].value_counts()) if patch else {})
    # evaluation on labelled rows (val + test copies of val questions)
    from lab_rows import labelled
    L = labelled(SUB)
    ids = {("val", int(x)) for x in res.get("val", {})} | {("test", int(x)) for x in res.get("test", {})}
    hit = [(s_, i) for s_, i in ids if ((L.split == s_) & (L.index == i)).any()]
    print("labelled rows changed:", len(hit))
    vids = set()
    for s_ in ("test", "val"):
        kfn = f"{R}/cache/tmp/ag_keyruler_keys_{s_}.json"
        if os.path.exists(kfn):
            vids |= {v for v, k in json.load(open(kfn)).items()}
    print("labelled rows on basketball-key videos at all:", int(L.video_id.astype(str).isin(vids).sum()))
    if hit:
        for s_, i in hit:
            r = L[(L.split == s_) & (L.index == i)].iloc[0]
            nv = res[s_][str(i)]["value"]
            print(s_, i, r.y, r.answer, nv, A.row_mra(np.array([r.answer]), np.array([r.y]))[0], A.row_mra(np.array([nv]), np.array([r.y]))[0])


if __name__ == "__main__":
    main()
