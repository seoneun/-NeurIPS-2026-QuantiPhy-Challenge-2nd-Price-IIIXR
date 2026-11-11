#!/usr/bin/env python
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/caw")
import analyze_caw as A

V2 = "geo_v2_abcdef"
RES = f"{R}/caw/results/v2caw"
OUT = f"{R}/caw/out/v2caw"
A.RES = RES
CATS = A.CATS
R_LIST = [1.25, 1.5, 2.0, 3.0, 5.0]
WSETS = ["w1", "w2", "w3"]
STRONG = ["2d_ratio", "3d_calib"] + [x for x in __import__("os").environ.get("QP_STRONG_EXTRA", "").split(",") if x]
DIS = ["geo", "caw", "min", "gm"] + [f"c{k}{w}" for k in "12" for w in WSETS]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 60)


TRI = 1.5


def dist_bound(split, index):
    sys.path.insert(0, f"{R}/geo")
    from rules import UNIT_SI
    U, si = pd.Series(np.nan, index=index), pd.Series(np.nan, index=index)
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln)
        if x["id"] not in U.index or x["q"]["kind"] != "dist2" or x["video_type"][1] != "3":
            continue
        dep = {}
        for e in x["depth"]:
            dep.setdefault(e["key"], []).append(e["d"])
        k1 = [k for k in (x.get("target_dkeys") or []) if k in dep]
        k2 = [k for k in (x.get("target2_dkeys") or []) if k in dep]
        if not k1 or not k2:
            continue
        if set(k1) == set(k2):
            ds = sorted((max(dep[k]) for k in set(k1)), reverse=True)
            if len(ds) < 2:
                continue
            U[x["id"]] = ds[0] + ds[1]
        else:
            U[x["id"]] = max(max(dep[k]) for k in k1) + max(max(dep[k]) for k in k2)
        si[x["id"]] = UNIT_SI[x["q"]["unit"]]
    return U, si


def methods2(d, split, caw):
    g = pd.read_csv(f"{R}/preds/{split}_{V2}_raw.csv").set_index("Unnamed: 0").reindex(d.index)
    assert (g.video_id.astype(str).str.strip().values == d.vid.values).all()
    assert (g.question.values == d.question.values).all()
    M = pd.DataFrame(index=d.index)
    M["h"] = pd.to_numeric(g.h1_value, errors="coerce").abs()
    M["g"] = pd.to_numeric(g.geo_value, errors="coerce").where(g.answer_source == "meas")
    M["q"] = A.sane(A.pos(g.vlm_value), M.h)
    M["c"] = A.pos(caw.reindex(d.index))
    M["cs"] = A.sane(M.c, M.h)
    U, si = dist_bound(split, d.index)
    with np.errstate(invalid="ignore"):
        M["tri_rej"] = (M.cs * si > TRI * U).fillna(False) & M.cs.notna()
        assert not ((M.g * si > TRI * U).fillna(False)).any()
    M["cs"] = M.cs.where(~M.tri_rej)
    meas = M.g.notna()
    M["mode"] = g.meas_mode.fillna("").astype(str).where(meas, "")
    M["kind"] = g.meas_kind.fillna("").astype(str)
    M["why"] = g.meas_why.fillna("").astype(str)
    M["file"] = pd.to_numeric(g.parsed_value, errors="coerce")
    M["w1"] = meas & M["mode"].str.contains("fovdefault|samedepth")
    M["w2"] = M.w1 | (meas & M["mode"].str.contains("tdepth=prior"))
    M["w3"] = meas & ~M["mode"].isin(STRONG)
    assert (M.w3 == (M.w2 | (meas & (M["mode"] == "3d_knownsize")))).all()
    M["lab3d"] = (d.src == "lab") & (d.video_type.str[1] == "3") & M.kind.isin(A.MOTION) & meas
    return M


def route_mask(M, name):
    k, w = name[1], name[2:]
    return {"1": M[w], "2": M[w] | M.lab3d, "3": M.lab3d}[k]


def recipe2(M, name):
    g, c, cs, q, h = M.g, M.c, M.cs, M.q, M.h
    if name == "geo_v2":
        return A.chain(("geo", g), ("qwen8b", q), ("h1", h))
    if name == "a_caw":
        return A.chain(("caw", c))
    if name == "a_caw_h1":
        return A.chain(("caw", c), ("h1", h))
    if name == "b":
        return A.chain(("geo", g), ("caw_nogeo", cs), ("h1", h))
    if re.fullmatch(r"c[12]w[123]|c3", name):
        wk = route_mask(M, name if name != "c3" else "c3w1")
        sel = wk & cs.notna()
        v, s = A.chain(("geo", g.where(~sel)), ("caw_nogeo", cs), ("h1", h))
        s = s.where(~sel, "caw_routed")
        return v, s
    m = re.fullmatch(r"d_r([\d.]+)_(geo|gm)_(geo|caw|min|gm|c[12]w[123])", name)
    if m:
        r, agree, dis = float(m.group(1)), m.group(2), m.group(3)
        v, s = recipe2(M, "b")
        both = g.notna() & cs.notna()
        s = s.where(~(g.notna() & cs.isna()), "geo_cawinvalid")
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.abs(np.log(g / cs))
        ag = both & (lr <= math.log(r) + 1e-12)
        di = both & ~ag
        gm = np.sqrt(g * cs)
        if agree == "gm":
            v[ag], s[ag] = gm[ag], "agree_gm"
        else:
            s[ag] = "agree_geo"
        if dis == "caw":
            v[di], s[di] = cs[di], "dis_caw"
        elif dis == "min":
            v[di], s[di] = np.minimum(g, cs)[di], "dis_min"
        elif dis == "gm":
            v[di], s[di] = gm[di], "dis_gm"
        elif dis == "geo":
            s[di] = "dis_geo"
        else:
            rt = route_mask(M, dis)
            v[di & rt], s[di & rt] = cs[di & rt], "dis_caw_routed"
            s[di & ~rt] = "dis_geo"
        return v, s
    raise ValueError(name)


def fam_d(dis_list):
    return ["b"] + [f"d_r{r:g}_{a}_{z}" for r in R_LIST for a in ["geo", "gm"] for z in dis_list if (a, z) != ("geo", "geo")]


FAM_C1 = ["b", "c1w1", "c2w1", "c3"]
FAM_D1 = fam_d(["geo", "caw", "min", "gm", "c2w1"])
FAM_C = FAM_C1 + ["c1w2", "c2w2", "c1w3", "c2w3"]
FAM_D = fam_d(DIS)
FAM_R2GM = ["b"] + [f"d_r2_gm_{z}" for z in ["c1w1", "c1w2", "c1w3", "c2w1", "c2w2", "c2w3"]]
FAM_GMC1 = ["b"] + [f"d_r{r:g}_gm_c1{w}" for r in R_LIST for w in WSETS]
FAMILIES = {
    "v1 structure kept (r = 2, gm on agreement), route re-derived: b + 6": FAM_R2GM,
    "gm on agreement, route c1w1-w3, r chosen: b + 15": FAM_GMC1,
    "v1 family (c): b, c1w1, c2w1, c3": FAM_C1,
    "v1 family (d): 45 members + b, route = c2w1": FAM_D1,
    "v1 family, everything": FAM_C1 + FAM_D1[1:],
    "(c) with w1-w3: 8 members": FAM_C,
    "(d) with w1-w3 routes: 95 members + b": FAM_D,
    "everything: (c) + (d) with w1-w3": FAM_C + FAM_D[1:],
}


def boot2(sa, sb, cat, vid, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vids = vid.unique()
    nv = len(vids)
    vi = pd.Series(np.arange(nv), index=vids)[vid.values].values
    ci = pd.Series(np.arange(len(CATS)), index=CATS)[cat.values].values
    D, N = np.zeros((nv, len(CATS))), np.zeros((nv, len(CATS)))
    np.add.at(D, (vi, ci), np.asarray(sa, float) - np.asarray(sb, float))
    np.add.at(N, (vi, ci), 1.0)
    draws = np.stack([rng.choice(vids, nv) for _ in range(n)])
    order = np.argsort(vids)
    posn = order[np.searchsorted(vids[order], draws)]
    cnt = np.stack([np.bincount(p_, minlength=nv) for p_ in posn]).astype(float)
    SD, SN = cnt @ D, cnt @ N
    ok = (SN > 0).all(1)
    out = (SD[ok] / SN[ok]).mean(1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > 0).mean())


def lovo_select(names, vals, S, d):
    held = pd.Series(np.nan, index=d.index)
    picks = {}
    for v in d.vid.unique():
        te = (d.vid == v).values
        best = max(names, key=lambda n: (round(A.macro(S[n][~te], d.cat[~te])[0], 12), -names.index(n)))
        held[te] = vals[best][te]
        picks[v] = best
    return held, picks


def per_mode_lovo(M, d, y):
    Sg = pd.Series(A.row_mra(M.g, y), index=d.index)
    Sc = pd.Series(A.row_mra(M.cs, y), index=d.index)
    hc, hd, picks = pd.Series(np.nan, index=d.index), pd.Series(np.nan, index=d.index), {}
    for v in d.vid.unique():
        te = d.vid == v
        tr = ~te & M.g.notna() & M.cs.notna()
        weak_modes = {m_ for m_, ix in M[tr].groupby("mode").groups.items() if Sc[ix].mean() > Sg[ix].mean()}
        seen = set(M["mode"][tr])
        wk = M.g.notna() & (M["mode"].isin(weak_modes) | (~M["mode"].isin(seen) & M.w3))
        Mx = M.assign(wx=wk)
        Mx["w1"] = wk
        hc[te] = recipe2(Mx, "c1w1")[0][te]
        hd[te] = recipe2(Mx, "d_r2_gm_c1w1")[0][te]
        picks[v] = sorted(weak_modes)
    return hc, hd, picks


def overlap(t):
    vv = A.add_keys(A.load_val())
    key = vv.drop_duplicates(["vid", "question"]).set_index(["vid", "question"])["Unnamed: 0"]
    val_id = pd.Series(key.reindex(pd.MultiIndex.from_arrays([t.vid, t.question])).values, index=t.index)
    dup = t.duplicated(["vid", "question"], keep=False)
    ov = t.index[val_id.notna() & ~dup]
    return ov, vv.loc[val_id[ov].astype(int)].copy()


def sc_tab(F):
    show = F.copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "")
    return show.to_string(index=False)


def cmd_val():
    os.makedirs(RES, exist_ok=True)
    d = A.add_keys(A.load_val())
    y = d.y
    caw = pd.read_csv(f"{R}/preds/val_caw9b.csv").set_index("Unnamed: 0").parsed_value
    M = methods2(d, "val", caw)
    M1 = A.methods(d, "val", caw)
    v_file = pd.read_csv(f"{R}/preds/val_{V2}.csv").set_index("Unnamed: 0").parsed_value.reindex(d.index)
    assert np.allclose(recipe2(M, "geo_v2")[0].values, v_file.values, rtol=1e-9)
    print('')
    S = {k: pd.Series(A.row_mra(M[k], y), index=d.index) for k in ["g", "c", "cs", "h", "q"]}

    mm = M.g.notna()
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        l_g, l_c, l_gc = np.log10(M.g / y), np.log10(M.c / y), np.log10(M.g / M.c)
    for key, ix in list(M[mm].groupby("mode").groups.items()) + [("ALL measured", M.index[mm])]:
        sub = d.loc[ix]
        a, b = S["g"][ix], S["c"][ix]
        rows.append(dict(mode=key, rows=len(ix), videos=sub.vid.nunique(), geo=a.mean(), caw=b.mean(),
                         geo_better=int((a > b).sum()), caw_better=int((a < b).sum()), tie=int((a == b).sum()),
                         geo_bias=float(np.nanmedian(l_g[ix])), caw_bias=float(np.nanmedian(l_c[ix])),
                         log10_geo_over_caw=float(np.nanmedian(l_gc[ix])),
                         agree2x=int((np.abs(l_gc[ix]) <= math.log10(2)).sum()),
                         w1=bool(M.w1[ix].all()), w2=bool(M.w2[ix].all()), w3=bool(M.w3[ix].all()),
                         videos_list=json.dumps(sub.vid.value_counts().to_dict())))
    for lab, msk in [("lab-3D motion", M.lab3d), ("not lab-3D motion", mm & ~M.lab3d)] + \
                    [(f"weak set {w}", M[w]) for w in WSETS] + [(f"not {w} (geo kept)", mm & ~M[w]) for w in WSETS]:
        ix = M.index[msk]
        a, b = S["g"][ix], S["c"][ix]
        rows.append(dict(mode=lab, rows=len(ix), videos=d.loc[ix].vid.nunique(), geo=a.mean(), caw=b.mean(),
                         geo_better=int((a > b).sum()), caw_better=int((a < b).sum()), tie=int((a == b).sum())))
    PM = pd.DataFrame(rows)
    PM.to_csv(f"{RES}/val_per_mode.csv", index=False)
    print('')
    print(sc_tab(PM.drop(columns=["videos_list"])))
    print("   videos per mode:", {r_.mode: r_.videos_list for r_ in PM.itertuples() if isinstance(r_.videos_list, str)})
    um = ~mm
    print('')

    base = ["geo_v2", "a_caw", "a_caw_h1", "b", "c1w1", "c2w1", "c3", "c1w2", "c2w2", "c1w3", "c2w3"]
    allnames = base + [n for n in FAM_D if n != "b"]
    vals, srcs = {}, {}
    for n in allnames:
        vals[n], srcs[n] = recipe2(M, n)
    Ssc = {n: pd.Series(A.row_mra(vals[n], y), index=d.index) for n in allnames}
    v1_best, _ = A.recipe(M1, "d_r2_gm_route")
    v1_b, _ = A.recipe(M1, "b")
    v1_geo, _ = A.recipe(M1, "geo_v1")
    s_v1best = pd.Series(A.row_mra(v1_best, y), index=d.index)
    s_v2geo = Ssc["geo_v2"]
    s_b = Ssc["b"]
    s_v1geo = pd.Series(A.row_mra(v1_geo, y), index=d.index)
    ck_a, ck_b = A.boot(s_v1best, s_v1geo, d.cat, d.vid), boot2(s_v1best, s_v1geo, d.cat, d.vid)
    print('')
    assert np.allclose(ck_a, ck_b, atol=1e-12)
    rows = []
    for n in allnames:
        h, cm = A.macro(Ssc[n], d.cat)
        rec = dict(recipe=n, MRA=h, **cm.to_dict(),
                   caw_rows=int(srcs[n].str.contains("caw|gm|min").sum()),
                   changed_vs_b=int((~np.isclose(vals[n], vals["b"], rtol=1e-9, equal_nan=True)).sum()))
        for lab, ref in [("geo_v2", s_v2geo), ("b", s_b), ("v1fusion", s_v1best)]:
            lo, hi, p = boot2(Ssc[n], ref, d.cat, d.vid)
            rec[f"vs_{lab}"] = h - A.macro(ref, d.cat)[0]
            rec[f"ci_vs_{lab}"] = f"[{lo:+.3f}, {hi:+.3f}]"
            rec[f"P>0_vs_{lab}"] = round(p, 2)
        rows.append(rec)
    F = pd.DataFrame(rows)
    F.to_csv(f"{RES}/val_fusion_all.csv", index=False)
    dd_top = F[F.recipe.str.startswith("d_")].sort_values("MRA", ascending=False).recipe.head(6).tolist()
    for extra in ["d_r2_gm_c2w1", "d_r2_gm_c1w1", "d_r2_gm_c1w2", "d_r2_gm_c1w3"]:
        if extra not in dd_top:
            dd_top.append(extra)
    off = {}
    for n in base + dd_top:
        p = A.write_official(d, vals[n], f"{OUT}/val/{n}.csv")
        off[n] = A.score_official(p, f"val_{n}")
        assert abs(off[n]["MRA"] - F.set_index("recipe").loc[n, "MRA"]) < 6e-5
    print('')
    cols = ["recipe", "MRA", "S2", "D2", "S3", "D3", "vs_geo_v2", "ci_vs_geo_v2", "vs_b", "ci_vs_b", "vs_v1fusion",
            "ci_vs_v1fusion", "caw_rows", "changed_vs_b"]
    print(sc_tab(F[F.recipe.isin(base + dd_top)].set_index("recipe").loc[base + dd_top].reset_index()[cols]))
    for n in base + dd_top:
        print(f"   official {n:16s} {off[n]}")
    print('')
    grid = []
    for a in ["geo", "gm"]:
        for z in DIS:
            if (a, z) == ("geo", "geo"):
                continue
            grid.append(dict(rule=f"{a} / {z}", **{f"r={r:g}": F.set_index("recipe").loc[f"d_r{r:g}_{a}_{z}", "MRA"] for r in R_LIST}))
    G = pd.DataFrame(grid)
    G.to_csv(f"{RES}/val_d_grid.csv", index=False)
    print(sc_tab(G))

    print('')
    lovo_rows = []
    for lab, names in FAMILIES.items():
        held, picks = lovo_select(names, vals, Ssc, d)
        sh = pd.Series(A.row_mra(held, y), index=d.index)
        h, cm = A.macro(sh, d.cat)
        safe = re.sub(r"[^a-z0-9]+", "_", lab.lower()).strip("_")[:48]
        so = A.score_official(A.write_official(d, held, f"{OUT}/val/lovo_{safe}.csv"), f"lovo_{safe}")
        rec = dict(family=lab, members=len(names), LOVO=h, official=so["MRA"], **cm.to_dict())
        for lab2, ref in [("geo_v2", s_v2geo), ("b", s_b), ("v1fusion", s_v1best)]:
            lo, hi, p = boot2(sh, ref, d.cat, d.vid)
            rec[f"vs_{lab2}"] = h - A.macro(ref, d.cat)[0]
            rec[f"ci_vs_{lab2}"] = f"[{lo:+.3f}, {hi:+.3f}]"
        best_in = max(names, key=lambda n: (round(A.macro(Ssc[n], d.cat)[0], 12), -names.index(n)))
        rec["in_sample_best"] = f"{best_in} {A.macro(Ssc[best_in], d.cat)[0]:.4f}"
        rec["picks"] = json.dumps(pd.Series(list(picks.values())).value_counts().to_dict())
        lovo_rows.append(rec)
    L = pd.DataFrame(lovo_rows)
    L.to_csv(f"{RES}/val_fusion_lovo.csv", index=False)
    print(sc_tab(L))
    print('')
    rr = []
    for a in ["geo", "gm"]:
        for z in DIS:
            if (a, z) == ("geo", "geo"):
                continue
            names = [f"d_r{r:g}_{a}_{z}" for r in R_LIST]
            held, picks = lovo_select(names, vals, Ssc, d)
            sh = pd.Series(A.row_mra(held, y), index=d.index)
            lo, hi, p = boot2(sh, s_b, d.cat, d.vid)
            rr.append(dict(rule=f"{a} / {z}", LOVO=A.macro(sh, d.cat)[0], vs_b=A.macro(sh, d.cat)[0] - A.macro(s_b, d.cat)[0],
                           ci_vs_b=f"[{lo:+.3f}, {hi:+.3f}]",
                           r_picked=json.dumps({k.split('_')[1]: v for k, v in pd.Series(list(picks.values())).value_counts().items()})))
    RR = pd.DataFrame(rr).sort_values("LOVO", ascending=False)
    RR.to_csv(f"{RES}/val_lovo_over_r.csv", index=False)
    print(sc_tab(RR))
    hc, hd, pk = per_mode_lovo(M, d, y)
    for lab, held in [("per-mode chooser, c1 form", hc), ("per-mode chooser, d_r2_gm form", hd)]:
        sh = pd.Series(A.row_mra(held, y), index=d.index)
        lo, hi, p = boot2(sh, s_b, d.cat, d.vid)
        print('')
    print('t128', pd.Series([json.dumps(x) for x in pk.values()]).value_counts().to_dict())

    print('')
    best_fixed = F.sort_values("MRA", ascending=False).iloc[0].recipe
    for n in ["b", "c1w3", "d_r2_gm_c2w1", "d_r2_gm_c1w3", best_fixed]:
        diff = Ssc[n] - s_v1best
        byv = diff.groupby(d.vid).sum() / 1.0
        print('')

    for lab, cols_ in {"oracle {geo_v2, CaW}": ["g", "c"], "oracle {geo_v2, CaW, H1}": ["g", "c", "h"]}.items():
        best = np.max(np.vstack([A.row_mra(M[k], y) for k in cols_]), 0)
        print(f"   {lab}: {A.fmt_macro(best, d.cat)}")
    both = M.g.notna() & M.c.notna()
    eg, ec = np.log10(M.g / y), np.log10(M.c / y)
    print('')

    t = A.add_keys(A.load_test())
    caw_t = pd.read_csv(f"{R}/preds/test_caw9b.csv").set_index("Unnamed: 0").parsed_value
    T = methods2(t, "test", caw_t)
    T1 = A.methods(t, "test", caw_t)
    ov, oo = overlap(t)
    print('')
    orows = []
    tv = {n: recipe2(T, n)[0] for n in base + dd_top}
    tv["v1: geo_v1"] = A.recipe(T1, "geo_v1")[0]
    tv["v1: b"] = A.recipe(T1, "b")[0]
    tv["v1: d_r2_gm_route (sub_caw_fusion)"] = A.recipe(T1, "d_r2_gm_route")[0]
    for n, vv_ in tv.items():
        tag = re.sub(r"[^A-Za-z0-9_.]+", "_", n)
        so = A.score_official(A.write_official(oo, vv_[ov].values, f"{OUT}/val124/{tag}.csv"), f"val124_{tag}")
        orows.append(dict(recipe=n, **so))
    O = pd.DataFrame(orows)
    O.to_csv(f"{RES}/val124_overlap.csv", index=False)
    print(O.to_string(index=False))
    tst = T.g.notna()
    print('')

    per = d[["video_id", "cat", "src", "inference_type", "question", "y"]].copy()
    for k in ["g", "c", "cs", "h"]:
        per[k] = M[k]
        per[f"mra_{k}"] = S[k]
    for k in ["mode", "kind", "why", "w1", "w2", "w3", "lab3d"]:
        per[k] = M[k]
    per["v1_fusion"] = v1_best
    per["mra_v1_fusion"] = s_v1best
    for n in ["b", "c1w3", "d_r2_gm_c2w1", "d_r2_gm_c1w3", best_fixed]:
        per[n] = vals[n]
        per[f"mra_{n}"] = Ssc[n]
    per.to_csv(f"{RES}/val_per_row.csv")
    print('')


def cmd_testdiag(*names):
    names = list(names) or ["b", "c1w1", "c1w3", "d_r2_gm_geo", "d_r2_gm_c1w1", "d_r2_gm_c1w3", "d_r2_gm_c2w1"]
    t = A.add_keys(A.load_test())
    caw = pd.read_csv(f"{R}/preds/test_caw9b.csv").set_index("Unnamed: 0").parsed_value
    M = methods2(t, "test", caw)
    d = A.add_keys(A.load_val())
    Mv = methods2(d, "val", pd.read_csv(f"{R}/preds/val_caw9b.csv").set_index("Unnamed: 0").parsed_value)
    rows = []
    for split, X in [("val", Mv), ("test", M)]:
        both = X.g.notna() & X.cs.notna()
        with np.errstate(divide="ignore", invalid="ignore"):
            l10 = np.log10(X.g / X.cs)
        for m_, ix in X[both].groupby("mode").groups.items():
            ll = l10[ix]
            rows.append(dict(split=split, mode=m_, rows_both=len(ix), agree2x=round(float((np.abs(ll) <= math.log10(2)).mean()), 3),
                             agree1_25x=round(float((np.abs(ll) <= math.log10(1.25)).mean()), 3),
                             median_log10_geo_over_caw=round(float(ll.median()), 3), q25=round(float(ll.quantile(.25)), 3),
                             q75=round(float(ll.quantile(.75)), 3), w1=bool(X.w1[ix].all()), w3=bool(X.w3[ix].all())))
    AG = pd.DataFrame(rows).sort_values(["mode", "split"])
    AG.to_csv(f"{RES}/test_val_mode_agreement.csv", index=False)
    print('')
    print(AG.to_string(index=False))
    both = M.g.notna() & M.cs.notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        agr = (np.abs(np.log10(M.g / M.cs)) <= math.log10(2))[both]
    tb = pd.DataFrame({"mode": M["mode"][both], "src": t.src[both], "agree": agr})
    print('')
    print(tb.groupby(["mode", "src"]).agree.agg(lambda s_: f"{s_.mean():.2f} ({len(s_)})").unstack().fillna("").to_string())
    vals = {}
    for n in names:
        v, s = recipe2(M, n)
        vals[n] = v
        print('')
        print(pd.crosstab(s, t.cat, margins=True).to_string())
    print('')
    D = pd.DataFrame({a: {b: int((~np.isclose(vals[a], vals[b], rtol=1e-9)).sum()) for b in names} for a in names})
    print(D.to_string())
    v1 = pd.read_csv(f"{R}/submissions/sub_caw_fusion.csv").parsed_value.values
    for n in names:
        print('')
    T1 = A.methods(t, "test", caw)
    newm = M.g.notna() & ~M["mode"].isin(set(Mv["mode"][Mv.g.notna()]))
    print('')
    print(pd.crosstab(M["mode"][newm], T1["mode"].where(T1.g.notna(), "unmeasured")[newm]).to_string())
    print("   by source:", pd.crosstab(M["mode"][newm], t.src[newm]).to_dict())


def same_text_but_id(p0, p1):
    import csv
    a = list(csv.reader(open(p0, encoding="utf-8", newline="")))
    b = list(csv.reader(open(p1, encoding="utf-8", newline="")))
    if len(a) != len(b) or a[0][1:] != b[0][1:] or a[0][0] != "Unnamed: 0" or b[0][0] != "id":
        return False
    return all(x[0] == str(i) and z[0] == str(i + 1) and x[1:] == z[1:] for i, (x, z) in enumerate(zip(a[1:], b[1:])))


def cmd_submit(name, out="sub_v2caw_fusion"):
    t = A.add_keys(A.load_test())
    caw = pd.read_csv(f"{R}/preds/test_caw9b.csv").set_index("Unnamed: 0").parsed_value
    M = methods2(t, "test", caw)
    v, s = recipe2(M, name)
    assert v.notna().all() and np.isfinite(v).all() and (v > 0).all()
    os.makedirs(f"{R}/submissions", exist_ok=True)
    p0, p1 = f"{R}/submissions/{out}.csv", f"{R}/submissions/{out}_id1.csv"
    o = t[[c for c in A.OUT_COLS if c != "parsed_value"]].copy()
    o["parsed_value"] = v.values
    o = o[A.OUT_COLS]
    o.to_csv(p0, index=False)
    o1 = o.rename(columns={"Unnamed: 0": "id"})
    o1["id"] = o1["id"] + 1
    o1.to_csv(p1, index=False)
    det = t[["Unnamed: 0", "video_id", "question", "cat", "src"]].copy()
    det["answer"], det["answer_source"] = v.values, s.values
    for k in ["g", "c", "cs", "q", "h", "mode", "kind", "w1", "w2", "w3", "lab3d", "tri_rej"]:
        det[k] = M[k].values
    det.to_csv(f"{R}/submissions/{out}_detail.csv", index=False)

    pq = pd.read_parquet(f"{R}/data/test/test_dataset.parquet").reset_index(drop=True)
    o, o1 = pd.read_csv(p0), pd.read_csv(f"{R}/submissions/{out}_id1.csv")
    ref0 = pd.read_csv(f"{R}/submissions/sub_caw_fusion.csv")
    ref1 = pd.read_csv(f"{R}/submissions/sub_caw_fusion_id1.csv")
    chk = {
        "rows": len(o) == len(pq) == len(o1) == 3289,
        "0-based ids 0..3288": (o["Unnamed: 0"].values == np.arange(len(pq))).all(),
        "id1 ids 1..3289": (o1["id"].values == np.arange(1, len(pq) + 1)).all(),
        "video_id = parquet row": (o.video_id.astype(str).values == pq.video_id.astype(str).values).all(),
        "question = parquet row": (o.question.values == pq.question.values).all(),
        "prior = parquet row": (o.ground_truth_prior.astype(str).values == pq.prior.astype(str).values).all(),
        "id1 file = 0-based file shifted (text)": same_text_but_id(p0, p1),
        "columns = sub_caw_fusion.csv": list(o.columns) == list(ref0.columns),
        "columns = sub_caw_fusion_id1.csv": list(o1.columns) == list(ref1.columns),
        "same rows/keys as sub_caw_fusion_id1.csv": (o1[["id", "video_id", "question"]].astype(str).values ==
                                                     ref1[["id", "video_id", "question"]].astype(str).values).all(),
        "finite positive": bool(np.isfinite(o1.parsed_value).all() and (o1.parsed_value > 0).all()),
    }
    print('')
    assert all(chk.values())
    print('')
    tr = M.tri_rej
    print(f"   CaW answers rejected by the triangle rule: {int(tr.sum())} -> now answered by {s[tr].value_counts().to_dict()}: "
          + "; ".join(f"row {i} {t.video_id[i]}: CaW {M.c[i]:g}, final {v[i]:.4g}" for i in t.index[tr]))
    lab = s.replace({"agree_gm": "gm of geo and CaW (agree within r)", "agree_geo": "geo (agree within r)",
                     "dis_caw_routed": "CaW (disagree, weak mode)", "dis_geo": "geo (disagree, geo kept)",
                     "caw_nogeo": "CaW (no geo measurement)", "geo_cawinvalid": "geo (CaW invalid or outside the sanity rules)",
                     "h1": "H1 (neither)", "geo": "geo", "caw_routed": "CaW (weak mode)"})
    print('')
    print(pd.crosstab(lab, t.cat, margins=True).to_string())
    print('')
    print(pd.crosstab(lab, t.src, margins=True).to_string())
    print('t139', M["mode"][s == "dis_caw_routed"].value_counts().to_dict())
    print('t140', M["mode"][s == "dis_geo"].value_counts().to_dict())
    print('t141', M["mode"][s == "agree_gm"].value_counts().to_dict())
    tst = M.g.notna()
    both = tst & M.cs.notna()
    ag = both & (np.abs(np.log(M.g / M.cs)) <= math.log(2))
    tab = pd.DataFrame({"mode": M["mode"].where(tst, "unmeasured"), "both": both, "agree2x": ag})
    x = tab.groupby("mode").agg(rows=("both", "size"), both=("both", "sum"), agree2x=("agree2x", "sum"))
    x["agree_share"] = (x.agree2x / x.both).round(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        l10 = np.log10(M.g / M.cs)
    x["median_log10_geo/caw"] = l10[both].groupby(tab["mode"][both]).median().round(3)
    x["q25"] = l10[both].groupby(tab["mode"][both]).quantile(0.25).round(3)
    x["q75"] = l10[both].groupby(tab["mode"][both]).quantile(0.75).round(3)
    print('')
    print(x.to_string())
    for ref_name in ["submissions/sub_caw_fusion.csv", f"preds/test_{V2}.csv", "preds/test_caw9b.csv"]:
        rv = pd.to_numeric(pd.read_csv(f"{R}/{ref_name}").parsed_value, errors="coerce").abs().values
        diff = ~np.isclose(v.values, rv, rtol=1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            big = np.abs(np.log10(v.values / rv)) > 0.1
        print('')
    print("   answer quantiles:", np.round(np.percentile(v.values, [0, 1, 50, 99, 100]), 5).tolist())
    ov, oo = overlap(t)
    so = A.score_official(A.write_official(oo, v[ov].values, f"{OUT}/val124/{out}.csv"), f"val124_{out}")
    print('')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "val":
        cmd_val()
    elif sys.argv[1] == "submit":
        cmd_submit(*sys.argv[2:])
    elif sys.argv[1] == "testdiag":
        cmd_testdiag(*sys.argv[2:])
    else:
        sys.exit(__doc__)
