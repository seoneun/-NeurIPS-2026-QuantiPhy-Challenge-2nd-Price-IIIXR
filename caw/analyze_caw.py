#!/usr/bin/env python
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_R") or os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/tools")
from qp_common import OUT_COLS, UNIT_TABLE, norm_unit, target_unit

THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CATS = ["S2", "D2", "S3", "D3"]
MOTION = {"speed", "avgspeed", "accel", "disp", "path", "hvel"}
RES = f"{R}/caw/results"
R_LIST = [1.25, 1.5, 2.0, 3.0, 5.0]
AGREE = ["geo", "gm"]
DISAGREE = ["geo", "caw", "min", "gm", "route"]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 400)


def row_mra(p, y):
    p = np.abs(np.asarray(p, float))
    y = np.asarray(y, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(p - y) / y
        s = sum((r < (1 - t)).astype(float) for t in THETAS) / 10
    return np.where(np.isfinite(p) & (p != 0), s, 0.0)


def macro(s, cat):
    m = pd.Series(np.asarray(s, float), index=cat.index).groupby(cat).mean().reindex(CATS)
    return float(m.mean()), m


def fmt_macro(s, cat):
    h, m = macro(s, cat)
    return f"{h:.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in m.items())


def score_official(csv, name):
    if not os.path.exists(f"{R}/score_val.sh"):
        d = load_val()
        p = pd.read_csv(csv)
        p.index = p[p.columns[0]].astype(int)
        y = d.y.reindex(p.index)
        cat = p.inference_type.str[0] + p.video_type.str[1]
        h, m = macro(row_mra(pd.to_numeric(p.parsed_value, errors="coerce"), y), cat)
        return {"MRA": round(h, 4), **{k: round(v, 4) for k, v in m.items()}, "LOCAL_DRY_RUN": 1}
    o = subprocess.run([f"{R}/score_val.sh", csv, f"{RES}/scores/{name}"], capture_output=True, text=True)
    line = [ln for ln in o.stdout.splitlines() if ln.startswith("RESULT")]
    if not line:
        print(o.stdout + o.stderr)
        return None
    vals = dict(re.findall(r"(MRA|S2|D2|S3|D3|invalid%)=([\d.]+)", line[0]))
    return {k: float(v) for k, v in vals.items()}


def boot(sa, sb, cat, vid, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vids = vid.unique()
    groups = {v: np.where(vid.values == v)[0] for v in vids}
    a, b, c = np.asarray(sa, float), np.asarray(sb, float), cat.values
    out = []
    for _ in range(n):
        idx = np.concatenate([groups[v] for v in rng.choice(vids, len(vids))])
        cc = c[idx]
        if len(set(cc)) < 4:
            continue
        da = pd.Series(a[idx]).groupby(cc).mean().mean()
        db = pd.Series(b[idx]).groupby(cc).mean().mean()
        out.append(da - db)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > 0).mean())


def load_val():
    d = pd.read_csv(f"{R}/data/val/validation_dataset.csv")
    d = d[[c for c in d.columns if not c.startswith("Unnamed") or c == "Unnamed: 0"]].copy()
    d.index = d["Unnamed: 0"].astype(int)
    d["y"] = d.ground_truth_posterior.astype(float)
    return d


def load_test():
    t = pd.read_parquet(f"{R}/data/test/test_dataset.parquet").reset_index(drop=True)
    t.insert(0, "Unnamed: 0", np.arange(len(t)))
    t = t.rename(columns={"prior": "ground_truth_prior"})
    t["ground_truth_posterior"] = np.nan
    t.index = t["Unnamed: 0"]
    return t


def add_keys(d):
    d["cat"] = d.inference_type.str[0] + d.video_type.str[1]
    d["src"] = d.video_source.replace({"segmentaion": "segmentation"})
    d["vid"] = d.video_id.astype(str).str.strip()
    return d


def pos(v):
    v = pd.to_numeric(v, errors="coerce").abs()
    return v.where(np.isfinite(v) & (v > 0))


def sane(v, h, maxdec=2.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.abs(np.log10(v / h)) <= maxdec
    return v.where(ok)


def methods(d, split, caw):
    g = pd.read_csv(f"{R}/preds/{split}_geo_v1_raw.csv").set_index("Unnamed: 0").reindex(d.index)
    M = pd.DataFrame(index=d.index)
    M["h"] = pd.to_numeric(g.h1_value, errors="coerce").abs()
    M["g"] = pd.to_numeric(g.meas_value, errors="coerce").where(g.answer_source == "meas")
    M["q"] = sane(pos(g.vlm_value), M.h)
    M["c"] = pos(caw.reindex(d.index))
    M["cs"] = sane(M.c, M.h)
    M["mode"] = g.meas_mode.fillna("").astype(str)
    M["kind"] = g.meas_kind.fillna("").astype(str)
    M["why"] = g.meas_why.fillna("").astype(str)
    M["geo_v1_file"] = pd.to_numeric(g.parsed_value, errors="coerce")
    M["weak_mode"] = M["mode"].str.contains("fovdefault|samedepth") & M.g.notna()
    M["lab3d_motion"] = (d.src == "lab") & (d.video_type.str[1] == "3") & M.kind.isin(MOTION) & M.g.notna()
    return M


def chain(*pairs):
    idx = pairs[0][1].index
    val = pd.Series(np.nan, index=idx)
    src = pd.Series("none", index=idx, dtype=object)
    for lab, s in pairs[::-1]:
        ok = s.notna()
        val[ok] = s[ok]
        src[ok] = lab
    return val, src


def recipe(M, name):
    g, c, cs, q, h = M.g, M.c, M.cs, M.q, M.h
    if name == "geo_v1":
        return chain(("geo", g), ("qwen8b", q), ("h1", h))
    if name == "a_caw":
        return chain(("caw", c))
    if name == "a_caw_h1":
        return chain(("caw", c), ("h1", h))
    if name == "b":
        return chain(("geo", g), ("caw", cs), ("h1", h))
    if name == "b_nosanity":
        return chain(("geo", g), ("caw", c), ("h1", h))
    if name in ("c1", "c2", "c3"):
        weak = {"c1": M.weak_mode, "c2": M.weak_mode | M.lab3d_motion, "c3": M.lab3d_motion}[name]
        g2 = g.where(~(weak & cs.notna()))
        v, s = chain(("geo", g2), ("caw", cs), ("h1", h))
        s = s.where(~(weak & cs.notna()), "caw_routed")
        return v, s
    m = re.fullmatch(r"d_r([\d.]+)_(geo|gm)_(geo|caw|min|gm|route)", name)
    if m:
        r, agree, dis = float(m.group(1)), m.group(2), m.group(3)
        v, s = recipe(M, "b")
        both = g.notna() & cs.notna()
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
        elif dis == "route":
            vr, _ = recipe(M, "c2")
            v[di], s[di] = vr[di], "dis_route_c2"
        else:
            s[di] = "dis_geo"
        return v, s
    raise ValueError(name)


def d_family():
    names = ["b"]
    for r in R_LIST:
        for a in AGREE:
            for z in DISAGREE:
                if (a, z) != ("geo", "geo"):
                    names.append(f"d_r{r:g}_{a}_{z}")
    return names


def lovo_select(names, M, d):
    vals = {n: recipe(M, n)[0] for n in names}
    S = {n: pd.Series(row_mra(vals[n], d.y), index=d.index) for n in names}
    held = pd.Series(np.nan, index=d.index)
    picks = {}
    for v in d.vid.unique():
        te = (d.vid == v).values
        best = max(names, key=lambda n: (round(macro(S[n][~te], d.cat[~te])[0], 12), -names.index(n)))
        held[te] = vals[best][te]
        picks[v] = best
    return held, picks


def write_official(d, value, path):
    o = d[[c for c in OUT_COLS if c != "parsed_value"]].copy()
    o["parsed_value"] = np.asarray(value, float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o[OUT_COLS].to_csv(path, index=False)
    return path


NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def resp_unit(text):
    t = re.sub(r"</?answer>", "", str(text), flags=re.I).strip()
    m = re.match(rf"^\s*({NUM})\s*(.*)$", t, flags=re.S)
    if not m:
        return "no-leading-number"
    u = m.group(2).strip().split("\n")[0].strip().rstrip(".,;")
    if not u:
        return "bare"
    return norm_unit(u.replace("meters per second squared", "m/s^2").replace("meters per second", "m/s")) or f"?{u[:20]}"


def prior_number(p):
    s = str(p)
    cut = max(s.rfind("="), s.rfind("~"))
    m = re.search(NUM, s[cut + 1:])
    return float(m.group(0)) if m else np.nan


def depth_numbers(t):
    return [float(x) for x in re.findall(r"=\s*(" + NUM + ")", str(t))] if isinstance(t, str) else []


def unit_table(d, resp):
    asked = pd.Series([target_unit(q, it) for q, it in zip(d.question, d.inference_type)], index=d.index)
    got = resp.map(resp_unit)
    same = [UNIT_TABLE.get(a) is not None and UNIT_TABLE.get(a) == UNIT_TABLE.get(b) for a, b in zip(asked, got)]
    kind = np.where(same, "same unit", np.where(got == "bare", "bare number",
                    np.where(got.str.startswith("?") | (got == "no-leading-number"), "other text", "different unit")))
    return asked, got, pd.Series(kind, index=d.index)


def err_summary(e):
    e = e.dropna()
    if not len(e):
        return {}
    a = np.abs(e)
    return dict(n=len(e), median=round(float(e.median()), 3), q25=round(float(e.quantile(.25)), 3),
                q75=round(float(e.quantile(.75)), 3), within_10pct=round(float((a < math.log10(1.1)).mean()), 3),
                within_x1_5=round(float((a < math.log10(1.5)).mean()), 3), within_x2=round(float((a < math.log10(2)).mean()), 3),
                over_x2=round(float((e > math.log10(2)).mean()), 3), under_x2=round(float((e < -math.log10(2)).mean()), 3),
                off_x10=round(float((a > 1).mean()), 3))


def wl(d, sa, sb, mask, by, la, lb):
    rows = []
    sub = d[mask]
    for key, gr in list(sub.groupby(by)) + [("ALL", sub)]:
        a, b = sa[gr.index], sb[gr.index]
        rows.append({by if isinstance(by, str) else "group": key, "n": len(gr), f"{la} MRA": round(a.mean(), 3),
                     f"{lb} MRA": round(b.mean(), 3), f"{la} better": int((a > b).sum()),
                     f"{lb} better": int((a < b).sum()), "tie": int((a == b).sum())})
    return pd.DataFrame(rows)


def cmd_val(run_dir, tag="env030"):
    os.makedirs(RES, exist_ok=True)
    d = add_keys(load_val())
    rc = pd.read_csv(f"{run_dir}/9b.csv")
    rc.index = rc[rc.columns[0]].astype(int)
    gen = pd.DataFrame([json.loads(x) for x in open(f"{run_dir}/9b.generations.jsonl", encoding="utf-8")])
    gen.index = gen.original_id.astype(int)
    info = json.load(open(f"{run_dir}/9b.json", encoding="utf-8"))
    assert set(rc.index) == set(d.index) and set(gen.index) == set(d.index)
    print('')
    print("   generation stats:", info["generation_stats"])

    caw = pd.to_numeric(rc.parsed_value, errors="coerce")
    p_off = write_official(d, caw.reindex(d.index), f"{R}/preds/val_caw9b.csv")
    raw = d[[c for c in OUT_COLS if c != "parsed_value"]].copy()
    raw["parsed_value"] = caw.reindex(d.index).values
    raw["raw_response"] = rc.raw_response.reindex(d.index).values
    raw["finish_reason"] = gen.finish_reason.reindex(d.index).values
    raw["num_generated_tokens"] = gen.num_generated_tokens.reindex(d.index).values
    raw.to_csv(f"{R}/preds/val_caw9b_raw.csv", index=False)
    s_rel = score_official(f"{run_dir}/9b.csv", f"caw9b_release_csv_{tag}")
    s_off = score_official(p_off, f"val_caw9b_{tag}")
    print('')
    print('')
    y = d.y
    sc_caw = pd.Series(row_mra(caw.reindex(d.index), y), index=d.index)
    print('')
    their = gen.mra.reindex(d.index).astype(float)
    diff = (their - sc_caw).abs() > 1e-9
    print('')
    if diff.any():
        print(pd.DataFrame({"id": d.index[diff], "resp": rc.raw_response.reindex(d.index)[diff].values,
                            "gt": y[diff].values, "theirs": their[diff].values, "official": sc_caw[diff].values}).to_string())
    print('')

    M = methods(d, "val", caw)
    resp = rc.raw_response.reindex(d.index).astype(str)
    inv = M.c.isna()
    print('')
    if inv.any():
        print(pd.DataFrame({"id": d.index[inv], "response": resp[inv].values}).to_string(index=False))
    asked, got, ukind = unit_table(d, resp)
    print('t93', ukind.value_counts().to_dict())
    odd = ukind.isin(["different unit", "other text"])
    if odd.any():
        print(pd.DataFrame({"id": d.index[odd], "asked": asked[odd].values, "got": got[odd].values,
                            "response": resp[odd].str[:60].values}).to_string(index=False))
    e = np.log10(M.c / y)
    rows = [dict(group="ALL", **err_summary(e))]
    for col in ["cat", "src"]:
        for k, gr in d.groupby(col):
            rows.append(dict(group=f"{col}={k}", **err_summary(e[gr.index])))
    for k, gr in M.groupby("kind"):
        rows.append(dict(group=f"kind={k}", **err_summary(e[gr.index])))
    E = pd.DataFrame(rows)
    print('')
    print(E.to_string(index=False))
    E.to_csv(f"{RES}/caw_val_error_summary.csv", index=False)
    pn = d.ground_truth_prior.map(prior_number)
    copy_h1 = np.isclose(M.c, M.h, rtol=1e-3)
    copy_num = np.isclose(M.c, pn, rtol=1e-3)
    copy_depth = pd.Series([any(np.isclose(cv, dn, rtol=1e-3) for dn in depth_numbers(t)) if np.isfinite(cv) else False
                            for cv, t in zip(M.c, d.depth_info)], index=d.index)
    print('')
    eg = np.log10(M.g / y)
    both = M.g.notna() & M.c.notna()
    print('')

    S = {k: pd.Series(row_mra(M[k], y), index=d.index) for k in ["g", "c", "q", "h"]}
    gm = M.g.notna()
    print('')
    tabs = {}
    for by in ["cat", "src", "kind", "mode", "inference_type"]:
        dd = d.assign(kind=M.kind, mode=M["mode"].replace("", "none"))
        tabs[by] = wl(dd, S["g"], S["c"], gm, by, "geo", "CaW")
        print(tabs[by].to_string(index=False))
    pd.concat([t.assign(by=k) for k, t in tabs.items()]).to_csv(f"{RES}/val_geo_vs_caw_wl.csv", index=False)
    print('')
    dd = d.assign(why=M.why.str.replace(r"\(.*", "", regex=True))
    S_q_or_h = pd.Series(row_mra(M.q.fillna(M.h), y), index=d.index)
    S_c_or_h = pd.Series(row_mra(M.cs.fillna(M.h), y), index=d.index)
    for by in ["cat", "src", "why"]:
        print(wl(dd, S_c_or_h, S_q_or_h, ~gm, by, "CaW", "geo_v1 fallback").to_string(index=False))
    print(wl(d, S_c_or_h, S["h"], ~gm, "cat", "CaW", "H1").to_string(index=False))

    print('')
    base = ["geo_v1", "a_caw", "a_caw_h1", "b", "b_nosanity", "c1", "c2", "c3"]
    fam = d_family()
    rows, vals = [], {}
    for n in base + [x for x in fam if x != "b"]:
        v, s = recipe(M, n)
        vals[n] = v
        sc = pd.Series(row_mra(v, y), index=d.index)
        h, cm = macro(sc, d.cat)
        rows.append(dict(recipe=n, MRA=h, **cm.to_dict(), caw_rows=int(s.str.contains("caw|gm|min").sum()),
                         changed_vs_b=int((~np.isclose(v, recipe(M, "b")[0], rtol=1e-9, equal_nan=True)).sum())))
    F = pd.DataFrame(rows)
    gv1 = F.set_index("recipe").loc["geo_v1", "MRA"]
    assert np.allclose(recipe(M, "geo_v1")[0].values, M.geo_v1_file.values, rtol=1e-9)
    print('')
    off = {}
    for n in base:
        p = write_official(d, vals[n], f"{R}/caw/out/val_fusion/{n}.csv")
        off[n] = score_official(p, f"fusion_{n}")
        assert abs(off[n]["MRA"] - F.set_index("recipe").loc[n, "MRA"]) < 6e-5
    sb = pd.Series(row_mra(vals["b"], y), index=d.index)
    sg = pd.Series(row_mra(vals["geo_v1"], y), index=d.index)
    for i, r in F.iterrows():
        sc = pd.Series(row_mra(vals[r.recipe], y), index=d.index)
        lo, hi, p = boot(sc, sg, d.cat, d.vid)
        F.loc[i, "vs_geo_v1"] = r.MRA - gv1
        F.loc[i, "ci_vs_geo_v1"] = f"[{lo:+.3f}, {hi:+.3f}]"
        F.loc[i, "P>0_vs_geo_v1"] = round(p, 2)
        lo, hi, p = boot(sc, sb, d.cat, d.vid)
        F.loc[i, "vs_b"] = r.MRA - F.set_index("recipe").loc["b", "MRA"]
        F.loc[i, "ci_vs_b"] = f"[{lo:+.3f}, {hi:+.3f}]"
    F.to_csv(f"{RES}/val_fusion_all.csv", index=False)
    show = F.copy()
    for c in ["MRA", "S2", "D2", "S3", "D3", "vs_geo_v1", "vs_b"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    print(show[show.recipe.isin(base)].to_string(index=False))
    print('')
    for n in base:
        print(f"   {n:12s} {off[n]}")
    print('')
    print(show[show.recipe.str.startswith("d_")].to_string(index=False))

    print('')
    fams = {"(c) routing variants {b, c1, c2, c3}": ["b", "c1", "c2", "c3"],
            "(d) agreement family (r x agree x disagree, includes b)": fam,
            "whole family {b, c1, c2, c3} + (d)": ["b", "c1", "c2", "c3"] + fam[1:]}
    lovo_rows = []
    for lab, names in fams.items():
        held, picks = lovo_select(names, M, d)
        sh = pd.Series(row_mra(held, y), index=d.index)
        h, cm = macro(sh, d.cat)
        lo, hi, p = boot(sh, sb, d.cat, d.vid)
        lo2, hi2, p2 = boot(sh, sg, d.cat, d.vid)
        cnt = pd.Series(list(picks.values())).value_counts().to_dict()
        safe = re.sub(r"[^a-z0-9]+", "_", lab.lower()).strip("_")[:40]
        po = write_official(d, held, f"{R}/caw/out/val_fusion/lovo_{safe}.csv")
        so = score_official(po, f"lovo_{safe}")
        print(f"   {lab}: LOVO {h:.4f} (official {so['MRA']:.4f}) | " + " ".join(f"{k} {v:.4f}" for k, v in cm.items()))
        print('')
        lovo_rows.append(dict(family=lab, LOVO=h, **cm.to_dict(), official=so["MRA"], picks=json.dumps(cnt)))
    pd.DataFrame(lovo_rows).to_csv(f"{RES}/val_fusion_lovo.csv", index=False)

    for lab, cols in {"oracle {geo, CaW}": ["g", "c"], "oracle {geo, CaW, H1}": ["g", "c", "h"],
                      "oracle {geo, Qwen8B, H1}": ["g", "q", "h"]}.items():
        best = np.max(np.vstack([row_mra(M[k], y) for k in cols]), 0)
        print(f"   {lab}: {fmt_macro(best, d.cat)}")

    per = d[["video_id", "cat", "src", "inference_type", "question", "ground_truth_prior", "y"]].copy()
    for k in ["g", "c", "q", "h"]:
        per[k] = M[k]
        per[f"mra_{k}"] = S[k]
    per["mode"], per["kind"], per["why"] = M["mode"], M.kind, M.why
    per["weak_mode"], per["lab3d_motion"] = M.weak_mode, M.lab3d_motion
    per["caw_response"] = resp
    per.to_csv(f"{RES}/val_per_row.csv")
    print('')


def cmd_cf(cf_jsonl, main_csv=None):
    d = add_keys(load_val())
    x = pd.DataFrame([json.loads(l) for l in open(cf_jsonl, encoding="utf-8")])
    x["id"] = x.original_id.astype(int)
    ctl = x[x.id < 100000].set_index("id")
    x2 = x[x.id >= 100000].copy()
    x2.index = x2.id - 100000
    ids = ctl.index.intersection(x2.index)
    pc, p2 = pos(ctl.parsed_value[ids]), pos(x2.parsed_value[ids])
    ratio = p2 / pc
    lr = np.log2(ratio)
    cls = np.where(ratio.isna(), "invalid", np.where((ratio >= 1.8) & (ratio <= 2.2), "doubled (x1.8-2.2)",
                   np.where((ratio > 1 / 1.1) & (ratio < 1.1), "unchanged (x0.91-1.1)",
                            np.where(ratio > 2.2, "more than x2.2", np.where(ratio > 1.1, "x1.1-1.8", "decreased")))))
    cls = pd.Series(cls, index=ids)
    g = pd.read_csv(f"{R}/preds/val_geo_v1_raw.csv").set_index("Unnamed: 0")
    kind = g.meas_kind.reindex(ids).fillna("")
    print('')
    print("   outcome:", cls.value_counts().to_dict())
    print('')
    for col, key in [("cat", d.cat.reindex(ids)), ("kind", kind), ("prior", d.ground_truth_prior.reindex(ids))]:
        t = pd.DataFrame({"g": key, "lr": lr, "dbl": cls.str.startswith("doubled"), "same": cls.str.startswith("unchanged")})
        print(t.groupby("g").agg(n=("lr", "size"), median_elasticity=("lr", "median"), doubled=("dbl", "sum"),
                                 unchanged=("same", "sum")).round(3).to_string())
    y = d.y.reindex(ids)
    m1 = row_mra(pc, y).mean()
    m2 = row_mra(p2, 2 * y).mean()
    print('')
    if main_csv:
        mv = pos(pd.read_csv(main_csv).set_index("id").parsed_value.reindex(ids))
        same = np.isclose(mv, pc, rtol=1e-9)
        print('')
    out = pd.DataFrame({"video_id": d.video_id.reindex(ids), "cat": d.cat.reindex(ids), "kind": kind,
                        "question": d.question.reindex(ids), "prior": d.ground_truth_prior.reindex(ids),
                        "prior_x2": x2.ground_truth_prior[ids], "gt": y, "resp_x1": ctl.response[ids],
                        "resp_x2": x2.response[ids], "ratio": ratio, "outcome": cls})
    out.to_csv(f"{RES}/val_counterfactual.csv")
    print(out[["video_id", "cat", "kind", "prior", "gt", "resp_x1", "resp_x2", "ratio", "outcome"]].to_string())


def cmd_test(jsonl):
    t = add_keys(load_test())
    x = pd.DataFrame([json.loads(l) for l in open(jsonl, encoding="utf-8")])
    x.index = x.original_id.astype(int)
    assert not x.index.duplicated().any()
    assert set(x.index) == set(t.index)
    x = x.loc[t.index]
    assert (x.video_id.values == t.video_id.str.strip().values).all()
    assert (x.question.values == t.question.str.strip().values).all()
    v = pd.to_numeric(x.parsed_value, errors="coerce")
    write_official(t, v, f"{R}/preds/test_caw9b.csv")
    raw = t[[c for c in OUT_COLS if c != "parsed_value"]].copy()
    raw["parsed_value"] = v.values
    for c in ["response", "finish_reason", "num_generated_tokens", "num_prompt_tokens", "video_tensor_shape",
              "meta_total_num_frames"]:
        raw[c] = x[c].values
    raw.to_csv(f"{R}/preds/test_caw9b_raw.csv", index=False)
    inv = pos(v).isna()
    print('')
    print("   video tensor shapes:", x.video_tensor_shape.astype(str).value_counts().head(8).to_dict())
    print("   invalid by category:", t.cat[inv].value_counts().to_dict())
    if inv.any():
        print(pd.DataFrame({"row": t.index[inv], "video": t.video_id[inv], "q": t.question[inv].str[:60],
                            "response": x.response[inv].str[:60]}).head(40).to_string(index=False))
    asked, got, ukind = unit_table(t, x.response.astype(str))
    print('t107', ukind.value_counts().to_dict())
    odd = ukind.isin(["different unit", "other text"])
    if odd.any():
        print(pd.DataFrame({"row": t.index[odd], "asked": asked[odd], "got": got[odd],
                            "response": x.response[odd].str[:50]}).head(40).to_string(index=False))
    h = pd.read_csv(f"{R}/preds/test_geo_v1_raw.csv").set_index("Unnamed: 0").h1_value.abs().reindex(t.index)
    pv = pos(v)
    rej = pv.notna() & sane(pv, h).isna()
    print('')
    print('')
    vv = add_keys(load_val())
    key = vv.drop_duplicates(["vid", "question"]).set_index(["vid", "question"])["Unnamed: 0"]
    vid_q = pd.MultiIndex.from_arrays([t.vid, t.question])
    t["val_id"] = key.reindex(vid_q).values
    dup = t.duplicated(["vid", "question"], keep=False)
    ov = t[t.val_id.notna() & ~dup]
    o = vv.loc[ov.val_id.astype(int)].copy()
    o["parsed_value"] = v[ov.index].values
    p = write_official(o, o.parsed_value, f"{R}/preds/val124_from_test_caw9b.csv")
    so = score_official(p, "val124_from_test_caw9b")
    vcaw = pos(pd.read_csv(f"{R}/preds/val_caw9b.csv").set_index("Unnamed: 0").parsed_value)
    same = np.isclose(vcaw.reindex(o.index).values, pos(o.parsed_value).values, rtol=1e-9)
    sv = score_official(write_official(o.assign(parsed_value=vcaw.reindex(o.index).values),
                                       vcaw.reindex(o.index).values, f"{R}/caw/out/val124_valrun_caw9b.csv"),
                        "val124_valrun_caw9b")
    print('')


PAPER = {"S2": 0.550, "D2": 0.529, "S3": 0.556, "D3": 0.581, "MRA": 0.554}


def cmd_cmpenv(dir_a, dir_b, la="env030", lb="env0191"):
    d = add_keys(load_val())

    def load(dr):
        rc = pd.read_csv(f"{dr}/9b.csv")
        rc.index = rc[rc.columns[0]].astype(int)
        return pd.to_numeric(rc.parsed_value, errors="coerce").reindex(d.index), rc.raw_response.reindex(d.index).astype(str)

    va, ra = load(dir_a)
    vb, rb = load(dir_b)
    oa = score_official(f"{dir_a}/9b.csv", f"caw9b_release_csv_{la}")
    ob = score_official(f"{dir_b}/9b.csv", f"caw9b_release_csv_{lb}")
    print('')
    print('')
    print("== paper (Table 1, Code-as-World-VL-9B): " + " ".join(f"{k} {v:.3f}" for k, v in PAPER.items()))
    for lab, o in [(la, oa), (lb, ob)]:
        print(f"   {lab} minus paper: " + " ".join(f"{k} {o[k] - PAPER[k]:+.3f}" for k in ["MRA", "S2", "D2", "S3", "D3"]))
    sa = pd.Series(row_mra(va, d.y), index=d.index)
    sb = pd.Series(row_mra(vb, d.y), index=d.index)
    same = pd.Series(np.isclose(va, vb, rtol=1e-9), index=d.index)
    print('')
    print('')
    print('')
    print("   " + " ".join(f"{k} {((sa - sb)[d.cat == k].sum() / (d.cat == k).sum()):+.4f}" for k in CATS))
    diff = ~same
    if diff.any():
        t = pd.DataFrame({"video": d.video_id, "cat": d.cat, "q": d.question.str[:55], "gt": d.y,
                          la: ra.str.strip(), lb: rb.str.strip(), f"mra_{la}": sa, f"mra_{lb}": sb})[diff]
        print(t.to_string())
        t.to_csv(f"{RES}/val_env_compare_{la}_{lb}.csv")


def cmd_submit(name, out="sub_caw_fusion"):
    t = add_keys(load_test())
    caw = pd.read_csv(f"{R}/preds/test_caw9b.csv").set_index("Unnamed: 0").parsed_value
    M = methods(t, "test", caw)
    v, s = recipe(M, name)
    assert v.notna().all() and (v.abs() > 0).all()
    os.makedirs(f"{R}/submissions", exist_ok=True)
    p0 = write_official(t, v.values, f"{R}/submissions/{out}.csv")
    o = pd.read_csv(p0)
    o1 = o.rename(columns={"Unnamed: 0": "id"})
    o1["id"] = o1["id"] + 1
    o1.to_csv(f"{R}/submissions/{out}_id1.csv", index=False)
    det = t[["Unnamed: 0", "video_id", "question", "cat", "src"]].copy()
    det["answer"], det["answer_source"] = v.values, s.values
    for k in ["g", "c", "cs", "q", "h", "mode", "kind"]:
        det[k] = M[k].values
    det.to_csv(f"{R}/submissions/{out}_detail.csv", index=False)
    print('')
    chk = pd.read_csv(f"{R}/preds/test_geo_v1.csv")
    ok = (o["Unnamed: 0"].values == np.arange(len(t))).all() and (o.video_id.values == chk.video_id.values).all() \
        and (o.question.values == chk.question.values).all() and list(o.columns) == list(chk.columns)
    ok1 = (o1["id"].values == np.arange(1, len(t) + 1)).all() and list(o1.columns) == ["id"] + list(chk.columns[1:])
    print('')
    print('')
    print(pd.crosstab(t.cat, s).to_string())
    diff = ~np.isclose(v.values, chk.parsed_value.values, rtol=1e-9)
    print('')
    gv = recipe(M, "geo_v1")[0]
    print('')
    vv = add_keys(load_val())
    key = vv.drop_duplicates(["vid", "question"]).set_index(["vid", "question"])["Unnamed: 0"]
    t["val_id"] = key.reindex(pd.MultiIndex.from_arrays([t.vid, t.question])).values
    dup = t.duplicated(["vid", "question"], keep=False)
    ov = t[t.val_id.notna() & ~dup]
    oo = vv.loc[ov.val_id.astype(int)].copy()
    others = [(n, recipe(M, n)[0]) for n in ["geo_v1", "a_caw_h1", "b", "c1", "c2", "d_r2_gm_route"] if n != name]
    for lab, val in [(f"{out} ({name})", v)] + others:
        tag = re.sub(r"[^A-Za-z0-9_.]+", "_", lab)
        p = write_official(oo, val[ov.index].values, f"{R}/caw/out/val124/{tag}.csv")
        print('')


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "val":
        cmd_val(*sys.argv[2:])
    elif cmd == "cf":
        cmd_cf(*sys.argv[2:])
    elif cmd == "test":
        cmd_test(sys.argv[2])
    elif cmd == "cmpenv":
        cmd_cmpenv(*sys.argv[2:])
    elif cmd == "submit":
        cmd_submit(*sys.argv[2:])
    else:
        sys.exit(__doc__)
