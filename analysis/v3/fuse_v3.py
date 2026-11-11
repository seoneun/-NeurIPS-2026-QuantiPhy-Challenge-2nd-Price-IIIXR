#!/usr/bin/env python
import json, os, sys
import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/caw")
sys.path.insert(0, f"{R}/tools")
import analyze_caw as A
import analyze_v2caw as V
from qp_common import OUT_COLS

pd.set_option("display.width", 250)
SPLIT, TD, TV, PFX = "val", "", "", "geo_v3x"
RULES = ["d", "d_v", "d_vw", "q", "gm", "v_d"]


def methods2(d, split, caw):
    g = pd.read_csv(f"{R}/preds/{split}_{V.V2}_raw.csv").set_index("Unnamed: 0").reindex(d.index)
    assert (g.question.values == d.question.values).all()
    M = pd.DataFrame(index=d.index)
    M["h"] = pd.to_numeric(g.h1_value, errors="coerce").abs()
    M["g"] = pd.to_numeric(g.geo_value, errors="coerce").where(g.answer_source == "meas")
    M["q"] = A.sane(A.pos(g.vlm_value), M.h)
    M["c"] = A.pos(caw.reindex(d.index))
    M["cs"] = A.sane(M.c, M.h)
    if os.environ.get("QP_SANE", "h1") in ("samekind", "h1only"):
        import consistency as C_
        sys.path.insert(0, f"{R}/geo")
        from rules import UNIT_SI
        info = C_.parse_info(split, d.index)
        dim_q = info.unit.map(lambda u: 0 if u in ("m", "cm", "mm", "km") else (2 if "^2" in str(u) else 1))
        dim_p = info.pkind.map({"size": 0, "speed": 1, "accel": 2})
        same = dim_q == dim_p
        PL = {0: (1e-4, 500.0), 1: (1e-4, 400.0), 2: (1e-3, 300.0)}
        c_si = M.c * info.unit.map(lambda u: UNIT_SI.get(u, np.nan))
        lo, hi = dim_q.map(lambda k: PL[k][0]), dim_q.map(lambda k: PL[k][1])
        plaus = (c_si >= lo) & (c_si <= hi)
        if os.environ.get("QP_SANE") == "samekind":
            M["cs"] = M.cs.where(same, M.c.where(plaus))
        else:
            M["cs"] = M.cs.where(~(~same & M.cs.isna() & M.g.isna() & plaus), M.c)
    U, si = V.dist_bound(split, d.index)
    with np.errstate(invalid="ignore"):
        M["tri_rej"] = (M.cs * si > V.TRI * U).fillna(False) & M.cs.notna()
    M["cs"] = M.cs.where(~M.tri_rej)
    meas = M.g.notna()
    M["mode"] = g.meas_mode.fillna("").astype(str).where(meas, "")
    M["kind"] = g.meas_kind.fillna("").astype(str)
    M["why"] = g.meas_why.fillna("").astype(str)
    M["file"] = pd.to_numeric(g.parsed_value, errors="coerce")
    M["w1"] = meas & M["mode"].str.contains("fovdefault|samedepth")
    M["w2"] = M.w1 | (meas & M["mode"].str.contains("tdepth=prior"))
    M["w3"] = meas & ~M["mode"].isin(V.STRONG)
    M["lab3d"] = (d.src == "lab") & (d.video_type.str[1] == "3") & M.kind.isin(A.MOTION) & meas
    return M


def load_raw(tag):
    return pd.read_csv(f"{R}/preds/{SPLIT}_geo{tag}_raw.csv").set_index("Unnamed: 0")


def qualifier_rows(index):
    out = pd.Series(False, index=index)
    for ln in open(f"{R}/geo/out/parse_{SPLIT}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln)
        if x["id"] not in out.index:
            continue
        f = False
        for ent in (x.get("target"), x.get("target2")):
            if not ent or not ent.get("phrase"):
                continue
            w = [k for k in (ent.get("where") or []) if k != "#second"]
            raw = (ent.get("raw") or "").strip().lower()
            if w or ent.get("rel") or (raw and len(raw.split()) > len(ent["phrase"].split())):
                f = True
        out[x["id"]] = f
    return out


def merge(rd, rv, rule, qual):
    gd = pd.to_numeric(rd.geo_value, errors="coerce")
    gv = pd.to_numeric(rv.geo_value, errors="coerce").reindex(gd.index)
    md = rd.meas_mode.fillna("").astype(str)
    mv = rv.meas_mode.reindex(gd.index).fillna("").astype(str)
    both = gd.notna() & gv.notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(gd / gv))
    if rule == "d":
        g, src = gd, pd.Series(np.where(gd.notna(), "d", ""), index=gd.index)
    elif rule in ("d_v", "d_vw"):
        g = gd.where(gd.notna(), gv)
        src = pd.Series(np.where(gd.notna(), "d", np.where(gv.notna(), "v", "")), index=gd.index)
    elif rule == "v_d":
        g = gv.where(gv.notna(), gd)
        src = pd.Series(np.where(gv.notna(), "v", np.where(gd.notna(), "d", "")), index=gd.index)
    elif rule == "gm":
        g = pd.Series(np.where(both, np.sqrt(gd * gv), np.where(gd.notna(), gd, gv)), index=gd.index)
        src = pd.Series(np.where(both, "gm", np.where(gd.notna(), "d", np.where(gv.notna(), "v", ""))), index=gd.index)
    elif rule == "q":
        takev = both & (lr > np.log(1.25)) & qual
        g = gd.where(gd.notna() & ~takev, gv)
        src = pd.Series(np.where(takev, "v_q", np.where(gd.notna(), "d", np.where(gv.notna(), "v", ""))), index=gd.index)
    else:
        raise ValueError(rule)
    mode = pd.Series(np.where(src.str.startswith("v"), mv, md), index=gd.index)
    mode = mode.where(src != "gm", md)
    if rule == "d_vw":
        mode = mode.where(src != "v", "vlm_" + mode)
    out = rd.copy()
    vlm = pd.to_numeric(rd.vlm_value, errors="coerce")
    h1 = pd.to_numeric(rd.h1_value, errors="coerce").abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        vok = np.isfinite(vlm) & (vlm > 0) & (np.abs(np.log10(vlm / h1)) <= 2)
    out["geo_value"] = g
    out["meas_valid"] = g.notna()
    out["answer_source"] = np.where(g.notna(), "meas", np.where(vok, "vlm", "h1"))
    out["parsed_value"] = np.where(g.notna(), g, np.where(vok, vlm, h1))
    out["meas_mode"] = mode.where(g.notna(), np.nan)
    out["merge_src"] = src
    return out


def main():
    rd, rv = load_raw(TD), load_raw(TV)
    qual = qualifier_rows(rd.index)
    caw_name = "val_caw9b" if SPLIT == "val" else "test_caw9b"
    caw = pd.to_numeric(pd.read_csv(f"{R}/preds/{caw_name}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce")
    d = A.add_keys(A.load_val() if SPLIT == "val" else A.load_test())
    outs = {}
    for rule in RULES:
        o = merge(rd, rv, rule, qual)
        name = f"{PFX}_{rule}"
        o.to_csv(f"{R}/preds/{SPLIT}_{name}_raw.csv")
        o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{SPLIT}_{name}.csv", index=False)
        outs[rule] = o
        print('')
    res, S = [], {}
    for rule in RULES:
        V.V2 = f"{PFX}_{rule}"
        M = methods2(d, SPLIT, caw)
        for rec in ["geo", "b", "d_r2_gm_c1w3"]:
            v, s = (pd.to_numeric(outs[rule].parsed_value, errors="coerce").reindex(d.index), None) if rec == "geo" else V.recipe2(M, rec)
            S[(rule, rec)] = v
    if SPLIT == "val":
        y = d.y
        ref_geo = pd.to_numeric(pd.read_csv(f"{R}/preds/val_geo_v2_abcdef.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").reindex(d.index)
        V.V2 = "geo_v2_abcdef"
        Mref = V.methods2(d, "val", caw)
        ref_f, _ = V.recipe2(Mref, "d_r2_gm_c1w3")
        sref_f = pd.Series(A.row_mra(ref_f, y), index=d.index)
        sref_g = pd.Series(A.row_mra(ref_geo, y), index=d.index)
        print('')
        for (rule, rec), v in S.items():
            sc = pd.Series(A.row_mra(v, y), index=d.index)
            h, m = A.macro(sc, d.cat)
            base = sref_g if rec == "geo" else sref_f
            lo, hi, p = V.boot2(sc, base, d.cat, d.vid)
            res.append(dict(rule=rule, recipe=rec, MRA=round(h, 4), **{k: round(x, 4) for k, x in m.items()},
                            vs_ref=round(h - A.macro(base, d.cat)[0], 4), ci=f"[{lo:+.3f}, {hi:+.3f}]"))
        T = pd.DataFrame(res)
        print(T.to_string(index=False))
        names = RULES
        vals = {r: S[(r, "d_r2_gm_c1w3")] for r in names}
        Ssc = {r: pd.Series(A.row_mra(vals[r], y), index=d.index) for r in names}
        held, picks = V.lovo_select(names, vals, Ssc, d)
        hs = pd.Series(A.row_mra(held, y), index=d.index)
        print('')
        best = T[T.recipe == "d_r2_gm_c1w3"].sort_values("MRA").iloc[-1]
        print("best fused rule:", best.rule)
    else:
        print('')
        c = A.pos(caw.reindex(d.index))
        for rule in RULES:
            g = pd.to_numeric(outs[rule].geo_value, errors="coerce").reindex(d.index)
            both = g.notna() & c.notna()
            with np.errstate(divide="ignore", invalid="ignore"):
                ag = (np.abs(np.log(g / c)) <= np.log(2))[both]
            print('')


if __name__ == "__main__":
    SPLIT, TD, TV = sys.argv[1], sys.argv[2], sys.argv[3]
    PFX = sys.argv[4] if len(sys.argv) > 4 else "geo_v3x"
    main()
