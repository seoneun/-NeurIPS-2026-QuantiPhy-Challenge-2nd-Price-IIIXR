#!/usr/bin/env python
import json, sys
import numpy as np
import pandas as pd

import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/caw")
sys.path.insert(0, f"{R}/analysis/v3")
import analyze_caw as A
import analyze_v2caw as V
import fuse_v3 as F

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
GEO = "geo_v3x_d_v"


def parse_info(split, index):
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln)
        P[x["id"]] = x
    info = pd.DataFrame(index=index)
    info["tphrase"] = [((P.get(i) or {}).get("target") or {}).get("phrase") or "" for i in index]
    info["t2phrase"] = [((P.get(i) or {}).get("target2") or {}).get("phrase") or "" for i in index]
    info["pphrase"] = [((P.get(i) or {}).get("prior_obj") or {}).get("phrase") or "" for i in index]
    info["kind"] = [((P.get(i) or {}).get("q") or {}).get("kind") or "" for i in index]
    info["unit"] = [((P.get(i) or {}).get("q") or {}).get("unit") or "" for i in index]
    pri = [((P.get(i) or {}).get("priors") or [{}])[0] for i in index]
    info["pkind"] = [p.get("kind") for p in pri]
    info["pval_si"] = [p.get("value_si") for p in pri]
    return info


def c1_mask(d, g, info):
    sys.path.insert(0, f"{R}/geo")
    from rules import UNIT_SI
    bad = pd.Series(False, index=d.index)
    why = pd.Series("", index=d.index)
    gs = g * info.unit.map(lambda u: UNIT_SI.get(u, np.nan))
    for v, idx in d.groupby("vid").groups.items():
        ix = [i for i in idx if np.isfinite(gs[i])]
        for a_ in range(len(ix)):
            for b_ in range(a_ + 1, len(ix)):
                i, j = ix[a_], ix[b_]
                if info.kind[i] != info.kind[j] or info.tphrase[i] == info.tphrase[j] or not info.tphrase[i] or not info.tphrase[j]:
                    continue
                if abs(gs[i] - gs[j]) <= 1e-6 * max(abs(gs[i]), 1e-12):
                    bad[i] = bad[j] = True
                    why[i] = why[j] = "dup_value"
        for i in ix:
            if info.kind[i] == "size" and info.pkind[i] == "size" and info.pval_si[i] and info.tphrase[i] != info.pphrase[i] \
                    and abs(gs[i] - info.pval_si[i]) <= 1e-6 * info.pval_si[i]:
                bad[i] = True; why[i] = "equals_prior"
        sizes = [gs[i] for i in ix if info.kind[i] == "size"]
        if sizes:
            for i in ix:
                if info.kind[i] == "dist2" and gs[i] < 0.05 * max(sizes):
                    bad[i] = True; why[i] = "dist_tiny"
    return bad, why


def c2_apply(d, M, info, base, thr, nmin, act, dims="2"):
    is2d = d.video_type.str[1].isin(list(dims))
    dyn = d.inference_type.str[0] == "D"
    sz = info.kind == "size"
    ok = M.g.notna() & M.cs.notna() & is2d & dyn & sz
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(M.cs / M.g)
    st = pd.DataFrame(dict(vid=d.vid[ok], lr=lr[ok])).groupby("vid").lr.agg(["median", "size"])
    badv = st[(st["size"] >= nmin) & (st["median"].abs() > thr)]
    rows = d.vid.isin(badv.index) & M.g.notna() & is2d & dyn
    v = base.copy()
    if act == "rs":
        v[rows] = M.g[rows] * 10 ** d.vid[rows].map(badv["median"])
    elif act == "mix":
        ms = rows & sz & M.cs.notna()
        v[ms] = M.cs[ms]
        mo = rows & ~sz
        v[mo] = M.g[mo] * 10 ** d.vid[mo].map(badv["median"])
    else:
        m = rows & M.cs.notna()
        v[m] = M.cs[m]
    return v, rows, badv


def run(split):
    d = A.add_keys(A.load_val() if split == "val" else A.load_test())
    caw = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_caw9b.csv").set_index("Unnamed: 0").parsed_value, errors="coerce")
    V.V2 = GEO
    M = F.methods2(d, split, caw)
    info = parse_info(split, d.index)
    base, src = V.recipe2(M, "d_r2_gm_c1w3")
    bad, why = c1_mask(d, M.g, info)
    M1 = M.copy()
    M1["g"] = M.g.where(~bad)
    for w in ["w1", "w2", "w3", "lab3d"]:
        M1[w] = M[w] & ~bad
    v1, _ = V.recipe2(M1, "d_r2_gm_c1w3")
    out = {"base": base, "C1": v1}
    touched = {"C1": bad & M.g.notna()}
    for thr in [0.2, 0.3]:
        for nmin in [2, 3]:
            for act in ["rs", "caw"]:
                v, rows, badv = c2_apply(d, M1, info, v1, thr, nmin, act)
                k = f"C1+C2_t{thr}_n{nmin}_{act}"
                out[k], touched[k] = v, rows
    print('')
    if split == "test":
        ex = d[bad & M.g.notna()][["vid", "question"]].assign(why=why, g=M.g, c=M.cs).head(25)
        print(ex.assign(question=ex.question.str.slice(0, 60)).round(4).to_string())
    rows = []
    for k, v in out.items():
        r = dict(rule=k, rows=int(touched.get(k, pd.Series(False, index=d.index)).sum()),
                 videos=int(d.vid[touched.get(k, pd.Series(False, index=d.index))].nunique()))
        if split == "val":
            y = d.y
            sb, sv = pd.Series(A.row_mra(base, y), index=d.index), pd.Series(A.row_mra(v, y), index=d.index)
            h, m = A.macro(sv, d.cat)
            lo, hi, p = V.boot2(sv, sb, d.cat, d.vid)
            r.update(MRA=round(h, 4), **{c: round(x, 4) for c, x in m.items()}, vs_base=round(h - A.macro(sb, d.cat)[0], 4),
                     ci=f"[{lo:+.3f}, {hi:+.3f}]")
        else:
            r.update(cats=d.cat[touched[k]].value_counts().to_dict() if k in touched else {})
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    return out, d


if __name__ == "__main__":
    GEO = sys.argv[1] if len(sys.argv) > 1 else GEO
    for s in ["val", "test"]:
        run(s)
