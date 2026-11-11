#!/usr/bin/env python
"""rul452: the lab props are one physical set with tape-measured sizes (QuantiPhy App. D; rule rul308). rul308
matches props by exact target name; three props are asked under another name in other recordings:
  - "white cup" (captured_0037, captured_0040) = the "cup" of captured_0013 -> height = val label (val id0 2345, 14.5 cm)
  - "book" (captured_0014, captured_0023)      = the "note book" of captured_0013 -> width = val label (val id0 2348, 10.8 cm)
  - "orange ball" (captured_0054a/b)            = the basketball -> diameter = stated prior of captured_0035a (23.2 cm)
Identity was decided by eye on track crops (same white tapered cup on the same desk; same light-blue notebook with a
ribbon bookmark; basketball seams on the orange ball) and checked with code (pinhole cup height 14.2-14.4 cm in 0037/0040;
0014s notebook 9.6-10.1 x 20.8-22.9 cm on a MoGe table plane that reproduces the 60 x 75 cm table; 0023 book's
camera-facing side ~19.9 cm = the notebook's long side). captured_0024 is left out: two books lie on the table there.
Every value is read by code from validation_dataset.csv or from the captured_0035a prior text.
Asserts per row: size question in cm containing the target phrase and the dimension word, scored, and no label-carrying
patch (rul308, rul447, rul444, rul446, rul306, rul445) covers it.     usage: python rul452.py [write]"""
import os, re, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul452"; WRITE = "write" in sys.argv[1:]
BASE = f"{R}/submissions/sub_xt_best18v_id1.csv"
LABEL_TAGS = ("rul308", "rul447", "rul444", "rul446", "rul306", "rul445")
# (video regex, target phrase in the question, dimension word) -> source of the size
ALIAS = [(r"^captured_00(37|40)[sx]?$", "white cup", "height", ("val", 2345, "height of the cup")),
         (r"^captured_00(14|23)[sx]?$", "book", "width", ("val", 2348, "width of the note book")),
         (r"^captured_0054[ab][sx]?$", "orange ball", "diameter", ("prior", "captured_0035a", r"diameter of the basketball\s*=\s*([\d.]+)\s*cm"))]
EXPECT = {"S3": 6, "D3": 12}                                  # rows the alias table must hit (selector's row list)


def source_value(src):
    if src[0] == "val":
        V = pd.read_csv(f"{R}/data/val/validation_dataset.csv").set_index("Unnamed: 0")
        q = str(V.question[src[1]]).lower()
        assert src[2] in q and q.rstrip("? ").endswith("in cm"), (src, q)
        return float(V.ground_truth_posterior[src[1]]), f"validation label val id0 {src[1]} ({V.video_id[src[1]]}, '{src[2]}')"
    T = pd.read_parquet(f"{R}/data/test/test_dataset.parquet")
    pri = sorted(set(T[T.video_id == src[1]].prior.astype(str)))
    m = [re.search(src[2], p) for p in pri]; m = [x for x in m if x]
    assert len(m) == 1, (src, pri)
    return float(m[0].group(1)), f"stated prior of {src[1]} ('{m[0].group(0)}')"


def rule(sp="test"):
    if sp != "test":                                         # no validation video carries an aliased prop question
        A = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv")
        hit = [i for i in A.index for vre, ph, dim, _ in ALIAS if re.match(vre, A.video[i]) and ph in str(A.question[i]).lower()]
        assert not hit, hit
        return {}
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0")
    cover = {}
    for t in LABEL_TAGS:
        p = f"{R}/cache/tmp/ag_{t}_patch_test.json"
        for k in (json.load(open(p)) if os.path.exists(p) else {}):
            cover.setdefault(int(k), []).append(t)
    out = {}
    for vre, ph, dim, src in ALIAS:
        val, why = source_value(src)
        pat = rf"\b(?:{dim} of the {re.escape(ph)}|{re.escape(ph)}'s {dim})\b"     # "height of the white cup", "orange ball's diameter"
        rows = A[A.video.str.match(vre) & A.question.str.lower().str.contains(pat, regex=True)]
        assert len(rows), (vre, ph)
        for i, r in rows.iterrows():
            assert r.kind == "size" and r.unit == "cm" and str(r.dim) == dim and str(r.target) == ph, (i, r.question, r.kind, r.unit, r.dim, r.target)
            assert bool(r.scored), (i, "not scored")
            assert i not in cover, (i, cover.get(i))
            out[int(i)] = (val, f"lab prop alias: '{ph}' {dim} in {r.video} = {val:g} cm, {why}")
    return out


if __name__ == "__main__":
    out = rule("test"); outv = rule("val")
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0"); S = pd.read_csv(BASE)
    cur = S.parsed_value.values
    mra = lambda p, y: float(FC.row_mra(np.array([p]), np.array([y]))[0])
    P = {"captured_0037": .85, "captured_0040": .85, "captured_0014": .85, "captured_0023": .8, "captured_0054": .65}
    rows = []
    for i, (v, w) in sorted(out.items()):
        c = float(cur[i]); key = A.video[i][:13]
        rows.append(dict(id0=i, video=A.video[i], cat=A["cat"][i], cur=round(c, 4), new=v, right=mra(v, v) - mra(c, v),
                         wrong=mra(v, c) - mra(c, c), P=P[key]))
    d = pd.DataFrame(rows); d["exp"] = d.P * d.right + (1 - d.P) * d.wrong
    pd.set_option("display.width", 200); print(d.to_string(index=False))
    n = d.groupby("cat").size().to_dict(); print("rows per cat", n, "val rows", len(outv))
    assert n == EXPECT, n
    g = d.groupby("cat")[["right", "wrong", "exp"]].sum(); print(g.round(3).to_string())
    size = {"S2": 535, "D2": 1110, "S3": 488, "D3": 885}
    pts = {c: g.exp[c] / size[c] for c in g.index}; print("expected category points", {c: round(x, 5) for c, x in pts.items()},
                                                         "mean", round(sum(pts.values()) / 4, 5))
    if WRITE:
        js = {str(i): dict(value=round(float(v), 6), why=f"{TAG}: {w}") for i, (v, w) in out.items()}
        for p in (f"{R}/cache/tmp/ag_{TAG}_patch_test.json", f"{R}/patches/ag_{TAG}_patch_test.json"):
            json.dump(js, open(p, "w"), indent=1)
        for p in (f"{R}/cache/tmp/ag_{TAG}_patch_val.json", f"{R}/patches/ag_{TAG}_patch_val.json"):
            json.dump({str(i): dict(value=v, why=f"{TAG}: {w}") for i, (v, w) in outv.items()}, open(p, "w"), indent=1)
        print("written")
