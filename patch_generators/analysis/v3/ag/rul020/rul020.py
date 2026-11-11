"""rul020: stated size prior copied from a linked simulation template, gated by focal-free witnesses.

Rule (all code conditions on inputs; labels are never read here):
 1. link simulation bases (video_id without _segmented) whose question sets share >= 2 questions, then take the
    transitive closure of that graph;
 2. P = a prior line '<dim> of <obj> = x unit' (dim in diameter|length|width|height|radius|thickness|wingspan|size)
    stated in a linked base; the row must ask 'what is the [body/total/...] <dim> of <obj> in <unit>' with the same
    dim and obj; all linked values must agree within 2 %;
 3. skip the row if its own base states that size (the pipeline already uses it);
 4. witness gate: rows in OTHER bases with the identical question text and identical prior text (spaces removed),
    category D2/S2, meas_mode 2d_ratio and geo > 0; median(geo / P) in (1/1.15, 1.15) and every ratio in (0.5, 2);
 5. answer P in the question unit.
writes cache/tmp/ag_c21wsp_patch_{test,val}.json  ({row_id: {"value", "why"}})"""
import sys, re, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SI = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
DIMS = r"(diameter|length|width|height|radius|thickness|wingspan|size)"
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"


def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", x.lower()).split() if w not in STOP)


def prior_items(p):
    """[(dim, obj, value_SI)] for every '<...dim...> of <obj> = x unit' line of a prior text"""
    out = []
    for line in str(p).split("\n"):
        m = re.match(r"\s*(.+?)\s*[=~]\s*([-+]?\d*\.?\d+)\s*([a-z/\^0-9]+)?", line.strip().lower())
        if not m:
            continue
        u = (m.group(3) or "").strip(); dm = re.search(DIMS, m.group(1))
        if not dm or u not in SI:
            continue
        out.append((dm.group(1), nobj(re.sub(DIMS, " ", m.group(1))), float(m.group(2)) * SI[u]))
    return out


def q_item(q):
    pre = r"(?:(?:body|total|overall|full|whole) )?"
    m = re.search(r"what is the " + pre + DIMS + r"(?: \([^)]*\))? of (?:the |a |each )?(.+?) in (meters|meter|m|cm|mm)\b",
                  str(q).lower())
    return (m.group(1), nobj(m.group(2)), m.group(3)) if m else None


def links(d, minq=2, trans=True):
    base = d.video_id.astype(str).str.replace("_segmented", "", regex=False)
    qs = d.groupby(base).question.apply(lambda z: set(z.astype(str).str.strip().str.lower()))
    sb = [b for b in qs.index if b.startswith("simulation_")]
    lk = {b: [c for c in sb if c != b and len(qs[b] & qs[c]) >= minq] for b in sb}
    if trans:
        comp = {}
        for b0 in sb:
            if b0 in comp:
                continue
            st, seen = [b0], {b0}
            while st:
                x = st.pop()
                for y in lk.get(x, []):
                    if y not in seen:
                        seen.add(y); st.append(y)
            for x in seen:
                comp[x] = seen
        lk = {b: sorted(comp[b] - {b}) for b in sb}
    return lk


def rule(d, cat, minq=2, trans=True, witness=True, link_override=None, verbose=False):
    """d: geo raw csv of one split (index = row id), cat: Series of categories. returns {row_id: (P, info)}"""
    base = d.video_id.astype(str).str.replace("_segmented", "", regex=False)
    issim = base.str.startswith("simulation_")
    pri = {b: prior_items(g.ground_truth_prior.iloc[0]) for b, g in d[issim].groupby(base[issim])}
    lk = link_override if link_override is not None else links(d, minq, trans)
    qn = d.question.astype(str).str.strip().str.lower()
    pn = d.ground_truth_prior.astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    W = pd.DataFrame({"q": qn, "p": pn, "base": base, "mode": d.meas_mode.astype(str),
                      "g": pd.to_numeric(d.geo_value, errors="coerce"), "cat": cat.reindex(d.index)})
    W = W[W["cat"].isin(["D2", "S2"]) & (W["mode"] == "2d_ratio") & (W.g > 0)]
    out = {}
    for i in d.index[issim]:
        it = q_item(d.question[i])
        if not it:
            continue
        dim, obj, unit = it; b = base[i]
        if any(dd == dim and oo == obj for dd, oo, _ in pri.get(b, [])):
            continue                                   # own base states it
        src = [(c, val) for c in lk.get(b, []) for dd, oo, val in pri.get(c, []) if dd == dim and oo == obj]
        if not src:
            continue
        vals = [v for _, v in src]
        if max(vals) / min(vals) > 1.02:
            continue
        P = float(np.median(vals)) / SI[unit]
        info = {"src": sorted({c for c, _ in src}), "wit": []}
        if witness:
            w = W[(W.q == qn[i]) & (W.p == pn[i]) & (W.base != b)]
            if len(w) == 0:
                continue
            r = w.g.values / P
            info["wit"] = [(int(j), w.base[j], round(float(w.g[j]), 3)) for j in w.index]
            if not (1 / 1.15 < np.median(r) < 1.15 and r.min() > 0.5 and r.max() < 2):
                continue
        out[i] = (P, info)
    return out


def load(sp, sub="sub_xt_Nzn"):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")
    det = pd.read_csv(f"{R}/submissions/{sub}_{'detail' if sp == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    assert len(raw) == len(det) and (raw.video_id.values == det.video_id.values).all()
    raw.index = det.index
    return raw, det


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzn"
    for sp in ("test", "val"):
        raw, det = load(sp, sub)
        fired = rule(raw, det["cat"])
        patch = {}
        for i, (P, info) in sorted(fired.items()):
            if abs(det.answer[i] - P) <= 1e-6 * max(1.0, abs(P)):
                continue
            patch[str(int(i))] = {"value": float(P), "why": f"rul020: stated size prior from linked sim template(s) "
                                  f"{','.join(info['src'])}, confirmed by identical-prior D2/S2 2d_ratio geo "
                                  f"{[w[2] for w in info['wit']]}"}
            print(sp, i, det.video_id[i], det["cat"][i], raw.question[i][:60], "| old", round(det.answer[i], 3),
                  det.answer_source[i], "-> new", round(P, 3), "| src", info["src"], "| wit", info["wit"])
        json.dump(patch, open(f"{R}/cache/tmp/ag_c21wsp_patch_{sp}.json", "w"), indent=1)
        print("wrote", sp, len(patch))
