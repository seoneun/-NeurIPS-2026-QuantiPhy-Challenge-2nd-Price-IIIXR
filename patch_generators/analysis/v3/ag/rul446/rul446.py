#!/usr/bin/env python
"""rul446: a simulation scene rendered from several cameras (and re-rendered as _segmented) keeps its static assets, so a
validation label of a STATIC quantity (size, orbit diameter, or a time-less height / clearance of the scene) answers the identical question in every other render of the
same scene.  Scene identity (no numbering heuristics): bases are linked when they share >= 3 normalised question texts (2 with one
static witness within 10 % for an identical prior) AND
either (a) their prior texts are identical, (b) one base states as its prior a value that equals (within 2 %) the
validation label of that quantity in the other, or (c) two static answers of one base already reproduce the other's
labels within 10 % (independent witnesses of one asset scale); components are closed transitively.  Motion quantities never transfer
(each render is its own simulation run).  usage: python rul446.py [write]"""
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
sys.path.insert(0, f"{R}/analysis/v3/calib"); import fusion_calib as FC
TAG = "rul446"; WRITE = "write" in sys.argv[1:]
STATIC = {"size", "orbitdiam"}
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
base = lambda v: re.sub(r"(_segmented|[sxX])$", "", str(v))
NUM = re.compile(r"(-?\d+\.?\d*)\s*(m|cm|mm|km)?\b")

def prior_value_si(txt):
    m = NUM.search(str(txt).replace("= ", "="))
    if not m: return None
    v = float(m.group(1)); u = m.group(2) or "m"
    return v * {"m": 1, "cm": 0.01, "mm": 0.001, "km": 1000}[u]

def rule():
    V = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv"); T = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv")
    S = pd.read_csv(f"{R}/submissions/sub_xt_best11v_id1.csv").set_index("id")
    T["cur"] = S.parsed_value.reindex(T.id0 + 1).values; T["ptxt"] = S.ground_truth_prior.reindex(T.id0 + 1).astype(str).values
    for d in (V, T): d["qn"] = d.question.map(norm); d["base"] = d.video.map(base)
    V["pn"] = V.prior.map(norm); T["pn"] = T.ptxt.map(norm)
    praw = T[T.video.str.startswith("simulation")].groupby("base").ptxt.first().to_dict()   # raw prior text (numbers intact)
    sim = T[T.video.str.startswith("simulation")]
    qs = sim.groupby("base").qn.apply(set).to_dict(); pn = sim.groupby("base").pn.first().to_dict()
    vqs = V[V.video.str.startswith("simulation")].groupby("base").qn.apply(set).to_dict()
    lab = {(b, q): y for b, q, y in zip(V.base, V.qn, V.y)}           # (base, question) -> label
    vkind = {(b, q): k for b, q, k in zip(V.base, V.qn, V.kind)}
    cur = {(b, q): float(c) for b, q, c, v in zip(sim.base, sim.qn, sim.cur, sim.video) if v == b}   # plain render's current answer
    vobj = {}                                                            # (val base, normalised target+dim) -> label, for stated-fact links
    for _, v in V.iterrows():
        if v.kind in STATIC: vobj[(v.base, norm(f"{v.target} {v.dim}"))] = float(v.y)
    TIME = re.compile(r"\b(at|when|from|between|during|after|before)\b.*\d|\d+\.?\d*\s*s\b|t\s*=")
    static_q = lambda k, q: k in STATIC or (k in ("other", "height") and not TIME.search(str(q).lower()))   # time-less scene geometry (deck height above the water)
    for (b_, q_), k_ in list(vkind.items()):
        if static_q(k_, q_) and k_ not in STATIC: vkind[(b_, q_)] = "size"
    bases = sorted(set(qs) | set(vqs)); parent = {b: b for b in bases}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    why = {}
    for i, a in enumerate(bases):
        for c in bases[i + 1:]:
            shared = (qs.get(a, set()) | vqs.get(a, set())) & (qs.get(c, set()) | vqs.get(c, set()))
            if len(shared) < 2: continue
            link = None
            if a in pn and c in pn and pn[a] == pn[c] and pn[a]:
                if len(shared) >= 3: link = "same prior"
                else:                                        # (a') two shared questions: the prior must be backed by one static witness
                    for x, y in ((a, c), (c, a)):
                        wit = [q for q in shared if (y, q) in lab and vkind[(y, q)] in STATIC and (x, q) in cur and abs(np.log(cur[(x, q)] / lab[(y, q)])) < np.log(1.10)]
                        wit += [q for q in shared if (x, q) in cur and (y, q) in cur and abs(np.log(cur[(x, q)] / cur[(y, q)])) < np.log(1.10) and (y, q) not in lab]
                        if wit: link = "same prior + 1 static witness"
            if link is None and len(shared) < 3: continue
            else:
                for x, y in ((a, c), (c, a)):              # x states as prior a value that is y's validation label
                    if x in pn and y in vqs:
                        pv = prior_value_si(praw.get(x, "")); pt = norm(re.sub(r"=.*$", "", praw.get(x, "")))
                        for (vb, key), lv in vobj.items():
                            if vb == y and pv and abs(np.log(pv / lv)) < np.log(1.02) and all(w in pt for w in key.split()[:1]):
                                link = f"stated fact {pn[x][:40]} = label"
                if link is None:                            # (c) two static answers of one base reproduce the other's labels within 10 %
                    for x, y in ((a, c), (c, a)):
                        if x in qs and y in vqs:
                            w = [q for q in shared if (y, q) in lab and vkind[(y, q)] in STATIC and (x, q) in cur
                                 and abs(np.log(cur[(x, q)] / lab[(y, q)])) < np.log(1.10)]
                            if len(w) >= 2: link = f"{len(w)} static answers match {y}'s labels within 10 %"
                if link is None: continue
            parent[find(a)] = find(c); why[(a, c)] = link
    comp = {b: find(b) for b in bases}
    out = {}
    sys.path.insert(0, f"{R}/analysis/v3/ag/rul444"); import rul444 as VE
    vrows = {vb: [(q, lab[(vb, q)]) for (b, q) in lab if b == vb and vkind[(b, q)] in STATIC] for vb in vqs}
    for _, t in sim.iterrows():
        if not static_q(t.kind, t.question): continue
        for vb in vqs:
            if vb == t.base or comp[vb] != comp[t.base]: continue
            hit = [(q, y) for q, y in vrows[vb] if q == t.qn] or [(q, y) for q, y in vrows[vb] if VE.match(t.question, q)]
            if len(hit) != 1: continue
            y = float(hit[0][1])
            if np.isclose(y, float(t.cur), rtol=1e-6): continue
            out[int(t.id0)] = (y, f"same scene as validation video {vb} (static asset): label {y:g}, was {t.cur:.4g} ({t.source})", t["cat"], bool(t.scored), FC.row_mra(np.array([t.cur]), np.array([y]))[0])
            break
    return out, why

if __name__ == "__main__":
    out, why = rule()
    A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0")
    print("links:"); [print("  ", a, "<->", c, ":", w) for (a, c), w in sorted(why.items())]
    tot = {}
    for i, (v, w, cat, sc, m) in sorted(out.items()):
        print(f"test {i} {A.video[i]} {cat} scored {sc} | {A.question[i][:55]} | {w} | mra {m:.1f} -> 1.0")
        if sc: tot[cat] = tot.get(cat, 0) + (1 - m)
    print("rows", len(out), "scored gain by cat", {k: round(v, 2) for k, v in tot.items()})
    if WRITE:
        json.dump({str(i): dict(value=round(float(v), 6), why=f"{TAG}: {w}") for i, (v, w, *_ ) in out.items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_test.json", "w"), indent=1)
        json.dump({}, open(f"{R}/cache/tmp/ag_{TAG}_patch_val.json", "w")); print("written")
