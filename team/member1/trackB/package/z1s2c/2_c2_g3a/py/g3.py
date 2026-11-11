import os, sys, re, math, json, numpy as np, pandas as pd
sys.path.insert(0, os.environ["QP_WORK"] + "/h")
import harness as H
OD = sys.argv[1]; ROWS = H.ROWS
C9U = f"{OD}/pred_t128_id1.csv"
BASE = {sp: pd.read_csv(f"{OD}/series/pred_t127_{sp}.csv", index_col=0)[sp] for sp in ("test", "val")}
CF = {sp: {int(k): v for k, v in json.load(open(f"{OD}/c9u/cf_{sp}.json")).items()} for sp in ("test", "val")}
for sp in ("test", "val"):
    for i, e in CF[sp].items():
        BASE[sp][i] = float(e["value"])
PAT = {nm: {sp: {int(k): v for k, v in json.load(open(f"{OD}/patches/r36_BUILD_{nm}_{sp}.json")).items()} for sp in ("test", "val")} for nm in ("g_AB", "g_A")}

TABLE = [
    (r"\bhelicopters?\b", dict(length=(8, 30), width=(1.2, 18), height=(2.5, 6), any=(1.2, 30))),
    (r"\b(air ?planes?|aeroplanes?|aircraft|jets?|planes?)\b", dict(length=(6, 75), width=(8, 80), height=(2, 20), any=(1.5, 80))),
    (r"\bfire ?trucks?\b|\bbus(es)?\b|\btrucks?\b", dict(length=(6, 14), width=(2.2, 2.6), height=(2.5, 4), any=(2.2, 14))),
    (r"\b(telephone|phone) booths?\b", dict(height=(2.1, 2.6), any=(0.8, 2.6))),
    (r"\b(billiard|pool|snooker) tables?\b", dict(length=(2.0, 2.9), width=(1.0, 1.7), height=(0.7, 0.9), any=(0.7, 2.9))),
    (r"\bbicycle wheels?\b|\bbike wheels?\b", dict(diameter=(0.4, 0.75), any=(0.4, 0.75))),
    (r"\b(wheels?|tires?|tyres?)\b", dict(diameter=(0.1, 1.5), any=(0.05, 1.5))),
    (r"\b(bicycles?|bikes?)\b", dict(length=(1.4, 1.9), height=(0.8, 1.2), any=(0.3, 1.9))),
    (r"\b(cars?|sedans?|suvs?|taxis?)\b", dict(length=(3.5, 5.5), width=(1.5, 2.1), height=(1.2, 2.0), any=(1.2, 5.5))),
    (r"\b(sofas?|couch(es)?)\b", dict(height=(0.6, 1.0), any=(0.6, 3.2))),
    (r"\bchairs?\b", dict(any=(0.35, 1.2))),
    (r"\btables?\b|\bdesks?\b", dict(height=(0.4, 1.1), any=(0.4, 3.0))),
    (r"\blamp ?shades?\b", dict(diameter=(0.15, 0.6), any=(0.1, 0.6))),
    (r"\bvases?\b", dict(diameter=(0.06, 0.4), height=(0.1, 0.8), any=(0.06, 0.8))),
    (r"\btypewriters?\b", dict(any=(0.1, 0.5))),
    (r"\b(papers?|sheets?)\b", dict(any=(0.075, 0.42))),
    (r"\b(letters?|envelopes?)\b", dict(any=(0.09, 0.33))),
    (r"\bbottles?\b", dict(diameter=(0.04, 0.12), height=(0.1, 0.4), any=(0.04, 0.4))),
    (r"\broads?\b|\bstreets?\b", dict(width=(3, 30), any=(3, 30))),
    (r"\bsharks?\b", dict(length=(1, 6), any=(0.2, 6))),
    (r"\bturtles?\b", dict(length=(0.15, 1.8), any=(0.05, 1.8))),
    (r"\b(person|man|woman|human|boy|girl|people|player)\b", dict(height=(1.0, 2.0), any=(0.2, 2.0))),
    (r"\bdoors?\b", dict(height=(1.9, 2.5), width=(0.7, 1.2), any=(0.7, 2.5))),
]
DIMRE = re.compile(r"\b(?:outer |inner )?(length|width|height|diameter|thickness)\b(?:\s*\([a-z]+\))?\s+of\s+(?:the|a|an|one)?\s*(.+)", re.I)
STOP = re.compile(r"\s+(?:on|of|in|at|next|placed|under|that|from|with|which|near|beside|inside|while|when|between)\b|\?|,", re.I)

def unit_scale(q):
    ql = q.lower()
    if re.search(r"\bin (cm|centimet)", ql): return 100.0
    if re.search(r"\bin (mm|millimet)", ql): return 1000.0
    if re.search(r"\bin (km|kilomet)", ql): return 0.001
    return 1.0

def plaus(q, v):
    m = DIMRE.search(q or "")
    if not m: return None, "not a size question"
    dim = m.group(1).lower(); obj = STOP.split(m.group(2), maxsplit=1)[0].strip().lower()
    if re.search(r"\b(hand|text|letter(ing)?s? on|logo|label|pocket|screen)\b", obj) and not re.search(r"^(letter|envelope)", obj): return None, f"part '{obj}'"
    for rx, rng in TABLE:
        if re.search(rx, obj):
            lo, hi = rng.get(dim, rng["any"]); s = unit_scale(q); glo, ghi = lo / 2 * s, hi * 1.5 * s
            ok = bool(glo <= v <= ghi)
            return ok, f"{dim} of '{obj}' ~ [{glo:.4g}, {ghi:.4g}] -> {v:.4g} {'ok' if ok else 'OUT'}"
    return None, f"'{obj}' not in table"

def apply_rows(items):
    out = {}
    for sp, i, q, b, e in items:
        if e is None: out[(sp, i)] = (b, "no patch"); continue
        ok, why = plaus(q, float(e["value"]))
        out[(sp, i)] = (b, f"guard: {why}") if ok is False else (float(e["value"]), f"{e['rule']}: {why}")
    return out

def build(nm):
    items = [(sp, i, ROWS[sp][i]["question"], float(BASE[sp][i]), PAT[nm][sp].get(i)) for sp in ("test", "val") for i in sorted(PAT[nm][sp])]
    return items, apply_rows(items)

for nm, outname in (("g_AB", "pred_t131"), ("g_A", "pred_t132")):
    items, res = build(nm)
    patch = {"test": {}, "val": {}}; dropped = []
    for (sp, i), (val, why) in res.items():
        if why.startswith("guard:"): dropped.append((sp, i, PAT[nm][sp][i]["rule"], why))
        elif abs(math.log(val / float(BASE[sp][i]))) > 1e-12:
            patch[sp][str(i)] = dict(value=val, before=float(BASE[sp][i]), rule=PAT[nm][sp][i]["rule"], why=why)
    for sp in ("test", "val"): json.dump(patch[sp], open(f"{OD}/patches/r38_REBASE_{nm}_{sp}.json", "w"), indent=0)
    OUT = f"{OD}/{outname}_id1.csv"
    a = pd.read_csv(C9U, dtype=str, keep_default_na=False); b = a.copy(); pos = {int(x) - 1: j for j, x in enumerate(a.id)}
    for k, e in patch["test"].items():
        j = pos[int(k)]; b.loc[j, "parsed_value"] = repr(float(e["value"]))
    b.to_csv(OUT, index=False)
