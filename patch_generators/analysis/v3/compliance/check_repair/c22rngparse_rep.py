"""c22rngparse_rep.py BASE: per-question version of analysis/v3/ag/rul024 (scan.py, remeas.py v2 min, fusion.py base /
fix, mkpatch.py).
Rule (unchanged): the row's own question has a time range 'between A[s] and B[s]' that its parse missed or glued to the
target (scan.gate, imported read-only); a 2d_gself row (ruler_build m2: own-size self ruler, substituted-prior parse
geo/out/parse_<split>_v2_rl<k>.jsonl, k = the first of the row's rulers in analysis/v3/ruler_<split>.json whose key is
the target key) is re-measured with q.t1 / q.t2 from the match and the span stripped from the target.
Changes vs production:
  - re-measure: remeas.py measured ALL rows of the video (video-wide track pool) on the rl<k> parse, tracks v2, no QP_*
    env ('min'); here the row alone with the own-text track pool (grpC_pre_ownmeas.py var m, OWNT_TRK_M=v2,
    GRPC_ENVSET=min, GRPC_PARSE_TAG=_rl<k>; GRPC_FIXJSON = the parse fix); the production check 'base reproduces the geo
    raw' is kept for the row (rel 1e-4 against the clean geo value);
  - fusion: the clean strict fusion (grpC_strict.py) of the main table with the row's geo_value replaced (mode stays
    2d_gself, as fusion.py fix); base = BASE's main fusion xt_<B> instead of the replayed xt_Nzo.
Patch = re-measured rows whose main strict answer changes -> cache/tmp/ag_c22rngparse_patch_{test,val}.json."""
import os, sys, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
sys.path.insert(0, f"{R}/analysis/v3/ag/rul024")
import scan as S  # noqa: E402
SPAN = re.compile(r"\s*\bbetween\s+\d+(?:\.\d+)?\s*s?\s+and\s+\d+(?:\.\d+)?\s*s?\b", re.I)    # = remeas.py SPAN
MEAS = {}


def strip(p):
    if not p:
        return p
    return SPAN.sub("", p).strip() or p


def key(ent):                                                      # = geo/measure_v3.py Video.key
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def rl_rec(sp, k, i):
    for l in open(f"{R}/geo/out/parse_{sp}_v2_rl{k}.jsonl"):
        r = json.loads(l)
        if r["id"] == i:
            return r
    return None


def remeasure(sp, raw):
    C = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l)
        m, bad = S.gate(r)
        if m is not None and bad:
            C[r["id"]] = (float(m.group(1)), float(m.group(2)), r["video_id"])
    print(sp, "gated rows", C)
    res = {}
    if not C:
        return res
    RMAP = json.load(open(f"{R}/analysis/v3/ruler_{sp}.json"))
    base_env = {"OWNT_TRK_M": "v2", "GRPC_ENVSET": "min"}
    for i, (t1, t2, v) in sorted(C.items()):
        mode0 = str(raw.meas_mode.get(i))
        if mode0 != "2d_gself":
            print(f"  {i}: geo mode {mode0} - only 2d_gself rows are re-measured"); continue
        rul = RMAP[str(i)]["rulers"]
        for k in range(min(3, len(rul))):
            rr = rl_rec(sp, k, i)
            if rr is None or rul[k]["key"] != key(rr["target"]):
                print(f"  {i}: rl{k} ruler {rul[k]['key']} is not the target {key(rr['target']) if rr else None}"); continue
            fix = {sp: {str(i): {"q": dict(t1=t1, t2=t2, t=None, target=strip(rr["q"].get("target"))),
                                 "target": dict(phrase=strip(rr["target"].get("phrase")), raw=strip(rr["target"].get("raw")))}}}
            fp = f"{R}/cache/tmp/ag_c22rngparse_rep_fix.json"
            json.dump(fix, open(fp, "w"))
            e = dict(base_env, GRPC_PARSE_TAG=f"_rl{k}")
            b, _ = G.ownmeas(sp, "m", {v}, f"_grpCc22_rl{k}_base", e)
            n, dg = G.ownmeas(sp, "m", {v}, f"_grpCc22_rl{k}_new", dict(e, GRPC_FIXJSON=fp))
            x = dict(raw=float(raw.geo_value[i]), base=float(b.value.get(i, np.nan)), new=float(n.value.get(i, np.nan)),
                     mode=n["mode"].get(i), why=n.why.get(i), k=k, t1=t1, t2=t2, vid=v, fix=fix[sp][str(i)],
                     pool=dg.get(str(i), {}).get("pool", {}).get("why"))
            x["reproduced"] = bool(np.isfinite(x["new"]) and x["new"] > 0 and np.isfinite(x["base"]) and abs(x["base"] / x["raw"] - 1) < 1e-4)
            res[i] = x
            print(sp, i, x)
            break
    return res


def edit(sp, u, raw):
    MEAS[sp] = remeasure(sp, raw)
    ch = []
    for i, x in MEAS[sp].items():
        if not x["reproduced"]:
            continue
        raw.loc[i, "geo_value"] = x["new"]
        raw.loc[i, "meas_why"] = "rul024"
        if raw.answer_source.get(i) == "meas":
            raw.loc[i, "parsed_value"] = x["new"]
        ch.append(i)
    return ch


out = G.run("rul024", edit)
for sp, (n, b, bs, rows) in out.items():
    keep = G.gate(n, b, bs, rows)
    P = {}
    for i in keep:
        m = MEAS[sp][i]
        P[i] = {"value": float(n.answer[i]),
                "why": (f"rul024: time range without unit parsed as t1={m['t1']:g}, t2={m['t2']:g} (span stripped from the "
                        f"target); geo {m['mode']} rl{m['k']} re-measured on the row alone over [{m['t1']:g}, {m['t2']:g}] "
                        f"{n.g[i]:.4g} -> strict fusion {n.answer_source[i]}; was {b.answer_source[i]} {b.answer[i]:.4g} (whole-clip window)")}
        print(sp, i, P[i])
    print(f"[rul024 {sp}] re-measured {len(MEAS.get(sp, {}))}, geo rows replaced {len(rows)}, written {len(keep)}")
    G.write_patch("rul024", sp, P)
json.dump({sp: {str(k): v for k, v in d.items()} for sp, d in MEAS.items()}, open(f"{R}/cache/tmp/ag_c22rngparse_rep_meas.json", "w"),
          indent=1, default=str)
