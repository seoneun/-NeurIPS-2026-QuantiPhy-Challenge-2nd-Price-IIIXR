"""c30win_rep.py BASE: per-question version of analysis/v3/ag/rul044 (remeas.py, mkraw.py, fusion.py, mkpatch.py).
Pre-stage (TSV line before this one): analysis/v3/ag/rul044/winparse.py, unchanged (own question text: the time window
'from A to B' without the seconds unit is rewritten and re-parsed per row -> out/cand_<split>.json, out/parse_<split>_v2_win.jsonl).
Changes vs production:
  - re-measure (remeas.py measured every row of the video with MV.measure_video, video-wide track pool, joint time shift,
    video FPERSON / scene ruler): the row alone with grpC_pre_ownmeas.py (= owntracks/meas.py, the clean geo's own-text
    per-row measurement) on the current parse (base) and with the row's record from the window-fixed parse (new);
    two chains: m on v2sc tracks (production remeas.py) and v on v3ntmvSIL tracks (owntracks var v);
  - mkraw.py gate 'the base measurement reproduces the value the current answer is built from' (rel 1e-4) is applied per
    chain against the clean geo value (BASE has no rul424 patch, so the value is geo_value); the first reproducing chain
    (m, then v) is used; the other mkraw.py gates are kept (new finite > 0, measure.PLAUS, 2 decades from h1);
  - fusion: the clean strict fusion (grpC_strict.py) of the main table with the accepted rows replaced (production: the
    non-strict xt_Nzr replay with rul424 left out); base = BASE's main fusion xt_<B>.
Patch = accepted rows whose main strict answer changes -> cache/tmp/ag_c30win_patch_{test,val}.json.
Env C30W_CHAIN=m|v (check only, not for the build): use that chain without the reproduce gate."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
sys.path.insert(0, f"{R}/geo")
from rules import UNIT_SI  # noqa: E402
PLAUS = {0: (1e-4, 500.0), 1: (1e-4, 400.0), 2: (1e-3, 300.0)}     # geo/measure.py:551
OD = f"{R}/analysis/v3/ag/rul044/out"
ENVC = {"m": {"OWNT_TRK_M": "v2sc"}, "v": {}}
REC = {}


def remeasure(sp):
    cand = json.load(open(f"{OD}/cand_{sp}.json"))
    if not cand:
        return {}
    vids = {c["vid"] for c in cand.values()}
    rows = ",".join(sorted(cand))
    M = {}
    for c in ("m", "v"):
        b, _ = G.ownmeas(sp, c, vids, "_grpCc30win_base", ENVC[c])
        n, dg = G.ownmeas(sp, c, vids, "_grpCc30win_new", dict(ENVC[c], GRPC_ALT_PARSE=f"{OD}/parse_{sp}_v2_win.jsonl", GRPC_ROWS=rows))
        M[c] = (b, n, dg)
    out = {}
    for k in cand:
        i = int(k)
        out[i] = {c: dict(base=float(M[c][0].value.get(i, np.nan)), new=float(M[c][1].value.get(i, np.nan)),
                          ok=bool(M[c][1].ok.get(i, False)), mode=M[c][1]["mode"].get(i), f=M[c][1].f.get(i),
                          unit=M[c][1].unit.get(i), why=M[c][1].why.get(i), alt=bool(M[c][2].get(str(i), {}).get("alt_parse")))
                  for c in ("m", "v")}
        out[i]["vid"] = cand[k]["vid"]
    return out


def edit(sp, u, raw):
    REC[sp] = {}
    meas = remeasure(sp)
    ch = []
    for i, m in sorted(meas.items()):
        mv = raw.geo_value.get(i)
        h1 = abs(float(raw.h1_value.get(i)))
        pick = None
        for c in ("m", "v"):
            x = m[c]
            rep = bool(np.isfinite(x["base"]) and pd.notna(mv) and abs(x["base"] / float(mv) - 1) < 1e-4)
            x["reproduces"] = rep
            if rep and pick is None:
                pick = c
        if os.environ.get("C30W_CHAIN") in ("m", "v"):              # check only: force a chain (same-rule comparison)
            pick = os.environ["C30W_CHAIN"]
        rec = dict(used=None if pd.isna(mv) else float(mv), mode0=raw.meas_mode.get(i), h1=h1, chain=pick, m=m["m"], v=m["v"])
        REC[sp][i] = rec
        if pick is None:
            print(sp, i, "no chain reproduces the clean geo", mv, {c: (m[c]["base"], m[c]["new"]) for c in "mv"}); continue
        x = m[pick]; new = x["new"]; un = x["unit"]
        k = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(un, 2 if "^2" in str(un) else 1)
        lo, hi = PLAUS[k]
        ok = bool(x["ok"]) and np.isfinite(new) and new > 0 and lo <= new * UNIT_SI[un] <= hi
        valid = bool(ok and abs(np.log10(new / h1)) <= 2)
        rec.update(ok=ok, valid=valid, accept=valid)
        print(sp, i, f"chain {pick}: base {x['base']:.6g} reproduces {mv}; new {new:.6g} ({x['mode']}) plaus={ok} 2-decade={valid}")
        if not valid:
            continue
        raw.loc[i, "geo_value"] = new; raw.loc[i, "meas_value"] = new; raw.loc[i, "meas_valid"] = True
        raw.loc[i, "meas_mode"] = x["mode"]; raw.loc[i, "meas_why"] = "rul044"
        if pick == "m":
            raw.loc[i, "merge_src"] = "d"
        raw.loc[i, "meas_f"] = x["f"] if pd.notna(x["f"]) else np.nan
        raw.loc[i, "answer_source"] = "meas"; raw.loc[i, "parsed_value"] = new
        ch.append(i)
    return ch


out = G.run("rul044", edit)
for sp, (n, b, bs, rows) in out.items():
    keep = G.gate(n, b, bs, rows)
    P = {}
    for i in keep:
        r = REC[sp][i]; x = r[r["chain"]]
        P[i] = {"value": float(n.answer[i]),
                "why": (f"rul044: time window without the seconds unit was not parsed (whole clip measured); re-parsed "
                        f"t1/t2 from the own question; geo re-measured on the row's own track alone (chain {r['chain']}) "
                        f"{x['base']:.4g} -> {x['new']:.4g} ({x['mode']}); strict fusion {n.answer_source[i]} with CaW {n.cs[i]:.4g}; "
                        f"was {b.answer_source[i]} {b.answer[i]:.4g}")}
        print(sp, i, P[i])
    for i, r in REC.get(sp, {}).items():
        print(sp, i, "record", {k: v for k, v in r.items() if k not in ("m", "v")}, "| m", r["m"], "| v", r["v"])
    print(f"[rul044 {sp}] candidates {len(REC.get(sp, {}))}, geo rows replaced {len(rows)}, written {len(keep)}")
    G.write_patch("rul044", sp, P)
json.dump({sp: {str(k): v for k, v in d.items()} for sp, d in REC.items()}, open(f"{R}/cache/tmp/ag_c30win_rep_meas.json", "w"),
          indent=1, default=str)
