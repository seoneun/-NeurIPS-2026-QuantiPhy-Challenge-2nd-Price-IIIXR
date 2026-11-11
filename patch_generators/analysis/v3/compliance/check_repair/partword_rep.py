"""partword_rep.py BASE: per-question version of analysis/v3/ag/rul329 (remeas.py test/val, run_fusion.py raw / fuse,
mkpatch.py).
Rule (unchanged): a size row whose own parse target is a part word ('longest side', 'side', 'edge', 'height', ...) of an
object X named in its own question ('longest side of the wallet') is re-targeted to X (QP_SIDE_FIX retarget); the track
for X is (1) an own-pool query whose key is X's key, else (2) the VLM-anchored v3nt track whose query 'expr' names X
(rul329/remeas.py v3_alias); candidates() and the regexes are copied from rul329/remeas.py (that module is not
imported: it sets QP_* measurement env vars at import).
Changes vs production:
  - re-measure: remeas.py ran MV.measure_video on ALL rows of the video with the video-wide track pool (s_audit.csv:
    other_phrase_tracks) and the pooled focal geo/out/focal_medians_test.json; here the row alone, with the own-text track
    pool (grpC_pre_ownmeas.py = owntracks/meas.py, hooks GRPC_FIXJSON = the retarget, GRPC_GRAFT = the v3nt alias), on
    both clean chains (m: v2scSIL / its fallback file, v: v3ntmvSIL); the chain used is the first (m, v) whose base
    measurement (current parse) reproduces the clean geo value (rel 1e-4, both NaN counts); a row whose retarget leaves
    the own measurement unchanged is skipped (production compared the new value with the geo raw directly);
  - fusion: the clean strict fusion (grpC_strict.py) of the main table with the row's geo replaced as run_fusion.py raw
    does (2-decade check vs h1); base = BASE's main fusion xt_<B> instead of xt_Nzg.
Patch = re-measured rows whose main strict answer changes -> cache/tmp/ag_partword_patch_{test,val}.json."""
import os, sys, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpC_strict as G
R = G.R
PARTW = r"(?:(?:longer|shorter|long|short|longest|shortest)\s+)?(?:side|edge|height|width|length)|moving speed|speed"
PART_TP = re.compile(r"^(?:the\s+)?(?:" + PARTW + r")$")
ART = r"(?:the\s+|a\s+|an\s+)?"
OF_X = re.compile(r"\b(?:" + PARTW + r")\s+of\s+" + ART + r"(?:(?:" + PARTW + r")\s+of\s+" + ART + r")*"
                  r"((?:[a-z]+'s\s+)?(?:\([a-z ]+\)\s+)?[a-z][a-z \-]*?)\s*(?:\bin\b|\bat\b|,|\?|$)")
SCENE = re.search(r'^SCENE = r"(.*?)"$', open(f"{R}/geo/measure.py").read(), re.M).group(1)   # = measure_v3.SCENE
MEAS = {}


def candidates(split):
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    C = {}
    for r in P:
        tp = ((r.get("target") or {}).get("phrase") or "").strip().lower()
        if not PART_TP.match(tp):
            continue
        ql = r["question"].lower().replace("’", "'")
        m = OF_X.search(ql)
        x = m.group(1).strip() if m else None
        if x:
            x = re.sub(r"^[a-z]+'s\s+", "", x)
            x = re.sub(r"\([a-z ]+\)\s*", "", x).strip()
        C[r["id"]] = dict(vid=r["video_id"], tp=tp, x=x, scene=bool(x and re.search(SCENE, x)), q=r["question"], kind=r["q"].get("kind"))
    return C


def remeasure(sp):
    C = candidates(sp)
    todo = {i: c for i, c in C.items() if c["x"] and not c["scene"] and c["kind"] == "size"}
    for i, c in sorted(C.items()):
        print(sp, i, c, "" if i in todo else "-> skip (no X, scene X or not a size row)")
    if not todo:
        return {}
    vids = {c["vid"] for c in todo.values()}
    fix = {sp: {str(i): {"target": {"phrase": c["x"], "raw": c["x"], "where": []}, "q": {"target": c["x"]}} for i, c in todo.items()}}
    gr = {sp: {str(i): {"x": c["x"], "src": "v3nt"} for i, c in todo.items()}}
    fp, gp = f"{R}/cache/tmp/ag_partword_rep_fix.json", f"{R}/cache/tmp/ag_partword_rep_graft.json"
    json.dump(fix, open(fp, "w")); json.dump(gr, open(gp, "w"))
    res = {}
    for trk in ("m", "v"):
        b, _ = G.ownmeas(sp, trk, vids, f"_grpCpw_base", {})
        n, dg = G.ownmeas(sp, trk, vids, f"_grpCpw_new", dict(GRPC_FIXJSON=fp, GRPC_GRAFT=gp))
        for i in todo:
            res.setdefault(i, {"x": todo[i]["x"], "vid": todo[i]["vid"]})[trk] = dict(
                base=float(b.value.get(i, np.nan)), base_why=b.why.get(i), new=float(n.value.get(i, np.nan)), mode=n["mode"].get(i),
                f=n.f.get(i), why=n.why.get(i), graft=dg.get(str(i), {}).get("graft"), pool=dg.get(str(i), {}).get("pool", {}).get("why"))
    for i, x in res.items():
        print(sp, i, x)
    return res


def eq(a, b, tol):
    return (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(float(a) / float(b) - 1) < tol)


def edit(sp, u, raw):
    MEAS[sp] = remeasure(sp)
    ch = []
    for i, x in sorted(MEAS[sp].items()):
        g0 = raw.geo_value.get(i)
        src = str(raw.merge_src.get(i))
        # chain: the one whose base (current parse) reproduces the clean geo (rel 1e-4; both NaN counts), m first
        # (the production remeas.py measured on the pipeline m-chain file when the video has one, else on v3nt)
        x["chain"] = next((c for c in ("m", "v") if eq(x[c]["base"], g0, 1e-4)), None)
        if x["chain"] is None:
            print(sp, i, "no chain reproduces the clean geo", g0, {c: x[c]["base"] for c in "mv"}); continue
        m = x[x["chain"]]
        new = m["new"]
        if eq(new, m["base"], 1e-6):
            print(sp, i, f"chain {x['chain']}: the retarget does not change the own measurement ({new}); skip"); continue
        same = (pd.isna(g0) and not np.isfinite(new)) or (np.isfinite(new) and pd.notna(g0) and abs(new / g0 - 1) < 1e-6)
        if same:
            continue
        raw.loc[i, "geo_value"] = new if np.isfinite(new) else np.nan
        raw.loc[i, "meas_mode"] = m["mode"] if np.isfinite(new) else np.nan
        raw.loc[i, "meas_f"] = m["f"] if pd.notna(m["f"]) else np.nan
        raw.loc[i, "meas_why"] = m["why"] if isinstance(m["why"], str) else np.nan
        h1 = abs(float(raw.h1_value.get(i)))
        ok = bool(np.isfinite(new) and new > 0 and abs(np.log10(new / h1)) <= 2)
        raw.loc[i, "meas_valid"] = ok
        raw.loc[i, "meas_value"] = new
        if ok:
            raw.loc[i, "answer_source"] = "meas"; raw.loc[i, "parsed_value"] = new
        else:
            raw.loc[i, "geo_value"] = np.nan
        print(sp, i, "geo", g0, "->", new, m["mode"], "valid", ok)
        ch.append(i)
    return ch


out = G.run("rul329", edit)
for sp, (n, b, bs, rows) in out.items():
    keep = G.gate(n, b, bs, rows)
    P = {}
    for i in keep:
        x = MEAS[sp][i]; m = x[x["chain"]]
        P[i] = {"value": float(n.answer[i]),
                "why": (f"rul329: '{x['x']}' track from the VLM-anchored v3nt file (chain {x['chain']}), row measured alone on its own-text pool; "
                        f"geo {n.g[i]:.4g} ({m['mode']}) -> strict fusion {n.answer_source[i]} with CaW {n.cs[i]:.4g}; "
                        f"was {b.answer_source[i]} {b.answer[i]:.4g}")}
        print(sp, i, P[i])
    print(f"[rul329 {sp}] re-measured {len(MEAS.get(sp, {}))}, geo rows replaced {len(rows)}, written {len(keep)}")
    G.write_patch("rul329", sp, P)
json.dump({sp: {str(k): v for k, v in d.items()} for sp, d in MEAS.items()}, open(f"{R}/cache/tmp/ag_partword_rep_meas.json", "w"),
          indent=1, default=str)
