"""rul024: re-run the measurement that produced the row's geo value, with the parse fixed.
The rows in question are 2d_gself (analysis/v3/ruler_build.py m2, RL_SELF_GRAV): a gravity-prior row without a free fall is
measured by geo/measure_v3.py (imported read-only) on the substituted-prior parse geo/out/parse_<split>_v2_rl<k>.jsonl whose
prior is the target's own CaW size (caw/sc/chain_ruler.sh: --fix abcdef --parse_tag _rl<k> --moge --moge_opts fbrej, default
tracks).  Fix (scan.gate): q.t1/q.t2 from the 'between A[s] and B[s]' match, the matched span stripped from the target.
All rows of the video are measured with the parse as is (must reproduce preds/<split>_geo_rl<k>_raw.csv and the geo raw)
and with the fix.
usage: python remeas.py [tracks=v2] [envset=min|full]
   -> cache/tmp/ag_c22rngparse_raw_{split}.csv (geo raw with the fixed rows replaced) + cache/tmp/ag_c22rngparse_meas.json"""
import os, sys, json, copy, re
TRK = sys.argv[1] if len(sys.argv) > 1 else "v2"
ENVSET = sys.argv[2] if len(sys.argv) > 2 else "min"
if ENVSET == "full":
    for _k, _v in dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
                       QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
                       QP_NEAREST="1", QP_BELT="1", QP_FPERSON="1", QP_FPERSON_MINN="2",
                       QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1", QP_TSHIFT="1").items():
        os.environ[_k] = _v
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo")
import numpy as np, pandas as pd
import measure_v3 as MV
sys.path.insert(0, f"{R}/analysis/v3/ag/holder")
import remeasure as RM            # setup()/measure() helpers (only used for MoGe/scene-ruler/focal loading)
sys.path.insert(0, f"{R}/analysis/v3/ag/rul024")
import scan as S
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
SPAN = re.compile(r"\s*\bbetween\s+\d+(?:\.\d+)?\s*s?\s+and\s+\d+(?:\.\d+)?\s*s?\b", re.I)


def strip(p):
    if not p:
        return p
    return SPAN.sub("", p).strip() or p


def rl_k(sp, i, rulers_map, rs_rl):
    """which substituted-prior parse holds the row's own-size ruler (ruler key == target key), as ruler_build m2 picks it"""
    info = rulers_map.get(str(i))
    if not info:
        return None
    return info


def run(sp):
    C = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        r = json.loads(l)
        m, bad = S.gate(r)
        if m is not None and bad:
            C[r["id"]] = (float(m.group(1)), float(m.group(2)), r["video_id"])
    print(sp, "gated rows", C, flush=True)
    G = pd.read_csv(f"{R}/preds/{sp}_{GEO}_raw.csv").set_index("Unnamed: 0")
    res = {}
    if C:
        vids = sorted({c[2] for c in C.values()})
        fmed = RM.setup(sp, tracks=TRK, vids=set(vids))
        RMAP = json.load(open(f"{R}/analysis/v3/ruler_{sp}.json"))
        for i, (t1, t2, v) in C.items():
            mode0 = str(G.meas_mode.get(i))
            if mode0 != "2d_gself":
                print(f"  {i}: geo mode {mode0} - only 2d_gself rows are re-measured here"); continue
            # ruler_build m2: the self ruler is the first of the row's rulers (up to K=3) whose key is the target key
            rul = RMAP[str(i)]["rulers"]
            print(f"  {i}: rulers {[(z['key'], z['qid']) for z in rul]}")
            for k in range(min(3, len(rul))):
                MV.PARSE_TAG = f"_rl{k}"
                byv, vjs = MV.load_split_tracks(sp, "e")
                rs = byv[v]
                rr = [r for r in rs if r["id"] == i][0]
                if rul[k]["key"] != MV.Video.key(rr["target"]):
                    print(f"  {i}: rl{k} ruler {rul[k]['key']} is not the target {MV.Video.key(rr['target'])}"); continue
                GR = pd.read_csv(f"{R}/preds/{sp}_geo_rl{k}_raw.csv").set_index("Unnamed: 0")
                rs2 = copy.deepcopy(rs)
                for r in rs2:
                    if r["id"] != i:
                        continue
                    print(f"  {i} rl{k} prior {r['priors']} | before: q.t={r['q'].get('t')} t1={r['q'].get('t1')} t2={r['q'].get('t2')} q.target={r['q'].get('target')!r} target={r['target']}")
                    r["q"] = dict(r["q"], t1=t1, t2=t2, t=None, target=strip(r["q"].get("target")))
                    r["target"] = dict(r["target"], phrase=strip(r["target"].get("phrase")), raw=strip(r["target"].get("raw")))
                    print(f"  {i} after : q.t1={t1} t2={t2} q.target={r['q']['target']!r} target={r['target']}")
                b, _ = RM.measure(rs, vjs[v], fmed)
                o, _ = RM.measure(rs2, vjs[v], fmed)
                allrep = True
                for r in rs:
                    j = r["id"]
                    g0 = GR.geo_value.get(j); gg = G.geo_value.get(j)
                    rep = (pd.isna(g0) and not np.isfinite(b[j]["value"])) or (pd.notna(g0) and np.isfinite(b[j]["value"]) and abs(b[j]["value"] / g0 - 1) < 1e-4)
                    allrep &= bool(rep) or pd.isna(g0)
                    print(f"{sp} {j} {v} {'FIRE' if j == i else '    '} rl{k} raw {g0!s:.8} ({GR.meas_mode.get(j)}) geo {gg!s:.8} ({G.meas_mode.get(j)}) | base {b[j]['value']:.5g} {b[j]['mode']} rep={rep} | new {o[j]['value']:.5g} {o[j]['mode']} {o[j]['why']} | {r['question'][:72]}", flush=True)
                x = o[i]
                ok = allrep and np.isfinite(x["value"]) and x["value"] > 0 and abs(b[i]["value"] / G.geo_value[i] - 1) < 1e-4
                res[str(i)] = dict(raw=float(G.geo_value[i]), base=b[i]["value"], new=x["value"], mode=x["mode"], why=x["why"], k=k,
                                   t1=t1, t2=t2, vid=v, reproduced=bool(ok))
                if ok:
                    G.loc[i, "geo_value"] = x["value"]           # mode stays 2d_gself (ruler_build m2 relabels the self-ruler 2d_ratio)
                    G.loc[i, "meas_why"] = "rul024"
                    if "parsed_value" in G.columns and G.answer_source.get(i) == "meas":
                        G.loc[i, "parsed_value"] = x["value"]
                break
    G.to_csv(f"{R}/cache/tmp/ag_c22rngparse_raw_{sp}.csv")
    print(f"[{sp}] geo raw rows replaced: {sum(1 for x in res.values() if x['reproduced'])}")
    return res


if __name__ == "__main__":
    out = {sp: run(sp) for sp in ("test", "val")}
    json.dump(out, open(f"{R}/cache/tmp/ag_c22rngparse_meas.json", "w"), indent=1, default=float)
