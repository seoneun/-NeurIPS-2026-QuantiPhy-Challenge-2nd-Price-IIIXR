#!/usr/bin/env python
import json, re, sys
import numpy as np, pandas as pd

import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import measure_v3 as MV
import track_common as tc
from qp_common import load_split, OUT_COLS

split, stage = sys.argv[1], sys.argv[2]
K = 3
import os
MV.FIX = set("abcdef"); MV.TRACK_VER = os.environ.get("RL_TRK", "v2")
FAIL = r"^(prior_|scene_prior)"
PERSON = r"\b(person|people|man|woman|boy|girl|player|pedestrian|walker|lady|kid|child|astronaut|runner|human|passer|receiver|swimmer|skier|cyclist|rider|driver)\b"
ROUND = r"\b(ball|balls|sphere|planet|moon|coin|disc|disk|wheel|tire|tyre|orb|bubble|droplet|drop|circle|plate|ring|globe)\b"
MOTION = {"speed", "avgspeed", "hvel", "accel", "disp", "path", "maxheight", "orbitdiam"}
RL_MAPTAG = os.environ.get("RL_MAPTAG", "")
RL_BASE_Q, RL_BASE_M2 = os.environ.get("RL_BASE_Q", "geo_v4x_d_vw"), os.environ.get("RL_BASE_M2", "geo_v4xg_d_vw")
RL_OUT = os.environ.get("RL_OUT", "geo_v4rlc")
MAP = f"{R}/analysis/v3/ruler_{split}{RL_MAPTAG}.json"
import os
SUF = os.environ.get("RL_SUFFIX", "")
ANS = os.environ.get("RL_ANS", "full")


def ruler_dim(phrase):
    p = (phrase or "").lower()
    if re.search(PERSON, p):
        return "height"
    if re.search(ROUND, p):
        return "diameter"
    return "length"


def stage_q():
    base = pd.read_csv(f"{R}/preds/{split}_{RL_BASE_Q}_raw.csv").set_index("Unnamed: 0")
    df = load_split(split); df.index = df["Unnamed: 0"].astype(int)
    byv, vjs = MV.load_split_tracks(split, "e")
    rows, qs, qid = {}, {}, 1000000
    n_fail = 0
    for v, recs in byv.items():
        vj = vjs.get(v)
        if vj is None or recs[0]["video_type"][1] != "2":
            continue
        V = MV.Video(vj, "e")
        ents = {}
        for r in recs:
            for role in ("target", "target2", "prior_obj"):
                e = r.get(role)
                k = MV.Video.key(e)
                if k and k not in ents:
                    ents[k] = e
        cand = {}
        for k, e in ents.items():
            trk = V.tr.get(k)
            ok, _ = MV.track_ok(trk, False)
            raw = (e.get("raw") or e.get("phrase") or "")
            if not ok or re.search(MV.M1.SCENE, raw) or re.search(MV.M1.SCENE, e.get("phrase") or ""):
                continue
            if re.search(MV.M1.PART, raw):
                continue
            cand[k] = (e, trk, float(trk.size_px))
        for r in recs:
            b = base.loc[r["id"]]
            if b.answer_source == "meas" or not re.search(FAIL, str(b.meas_why or "")):
                continue
            n_fail += 1
            kind = r["q"]["kind"]
            tk = MV.Video.key(r["target"])
            ttr = V.get(r["target"])
            rl = []
            for k, (e, trk, sz) in cand.items():
                if kind == "size" and (k == tk or (ttr is not None and (trk is ttr or MV.M1.same_object(trk, ttr)))):
                    continue
                rl.append((sz, k))
            rl = [k for _, k in sorted(rl, reverse=True)][:K]
            if not rl:
                continue
            rows[r["id"]] = dict(video_id=v, kind=kind, rulers=[])
            for k in rl:
                e = cand[k][0]
                dim = ruler_dim(e.get("raw") or e.get("phrase"))
                key = (v, k, dim)
                if key not in qs:
                    raw = (e.get("raw") or e.get("phrase")).strip()
                    o = df.loc[r["id"]].copy()
                    o["Unnamed: 0"] = qid
                    o["question"] = f"What is the {dim} of the {raw} in meters?"
                    o["ground_truth_posterior"] = ""
                    qs[key] = dict(qid=qid, row=o, ent=e, dim=dim, size_px=cand[k][2])
                    qid += 1
                rows[r["id"]]["rulers"].append(dict(qid=qs[key]["qid"], key=k, dim=dim, ent=qs[key]["ent"], size_px=qs[key]["size_px"]))
    cols = ["Unnamed: 0", "video_id", "video_source", "video_type", "fps", "inference_type", "question", "ground_truth_prior",
            "depth_info", "ground_truth_posterior"]
    out = pd.DataFrame([q["row"] for q in qs.values()])
    out = out[cols].rename(columns={"Unnamed: 0": ""})
    out.to_csv(f"{R}/caw/sc/inputs/{split}_ruler{RL_MAPTAG}.csv", index=False)
    json.dump({str(k): v for k, v in rows.items()}, open(MAP, "w"))
    kinds = pd.Series([v["kind"] for v in rows.values()]).value_counts().to_dict()
    print('')
    print(out.question.head(12).tolist())


def caw_answers():
    a = _read_answers(f"{R}/caw/sc/out/{split}_ruler{RL_MAPTAG}/g16.jsonl")
    if ANS == "crop" and os.path.exists(f"{R}/caw/sc/out/{split}_ruler_crop/g16.jsonl"):
        a.update(_read_answers(f"{R}/caw/sc/out/{split}_ruler_crop/g16.jsonl"))
    return a


def _read_answers(path):
    a = {}
    for ln in open(path, encoding="utf-8"):
        x = json.loads(ln)
        t = (x["responses"] or [""])[0]
        p = [z for z in x["parsed"] if z is not None and np.isfinite(z) and z > 0]
        if not p:
            continue
        v = float(p[0])
        u = re.search(r"\d\s*(km|cm|mm|m)\b", t.lower())
        if u:
            v *= {"km": 1000.0, "cm": 0.01, "mm": 0.001, "m": 1.0}[u.group(1)]
        a[int(x["original_id"])] = v
    return a


def stage_p():
    rows = {int(k): v for k, v in json.load(open(MAP)).items()}
    ans = caw_answers()
    recs = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    for k in range(K):
        n = 0
        with open(f"{R}/geo/out/parse_{split}_v2_rl{SUF}{k}.jsonl", "w") as fo:
            for r in recs:
                x = rows.get(r["id"])
                rl = [z for z in (x["rulers"] if x else []) if z["qid"] in ans]
                if len(rl) > k:
                    z = rl[k]
                    r = dict(r)
                    e = z["ent"]
                    r["priors"] = [dict(kind="size", dim=z["dim"], value_si=ans[z["qid"]], t=None, t1=None, t2=None,
                                        obj=e.get("phrase"), gravity=False, ok=True, raw=f"ruler {e.get('phrase')} (CaW)")]
                    r["prior_obj"] = dict(e)
                    r["prior_dkeys"], r["prior_dsrc"], r["prior_whole"] = [], "none", True
                    n += 1
                fo.write(json.dumps(r) + "\n")
        print('')


def stage_m():
    rows = {int(k): v for k, v in json.load(open(MAP)).items()}
    base = pd.read_csv(f"{R}/preds/{split}_geo_v4x_d_vw_raw.csv").set_index("Unnamed: 0")
    vals = []
    for k in range(K):
        g = pd.read_csv(f"{R}/preds/{split}_geo_rl{k}_raw.csv").set_index("Unnamed: 0").reindex(base.index)
        ok = (g.answer_source == "meas") & g.meas_mode.astype(str).str.startswith("2d_ratio") & base.index.isin(list(rows))
        vals.append(pd.to_numeric(g.geo_value, errors="coerce").where(ok))
    Vv = pd.concat(vals, axis=1)
    Vv.columns = list(range(K))
    with np.errstate(divide="ignore", invalid="ignore"):
        med = np.exp(np.log(Vv).median(axis=1, skipna=True))
    nr = Vv.notna().sum(axis=1)
    take = med.notna() & (base.answer_source != "meas")
    kinds = pd.Series({i: rows[i]["kind"] for i in rows}).reindex(base.index)
    for tag, strong_motion in (("geo_v4rlw_d_vw", False), ("geo_v4rls_d_vw", True)):
        o = base.copy()
        o.loc[take, "geo_value"] = med[take]
        o.loc[take, "answer_source"] = "meas"; o.loc[take, "meas_valid"] = True
        mode = pd.Series("2d_vlmruler", index=o.index)
        if strong_motion:
            mode = mode.where(~kinds.isin(MOTION), "2d_ratio")
        o.loc[take, "meas_mode"] = mode[take]
        o.loc[take, "merge_src"] = "ruler"
        o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
        o.to_csv(f"{R}/preds/{split}_{tag}_raw.csv")
        o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_{tag}.csv", index=False)
        print('')
    if split == "val":
        d = load_split("val"); d.index = d["Unnamed: 0"].astype(int)
        y = pd.to_numeric(d.ground_truth_posterior, errors="coerce")
        T = pd.read_csv(f"{R}/analysis/v3/v7_val_rows.csv", index_col=0)
        S = pd.DataFrame(dict(kind=kinds, med=med, n=nr, cs=T.cs, y=y, mra7=T.mra, q=d.question.str.slice(0, 60)))[take]
        for k in range(K):
            S[f"r{k}"] = Vv[k][take]
        pd.set_option("display.width", 250)
        print(S.round(3).to_string())


def stage_m2():
    rows = {int(k): v for k, v in json.load(open(MAP)).items()}
    base = pd.read_csv(f"{R}/preds/{split}_{RL_BASE_M2}_raw.csv").set_index("Unnamed: 0")
    vals = []
    for k in range(K):
        g = pd.read_csv(f"{R}/preds/{split}_geo_rl{SUF}{k}_raw.csv").set_index("Unnamed: 0").reindex(base.index)
        ok = (g.answer_source == "meas") & g.meas_mode.astype(str).str.startswith("2d_ratio") & base.index.isin(list(rows))
        vals.append(pd.to_numeric(g.geo_value, errors="coerce").where(ok))
    Vv = pd.concat(vals, axis=1); Vv.columns = list(range(K))
    with np.errstate(divide="ignore", invalid="ignore"):
        med = np.exp(np.log(Vv).median(axis=1, skipna=True))
        spread = Vv.max(axis=1) / Vv.min(axis=1)
    nr = Vv.notna().sum(axis=1)
    byv, vjs = MV.load_split_tracks(split, "e")
    moving = pd.Series(np.nan, index=base.index)
    for v, recs in byv.items():
        if v not in vjs:
            continue
        Vd = MV.Video(vjs[v], "e")
        for r in recs:
            if r["id"] not in rows:
                continue
            trk = Vd.get(r["target"])
            if trk is not None and trk.n:
                rng = float(np.linalg.norm(trk.uv.max(0) - trk.uv.min(0)))
                moving[r["id"]] = rng / max(trk.size_px, 1.0)
    kinds = pd.Series({i: rows[i]["kind"] for i in rows}).reindex(base.index)
    mot = kinds.isin(MOTION)
    selfg = pd.Series(False, index=base.index)
    selfv = pd.Series(np.nan, index=base.index)
    if os.environ.get("RL_SELF") == "1":
        tkey = {}
        for v, recs in byv.items():
            for r in recs:
                if r["id"] in rows:
                    tkey[r["id"]] = MV.Video.key(r["target"])
        grav_ = base.ground_truth_prior.astype(str).str.contains(r"gravit|9\.8|9\.81", case=False, regex=True)
        for i, info_ in rows.items():
            g_ = bool(grav_.get(i, False))
            gself_ = g_ and os.environ.get("RL_SELF_GRAV") == "1" and str(base.meas_why.get(i, "")).startswith("prior_")
            if i not in base.index or info_["kind"] not in (("speed", "avgspeed", "hvel", "disp") if gself_ else ("speed", "avgspeed", "hvel")):
                continue
            if os.environ.get("RL_SELF_NOGRAV") == "1" and g_ and not gself_:
                continue
            for j, rr in enumerate(info_["rulers"][:K]):
                if rr["key"] == tkey.get(i) and pd.notna(Vv.loc[i, j]):
                    selfv[i] = Vv.loc[i, j]
                    selfg[i] = gself_
                    break
    weak_vlm = (base.answer_source == "meas") & base.meas_mode.astype(str).str.startswith("vlm_") if os.environ.get("RL_OVER_VLM") == "1"         else pd.Series(False, index=base.index)
    NMIN, SPR, MOV = int(os.environ.get("RL_NMIN", 2)), float(os.environ.get("RL_SPREAD", 1.3)), float(os.environ.get("RL_MOVE", 2.0))
    take = med.notna() & ((base.answer_source != "meas") | weak_vlm) & (nr >= NMIN) & (spread <= SPR) & (~mot | (moving >= MOV))
    take_self = ~take & selfv.notna() & ((base.answer_source != "meas") | weak_vlm) & (moving >= MOV)
    med = med.where(~take_self, selfv)
    o = base.copy()
    o.loc[take, "geo_value"] = med[take]
    o.loc[take, "answer_source"] = "meas"; o.loc[take, "meas_valid"] = True
    o.loc[take, "meas_mode"] = pd.Series("2d_vlmruler", index=o.index).where(~mot, "2d_ratio")[take]
    o.loc[take, "merge_src"] = "ruler"
    if take_self.any():
        o.loc[take_self, "geo_value"] = med[take_self]
        o.loc[take_self, "answer_source"] = "meas"; o.loc[take_self, "meas_valid"] = True
        o.loc[take_self, "meas_mode"] = os.environ.get("RL_SELF_MODE", "2d_selfruler")
        tg_ = take_self & selfg
        if tg_.any():
            o.loc[tg_, "meas_mode"] = os.environ.get("RL_SELF_GRAV_MODE", "2d_ratio")
            print('')
        o.loc[take_self, "merge_src"] = "selfruler"
        print('')
    o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
    o.to_csv(f"{R}/preds/{split}_{RL_OUT}{SUF}_d_vw_raw.csv")
    o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_{RL_OUT}{SUF}_d_vw.csv", index=False)
    print('')
    if split == "val":
        d = load_split("val"); d.index = d["Unnamed: 0"].astype(int)
        y = pd.to_numeric(d.ground_truth_posterior, errors="coerce")
        S = pd.DataFrame(dict(kind=kinds, med=med, n=nr, spread=spread, moving=moving, y=y, q=d.question.str.slice(0, 50)))[take]
        pd.set_option("display.width", 220)
        print(S.round(3).to_string())


{"q": stage_q, "p": stage_p, "m": stage_m, "m2": stage_m2}[stage]()
