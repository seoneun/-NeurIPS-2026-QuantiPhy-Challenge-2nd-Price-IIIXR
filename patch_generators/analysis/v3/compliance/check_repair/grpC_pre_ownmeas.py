#!/usr/bin/env python
"""grpC_pre_ownmeas.py: copy of analysis/v3/compliance/owntracks/meas.py (per-row measure_video, own-text track pool) with
hooks for the grpC re-measurements (check_repair/{rul044,rul024,rul329,rul099}_rep.py); without any GRPC_* env it
behaves as meas.py. Hooks (env):
  GRPC_FIXJSON=<json {split: {id: {field: value}}}>: per-row parse edits applied after loading (before the per-row time
    shift): field 'q' is merged into r['q']; 'target' / 'target2' / 'prior_obj' are merged into the entity (None removes it).
  GRPC_ALT_PARSE=<jsonl> + GRPC_ROWS=<id,...>: the listed rows take their record from the alternate parse file, loaded by
    the same MV.load_split_tracks (the production rul044/remeas.py mechanism, MV.open redirected for the parse path).
  GRPC_GRAFT=<json {split: {id: {"x": phrase, "src": track tag}}}>: after the own-text restriction, a query of the track file
    tracks_<split>_<src>/<video>.json whose 'expr' names x (analysis/v3/ag/rul329/remeas.py v3_alias) is added under the
    row's (edited) target key, with its SAM2 object (the production rul329 graft).
  GRPC_PARSE_TAG=<suffix>: measure_v3 --parse_tag (e.g. _rl0, the substituted-prior parse of ruler_build m2).
  GRPC_ENVSET=min: do not set the QP_* measurement environment (rul024 remeas.py 'min').
  GRPC_NOMOGE=1: no --moge.
--- docstring of owntracks/meas.py follows ---
owntracks (meas): copy of analysis/v3/compliance/solo/meas.py (the sil7 'new' geo measurement, one row per
measure_video call, per-row QP_TSHIFT / FPERSON / scene-ruler kind) with the video's track pool restricted to the row's
OWN-TEXT pool (analysis/v3/compliance/owntracks/owntext.py: tracks whose detector phrase is named in the row's question,
prior text or depth_info keys, plus the row's own parse entities). Modes:
  solo   no restriction. Control: equals cache/tmp/ag_solo_meas_<split>_<var>_solo.csv on every column.
  ownt   own-text pool, owntext.py mode 'tok' (the track's phrase is a contiguous phrase of the row's own text; primary).
  ownh   own-text pool, owntext.py mode 'head' (also tracks whose head noun is named in the row's own text; variant).
In ownt / ownh the restricted video json keeps only the own-pool queries (queries and queries_d) and only the SAM2 objects
that an own-pool query points to in the track file or in an earlier file of its chain (CHAIN below: the mover patch
re-pointed some '#second' queries and left the object they seeded unreferenced); the QP_SECOND_FIX search
(geo/measure_v3.py:985-1014) ranges over vj['objs'].
Every mode runs an in-memory copy of geo/measure_v3.py with logging lines added (the file itself is not changed); the
logging records, per row, which tracks each video-wide pool used (POOLLOG):
  ff         falling track chosen by the free-fall search over all tracks (1204-1218; only when the prior names no object)
  belt       tracks whose speeds gave the belt speed (1230-1241)
  sameclass  tracks of the prior's class whose extents replaced the prior's (1294-1310)
  nearest    candidate tracks of 'the nearest X' other than target2 (nearest_cands, 1017-1031)
  floor      tracks whose box bottoms entered floor_level (905-917)
  gclass     tracks counted by the class-size plausibility check of a gravity scale (gravity_scale_check, 632-643)
  second     SAM2 object taken by second_instance (985-1014) and the query keys pointing to it
  colour     colour re-assignments of Video.colour_assign (239-276)
usage: python analysis/v3/compliance/owntracks/meas.py SPLIT m|v MODE [VIDEO,VIDEO,...]   (env OWNT_SUF: output suffix;
env SOLO_Z=1: lab depth_info read as ToF Z on the production tracks v2sc / v3ntmv, as solo/meas.py; env OWNT_TRK_M /
OWNT_TRK_V: other track tags, e.g. v2scSILseg2d / v3ntmvSILseg2d with SIL_NO_MOGE=1 = the seg2d guard run of
analysis/v3/compliance/rules/seg2d_meas.py)
writes cache/tmp/ag_ownt_meas_<split>_<var>_<mode><SUF>.csv and cache/tmp/ag_ownt_diag_<split>_<var>_<mode><SUF>.json
(per row: rot, tshift / fperson / sruler as solo/meas.py; pool = {n_video, n_own, why (own key -> e/q/p/d)}; log = POOLLOG)."""
import os, sys, re, ast, json, types
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../owntracks"))
split, var, mode = sys.argv[1:4]
SEL = set(x for x in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if x)
SUF = os.environ.get("OWNT_SUF", "")
assert mode in ("solo", "ownt", "ownh"), mode
ZMODE = os.environ.get("SOLO_Z") == "1"
TRK = ({"m": "v2sc", "v": "v3ntmv"} if ZMODE else {"m": "v2scSIL", "v": "v3ntmvSIL"})[var]
CHAIN = {"v2scSIL": ["v2", "v2sc", "v2scSIL"], "v3ntmvSIL": ["v3nt", "v3ntmv", "v3ntmvSIL"], "v2sc": ["v2", "v2sc"],
         "v3ntmv": ["v3nt", "v3ntmv"]}[TRK]
if os.environ.get(f"OWNT_TRK_{var.upper()}"):          # e.g. the seg2d merge-guard tracks v2scSILseg2d / v3ntmvSILseg2d
    TRK = os.environ[f"OWNT_TRK_{var.upper()}"]; CHAIN = CHAIN + [TRK]
ENV = dict(QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1", QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
           QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1", QP_FF_SUB_SRC="simulation", QP_PARSE2="1",
           QP_NEAREST="1", QP_BELT="1", RL_SELF_GRAV="1", RL_SELF_GRAV_MODE="2d_gself", QP_FPERSON_MINN="2", QP_FPERSON="1",
           QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3", QP_SCENE_HEAD="1", QP_TSHIFT="1")
if os.environ.get("GRPC_ENVSET") != "min":
    os.environ.update(ENV)
GFIX = json.load(open(os.environ["GRPC_FIXJSON"])).get(split, {}) if os.environ.get("GRPC_FIXJSON") else {}
GRAFT = json.load(open(os.environ["GRPC_GRAFT"])).get(split, {}) if os.environ.get("GRPC_GRAFT") else {}
GROWS = {int(x) for x in os.environ.get("GRPC_ROWS", "").split(",") if x}
sys.path.insert(0, f"{R}/geo")
from owntext import own_keys, obj_owners  # noqa: E402

# ---- logging patches (exactly one occurrence each); they only append to POOLLOG
LOGP = [
    ("            cands = [prior_tr] if prior_tr is not None else [trk for trk in V.tr.values() if track_ok(trk, True)[0]]",
     "            cands = [prior_tr] if prior_tr is not None else [trk for trk in V.tr.values() if track_ok(trk, True)[0]]\n"
     "            _ffpool = prior_tr is None"),
    ("            prior_tr = best if best is not None else prior_tr",
     "            prior_tr = best if best is not None else prior_tr\n"
     "            if _ffpool and best is not None:\n"
     "                POOLLOG.setdefault(r[\"id\"], {})[\"ff\"] = [best.q.get(\"key\")]"),
    ("                sp_ = []", "                sp_ = []; _bk = []"),
    ("                            sp_.append(float(v_))", "                            sp_.append(float(v_)); _bk.append(trk_.q.get(\"key\"))"),
    ("                    belt_ppx = float(np.median(sp_)); res[\"grav\"] = f\"belt_from_{len(sp_)}_objects\"",
     "                    belt_ppx = float(np.median(sp_)); res[\"grav\"] = f\"belt_from_{len(sp_)}_objects\"; "
     "POOLLOG.setdefault(r[\"id\"], {})[\"belt\"] = list(_bk)"),
    ("                alts_ = []", "                alts_ = []; _sk = []"),
    ("                        alts_.append(float(a_))", "                        alts_.append(float(a_)); _sk.append(trk_.q.get(\"key\"))"),
    ("                        res[\"grav\"] = f\"sameclass_{len(alts_)}\"; ppx = am_",
     "                        res[\"grav\"] = f\"sameclass_{len(alts_)}\"; ppx = am_; POOLLOG.setdefault(r[\"id\"], {})[\"sameclass\"] = list(_sk)"),
]
src = open(f"{R}/geo/measure_v3.py").read()
for old, new in LOGP:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
MV = types.ModuleType("measure_v3")
MV.__file__ = f"{R}/geo/measure_v3.py"
POOLLOG = {}
MV.POOLLOG = POOLLOG
sys.modules["measure_v3"] = MV
exec(compile(src, MV.__file__, "exec"), MV.__dict__)
CUR = {"id": None}

# ---- logging wrappers (module globals: measure_video resolves them at call time)
_nc, _fl, _gsc, _si = MV.nearest_cands, MV.floor_level, MV.gravity_scale_check, MV.second_instance


def nearest_cands(V, r, tgt2):
    c = _nc(V, r, tgt2)
    ks = [t.q.get("key") for t in c if t is not tgt2 and t is not None]
    if ks:
        POOLLOG.setdefault(r["id"], {})["nearest"] = ks
    return c


def floor_level(V):
    fl = _fl(V)
    lim = V.W if MV.ROT_CUR == "+x" else (0.0 if MV.ROT_CUR == "-x" else V.H)
    ks = set()
    for trk in list({id(x): x for x in V.tr.values() if x is not None}.values()):
        for j in trk.oke:
            b = trk.st[j]["box"]
            edge = b[2] if MV.ROT_CUR == "+x" else (b[0] if MV.ROT_CUR == "-x" else b[3])
            if abs(edge - lim) > 2:
                ks.add(trk.q.get("key")); break
    POOLLOG.setdefault(CUR["id"], {})["floor"] = sorted(ks)
    return fl


def gravity_scale_check(mpp, fall_tr, V):
    res = _gsc(mpp, fall_tr, V)
    ks = []
    for k, trk in V.tr.items():
        if trk.n < 3 or trk.best < 0.3 or not len(trk.oke):
            continue
        h = MV.tc.head_noun(trk.q.get("phrase"))
        if any(re.search(pat, h) for pat, _ in MV.CLASS_RANGE):
            ks.append(k)
    POOLLOG.setdefault(CUR["id"], {}).setdefault("gclass", []).append([ks, bool(res[0]), res[1]])
    return res


def second_instance(V, vj, tgt):
    a = _si(V, vj, tgt)
    if a is not None:
        o = a.q.get("obj")
        POOLLOG.setdefault(CUR["id"], {})["second"] = dict(obj=o, refs=[q["key"] for q in vj["queries"] if q.get("obj") == o])
    return a


MV.nearest_cands, MV.floor_level, MV.gravity_scale_check, MV.second_instance = nearest_cands, floor_level, gravity_scale_check, second_instance

ZCUR = {"src": None}
_to3d = MV.to3d


def to3d_z(uv, D, f, cx, cy):
    """SOLO_Z=1: copy of analysis/v3/ag/zdepth/meas.py to3d_z (lab sources: D is the optical-axis depth Z)"""
    if ZCUR["src"] != "lab":
        return _to3d(uv, D, f, cx, cy)
    r = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(len(uv))], 1)
    return r * np.asarray(D, float)[:, None], np.ones(len(uv))


if ZMODE:
    MV.to3d = to3d_z
DIAG = {}
OWNERS = {}

# ---- per-row inputs read from the sources of the video-level files (as solo/meas.py)
_srsrc = open(f"{R}/geo/scene_ruler.py").read()
KINDS = ast.literal_eval(re.search(r"^KINDS = (\[.*?\])\n", _srsrc, re.S | re.M).group(1))
PER = re.compile(re.search(r'^PER = re\.compile\(r"(.*?)", re\.I\)', open(f"{R}/geo/person_focal.py").read(), re.M).group(1), re.I)
RAW = {}
for _l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
    _x = json.loads(_l)
    RAW[_x["id"]] = _x
VIDPER = {}
for _x in RAW.values():
    for _e in _x.get("depth") or []:
        if PER.search(_e["key"]):
            VIDPER.setdefault(_x["video_id"], set()).add((_e["key"], _e["t"], _e["d"]))
_pf = f"{R}/geo/out/person_focal_{split}.json"
PF = json.load(open(_pf)) if os.path.exists(_pf) else {}


def own_kind(rid):
    p = str(RAW[rid].get("prior_text") or "")
    for k, rx, _ in KINDS:
        if re.search(rx, p, re.I):
            return k
    return None


def person_row(rid, vid):
    """FPERSON entry from the row's own person distances; returns ('same', None) when they equal the video's pool"""
    own = {(e["key"], e["t"], e["d"]) for e in (RAW[rid].get("depth") or []) if PER.search(e["key"])}
    allv = VIDPER.get(vid, set())
    if vid not in PF or own == allv:
        return "same", None
    ests = [e for e in PF[vid]["ests"] if (e["key"], e["t"], e["Z"]) in own]
    keys_own = {k for k, _, _ in own}; keys_all = {k for k, _, _ in allv}
    note = f"own {len(own)} of {len(allv)} person entries, ests {len(ests)} of {len(PF[vid]['ests'])}" + (
        ", key set differs" if keys_own != keys_all else "")
    if not ests:
        return note, None
    fw = np.array([e["f_over_W"] for e in ests])
    return note, dict(PF[vid], f_over_W=float(np.median(fw)), n=len(ests), spread=float(fw.max() / fw.min()), ests=ests)


def restrict_own(vj, r, d):
    """the video json with only the row's own-text queries and the SAM2 objects they point to"""
    keys, why = own_keys(r, vj["queries"], RAW[r["id"]], mode="tok" if mode == "ownt" else "head")
    d["pool"] = dict(n_video=len(vj["queries"]), n_own=len(keys), why=why)
    vj2 = dict(vj)
    for qn in ("queries", "queries_d"):
        if qn in vj:
            vj2[qn] = [q for q in vj[qn] if q["key"] in keys]
    if "objs" in vj:
        vid = vj["video_id"]
        if vid not in OWNERS:
            OWNERS[vid] = obj_owners([f"{R}/geo/out/tracks_{split}_{c}/{vid}.json" for c in CHAIN])
        objs = {o for o in vj["objs"] if OWNERS[vid].get(o, set()) & keys}
        vj2["objs"] = {o: fr for o, fr in vj["objs"].items() if o in objs}
        d["pool"]["objs_dropped"] = sorted(set(vj["objs"]) - objs)
    return vj2


# ---- split loading: video selection and per-row time shift (as solo/meas.py)
_lst = MV.load_split_tracks


def load_split_tracks(s_, c_):
    old = os.environ.pop("QP_TSHIFT", None)
    byv, vjs = _lst(s_, c_)
    if old is not None:
        os.environ["QP_TSHIFT"] = old
    byv = {k: v for k, v in byv.items() if not SEL or k in SEL}
    if os.environ.get("GRPC_ALT_PARSE") and GROWS:                  # hook GRPC_ALT_PARSE / GRPC_ROWS
        import builtins
        pp_ = os.path.basename(os.environ["GRPC_ALT_PARSE"])
        MV.open = lambda p, *a, **k: builtins.open(os.environ["GRPC_ALT_PARSE"] if str(p).endswith(f"parse_{s_}_v2{MV.PARSE_TAG}.jsonl") else p, *a, **k)
        old_ = os.environ.pop("QP_TSHIFT", None)
        byv2_, _ = _lst(s_, c_)
        if old_ is not None:
            os.environ["QP_TSHIFT"] = old_
        del MV.open
        for v_, rs_ in byv.items():
            m2_ = {r_["id"]: r_ for r_ in byv2_.get(v_, [])}
            for j_, r_ in enumerate(rs_):
                if r_["id"] in GROWS and r_["id"] in m2_:
                    DIAG.setdefault(r_["id"], {})["alt_parse"] = dict(q_before=r_["q"], q_after=m2_[r_["id"]]["q"], src=pp_)
                    rs_[j_] = m2_[r_["id"]]
    for v_, rs_ in byv.items():                                     # hook GRPC_FIXJSON
        for r_ in rs_:
            fx_ = GFIX.get(str(r_["id"]))
            if not fx_:
                continue
            for f_, val_ in fx_.items():
                if f_ == "q":
                    r_["q"] = dict(r_["q"], **val_)
                elif val_ is None:
                    r_[f_] = None
                elif isinstance(val_, dict) and isinstance(r_.get(f_), dict):
                    r_[f_] = dict(r_[f_], **val_)
                else:
                    r_[f_] = val_
            DIAG.setdefault(r_["id"], {})["fix"] = fx_

    def shift_ok(ts, dur):
        return not (not ts or max(ts) <= dur + 0.5 or min(ts) - dur < 0 or max(ts) - dur > dur)
    for v, rs in byv.items():
        if v not in vjs or not vjs[v].get("fidx"):
            continue
        dur = (max(vjs[v]["fidx"]) + 1) / float(vjs[v]["fps"])
        tsv = [float(r["q"][k]) for r in rs for k in ("t", "t1", "t2") if r["q"].get(k) is not None]
        jsh = shift_ok(tsv, dur)
        for r in rs:
            ts = [float(r["q"][k]) for k in ("t", "t1", "t2") if r["q"].get(k) is not None]
            own = shift_ok(ts, dur)
            if own:
                for k in ("t", "t1", "t2"):
                    if r["q"].get(k) is not None:
                        r["q"][k] = float(r["q"][k]) - dur
            if ts and own != jsh:
                DIAG.setdefault(r["id"], {})["tshift"] = f"solo {own} joint {jsh} dur {dur:.2f}"
    return byv, vjs


MV.load_split_tracks = load_split_tracks

# ---- measure_video: one call per row
_mv = MV.measure_video


def measure_video(recs, vj, fdef, cfg):
    vid = vj["video_id"]
    ZCUR["src"] = str(recs[0].get("video_source", "")) if recs else None
    out, V = [], None
    for r in recs:
        CUR["id"] = r["id"]
        d = DIAG.setdefault(r["id"], {})
        fp_has, fp_old = vid in MV.FPERSON, MV.FPERSON.get(vid)
        note, fp_new = person_row(r["id"], vid)
        if note != "same":
            d["fperson"] = note
            if fp_new is None:
                MV.FPERSON.pop(vid, None)
            else:
                MV.FPERSON[vid] = fp_new
        sr_has, sr_old = vid in MV.SCENE_RULER, MV.SCENE_RULER.get(vid)
        if sr_has and own_kind(r["id"]) != sr_old.get("kind"):
            MV.SCENE_RULER.pop(vid)
            d["sruler"] = f"own kind {own_kind(r['id'])} != video kind {sr_old.get('kind')}: ruler dropped"
        vjr = restrict_own(vj, r, d) if mode != "solo" else vj
        if str(r["id"]) in GRAFT:                                   # hook GRPC_GRAFT
            vjr = graft(vjr, r, GRAFT[str(r["id"])], d)
        o, V = _mv([r], vjr, fdef, cfg)
        if fp_has:
            MV.FPERSON[vid] = fp_old
        else:
            MV.FPERSON.pop(vid, None)
        if sr_has:
            MV.SCENE_RULER[vid] = sr_old
        d["rot"] = MV.ROT_CUR
        if V.colour_swaps:
            POOLLOG.setdefault(r["id"], {})["colour"] = [[h, ks, [int(x) for x in cur], [int(x) for x in best]]
                                                         for h, ks, cur, best in V.colour_swaps]
        out += o
    return out, V


MV.measure_video = measure_video


def graft(vj, r, g, d):
    """add the alias query (a query of tracks_<split>_<src> whose expr names g['x']) under the row's target key"""
    p3 = f"{R}/geo/out/tracks_{split}_{g['src']}/{vj['video_id']}.json"
    if not os.path.exists(p3):
        d["graft"] = "no file " + p3; return vj
    vj3 = json.load(open(p3)); x = g["x"]
    k = MV.Video.key(r["target"])
    if any(q["key"] == k for q in vj.get("queries", [])):
        d["graft"] = f"own pool already has {k!r}"; return vj
    qa = None
    for q in vj3.get("queries", []):
        e = (q.get("expr") or "").lower()
        if q.get("obj") is not None and x and re.search(r"\b" + re.escape(x) + r"\b", e):
            qa = q; break
    if qa is None:
        d["graft"] = f"no {g['src']} query names {x!r}"; return vj
    if vj.get("fidx") != vj3.get("fidx"):
        d["graft"] = "frame mismatch"; return vj
    vj2 = dict(vj)
    v1fmt = vj.get("version") != "v2" and "objs" not in vj and any(q.get("frames") for q in vj.get("queries", []))
    if v1fmt:                                                       # v1 file: the track lives in the query's frames
        qn = dict(qa, key=k, phrase=x, where=[], frames=vj3["objs"][str(qa["obj"])]); qn.pop("obj", None); oid = "frames"
    else:
        vj2["objs"] = dict(vj.get("objs", {}))
        oid = str(10000 + int(qa["obj"])) if str(qa["obj"]) in vj2["objs"] else str(qa["obj"])
        vj2["objs"][oid] = vj3["objs"][str(qa["obj"])]
        qn = dict(qa, key=k, phrase=x, where=[], obj=int(oid))
    for qn_ in ("queries", "queries_d"):
        if qn_ in vj:
            vj2[qn_] = list(vj[qn_]) + [dict(qn)]
    d["graft"] = dict(key=k, src=g["src"], obj=qa["obj"], as_obj=oid, expr=qa.get("expr"), fmt="v1" if v1fmt else "objs")
    return vj2

# ---- every geo/out write goes to cache/tmp/ag_ownt_<name>
_to_csv = pd.DataFrame.to_csv


def _redir(self, path=None, *a, **k):
    if isinstance(path, str) and path.startswith(f"{R}/geo/out/"):
        path = f"{R}/cache/tmp/ag_ownt_" + os.path.basename(path)
    return _to_csv(self, path, *a, **k)


pd.DataFrame.to_csv = _redir
tag = f"_{var}_{mode}{SUF}"
sys.argv = ["measure_v3.py", "--split", split, "--fix", "abcdef", "--tag", tag, "--fdef_json",
            os.environ.get("QP_FDEF_JSON", f"{R}/analysis/v3/compliance/focal_spec_all.json"),
            "--moge", "--moge_opts", "fbrej", "--tracks", TRK]
if os.environ.get("SIL_NO_MOGE") == "1":        # as rules/seg2d_meas.py (the seg2d guard run used it)
    sys.argv = [x for x in sys.argv if x not in ("--moge", "--moge_opts", "fbrej")]
if os.environ.get("GRPC_NOMOGE") == "1":
    sys.argv = [x for x in sys.argv if x not in ("--moge", "--moge_opts", "fbrej")]
if os.environ.get("GRPC_PARSE_TAG"):
    sys.argv += ["--parse_tag", os.environ["GRPC_PARSE_TAG"]]
MV.main()
os.replace(f"{R}/cache/tmp/ag_ownt_meas_{split}_v2{tag}.csv", f"{R}/cache/tmp/ag_ownt_meas_{split}_{var}_{mode}{SUF}.csv")
for rid, lg in POOLLOG.items():
    DIAG.setdefault(rid, {})["log"] = lg
json.dump(DIAG, open(f"{R}/cache/tmp/ag_ownt_diag_{split}_{var}_{mode}{SUF}.json", "w"), default=str)
print("done", split, var, mode, SUF, "rows with pool log", len(POOLLOG))
