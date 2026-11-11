"""grpB_ownpool.py: own-text track pool for the tedge-family builders (check_repair grpB, 2026-10-17).

The tedge-family scripts (tedge/diag.py, rul433/diag_own.py, rul015/diag3d.py, rul127/diag.py, rul146.py,
rul182.py) read a row's target track (and diag_own also the prior object's track) with measure_v3.Video(vj, "e").get(
<own parse entity>). Video.get looks the track up by the row's own key (phrase|where|rel), but Video.__init__ first runs
colour_assign (geo/measure_v3.py:239-276) over ALL queries of the video, i.e. over the phrases of other questions: it can
re-assign the row's track to the SAM2 object of another question's phrase (checked 10-17: the only video where it fires is
test internet_0012, v2sc tracks, grey car / white car). With env QP_SHARED_WEAK=1 flag_shared also reads all queries.

This module patches measure_v3 (imported from <QP_ROOT>/geo) so that every Video.get(ent) whose ent is an entity of a row
record (target / target2 / prior_obj of a record returned by MV.load_split_tracks) returns the track of a Video built from
the row's OWN-TEXT queries only (analysis/v3/compliance/owntracks/owntext.py own_keys, mode 'tok': the own parse entities
and the phrases named in the row's question, prior text or depth_info keys), as owntracks/meas.py restrict_own does.
When no subset of the video's queries can change the lookup (no colour group of >= 2 colour-named queries with one head
noun, not a v2 'e' track file, QP_SHARED_WEAK unset) the original lookup is returned unchanged (exact: colour_assign
needs >= 2 members of one group). Track objects are otherwise identical (Track(q, vj, objs[q.obj])).

usage as a runner: python grpB_ownpool.py <builder path relative to the repo root> [builder argv ...]
  (runs the builder with runpy as __main__ in the same process after the patch; prints the pool statistics to stderr)
used as a module by the grpB wrappers (<tag>_rep.py, <tag>_pre_*.py) through run(path, argv, env, skip).

skip=(TAG, LISTVAR): skip-if-patched gate as in production. The production base detail (e.g. sub_xt_Nzz_detail.csv) marks
rows set by an applied patch of its round list with answer_source 'patch:<tag>', and every grpB builder skips such rows
(rul433/rule.py f_patch; rul015 'patch' source gate; rul146 / rul182 g_unpatched). The clean base
sub_xt_clean4f has no applied patch, so here each read of submissions/sub_xt_*_{detail,valdetail}.csv with index_col=0 gets
answer_source 'patch:<t>' on the rows that a tag t of the round list LISTVAR (scripts/fusion_recipe.sh; t before TAG,
TAG itself excluded) writes in the shadow's cache/tmp/ag_<t>_patch_<split>.json (re-run patches only: the SEED copy of
the clean4 re-run and the earlier lines of this run; rows with a finite value > 0; the last such t in list order).
LISTVAR 'P_NZZ48A<' = the tags of P_NZZ48A before TAG (used when the base round's list is not defined in the recipe)."""
import os, sys, json, runpy, collections
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
if R + "/geo" not in sys.path:
    sys.path.insert(0, R + "/geo")
if R + "/analysis/v3/compliance/owntracks" not in sys.path:
    sys.path.insert(0, R + "/analysis/v3/compliance/owntracks")
import measure_v3 as MV  # noqa: E402
from owntext import own_keys  # noqa: E402

ENT = {}                      # id(entity dict) -> (entity, record, split)
RAW = {}                      # split -> {id: unmodified parse record}
CACHE = {}                    # (id(Video), frozenset(own keys)) -> (Video, restricted Video)
STATS = collections.Counter()
CHANGED = []                  # (split, row id, key, obj full pool, obj own pool)
_orig_load, _orig_init, _orig_get = MV.load_split_tracks, MV.Video.__init__, MV.Video.get


def _parse_path(split):
    return f"{R}/geo/out/parse_{split}_v2{MV.PARSE_TAG}.jsonl" if "d" in MV.FIX else f"{R}/geo/out/parse_{split}.jsonl"


def load_split_tracks(split, cfg):
    byv, vjs = _orig_load(split, cfg)
    if split not in RAW:
        RAW[split] = {}
        for l in open(_parse_path(split)):
            j = json.loads(l); RAW[split][j["id"]] = j
    for recs in byv.values():
        for r in recs:
            for k in ("target", "target2", "prior_obj"):
                e = r.get(k)
                if isinstance(e, dict):
                    ENT[id(e)] = (e, r, split)
    return byv, vjs


def __init__(self, vj, cfg):
    self._op_cfg = cfg
    _orig_init(self, vj, cfg)


def _may_differ(V):
    """True when some subset of the video's queries could resolve a key to another track (see module doc)"""
    if os.environ.get("QP_SHARED_WEAK") == "1":
        return True
    if getattr(V, "_op_cfg", None) != "e" or V.vj.get("version") != "v2":
        return False
    groups = collections.Counter()
    for q in V.vj.get("queries", []):
        ph = q.get("phrase") or ""
        cw = [c for c in MV.tc.COLOR_WORDS if MV.re.search(r"\b" + c + r"\b", ph)]
        if len(cw) == 1 and q.get("obj") is not None:
            groups[MV.tc.head_noun(ph)] += 1
    return any(n >= 2 for n in groups.values())


def get(self, ent):
    m = ENT.get(id(ent)) if isinstance(ent, dict) else None
    if m is None or m[0] is not ent or getattr(self, "_op_inner", False):
        STATS["get_unregistered"] += 1
        return _orig_get(self, ent)
    if not _may_differ(self):
        STATS["get_same_nocolour"] += 1
        return _orig_get(self, ent)
    _, rec, split = m
    qn = "queries" if self._op_cfg == "e" else "queries_d"
    qs = self.vj.get(qn, [])
    keys, _ = own_keys(rec, qs, RAW[split].get(rec["id"], rec), mode="tok")
    if keys >= {q["key"] for q in qs}:
        STATS["get_same_fullpool"] += 1
        return _orig_get(self, ent)
    ck = (id(self), frozenset(keys))
    if ck not in CACHE:
        vj2 = dict(self.vj)
        for n_ in ("queries", "queries_d"):
            if n_ in vj2:
                vj2[n_] = [q for q in vj2[n_] if q["key"] in keys]
        V2 = MV.Video(vj2, self._op_cfg); V2._op_inner = True
        CACHE[ck] = (self, V2)
    V2 = CACHE[ck][1]
    t_old, t_new = _orig_get(self, ent), _orig_get(V2, ent)
    o_old = None if t_old is None else t_old.q.get("obj")
    o_new = None if t_new is None else t_new.q.get("obj")
    if (t_old is None) == (t_new is None) and o_old == o_new:
        STATS["get_same_ownpool"] += 1
        return t_old
    STATS["get_changed_ownpool"] += 1
    CHANGED.append((split, rec["id"], MV.Video.key(ent), o_old, o_new))
    return t_new


MV.load_split_tracks = load_split_tracks
MV.Video.__init__ = __init__
MV.Video.get = get


def report():
    print(f"[grpB_ownpool] Video.get calls: {dict(STATS)}; own-pool track changes: {CHANGED}", file=sys.stderr)


SKIP = {}                     # split -> {row id: tag}


def install_skip(tag, listvar):
    import math, subprocess
    import pandas as pd
    v = listvar.rstrip("<")
    lst = subprocess.run(["bash", "-c", 'source scripts/fusion_recipe.sh >/dev/null 2>&1; printf "%s" "${' + v + '}"'],
                         cwd=R, capture_output=True, text=True, check=True).stdout.strip().split(",")
    lst = [t.strip() for t in lst if t.strip()]
    if not lst:
        sys.exit(f"empty round list {listvar}")
    if listvar.endswith("<"):
        lst = lst[:lst.index(tag)]
    lst = [t for t in lst if t != tag]
    for split in ("test", "val"):
        m, nf = {}, 0
        for t in lst:
            p = f"{R}/cache/tmp/ag_{t}_patch_{split}.json"
            if not os.path.exists(p):
                nf += 1; continue
            for k, x in json.load(open(p)).items():
                x = x.get("value") if isinstance(x, dict) else x
                try:
                    ok = x is not None and math.isfinite(float(x)) and float(x) > 0
                except (TypeError, ValueError):
                    ok = False
                if ok:
                    m[int(k)] = t
        SKIP[split] = m
        print(f"[grpB_ownpool] skip-if-patched {tag}: list {listvar} ({len(lst)} tags, {nf} without a {split} patch file in the "
              f"shadow); {len(m)} {split} rows marked patched", file=sys.stderr)
    _rc = pd.read_csv

    def read_csv(*a, **k):
        df = _rc(*a, **k)
        f = str(a[0]) if a else str(k.get("filepath_or_buffer", ""))
        b = os.path.basename(f)
        if k.get("index_col") == 0 and "/submissions/" in f and b.startswith("sub_xt_") and "answer_source" in df.columns:
            sp = "val" if b.endswith("_valdetail.csv") else ("test" if b.endswith("_detail.csv") else None)
            if sp:
                hit = [i for i in df.index if int(i) in SKIP[sp]]
                if hit:
                    df = df.copy()
                    df["answer_source"] = df["answer_source"].astype(object)
                    for i in hit:
                        df.at[i, "answer_source"] = "patch:" + SKIP[sp][int(i)]
                STATS[f"marked_{sp}"] += len(hit)
        return df
    pd.read_csv = read_csv


def run(path, argv, env=None, skip=None):
    """run the builder <QP_ROOT>/<path> as __main__ with argv (list), extra env (dict) and skip=(TAG, LISTVAR)"""
    for k, v in (env or {}).items():
        os.environ[k] = v
    if skip:
        install_skip(*skip)
    p = os.path.join(R, path)
    sys.argv = [p] + list(argv)
    sys.path.insert(0, os.path.dirname(p))        # as `python <builder>` does
    try:
        runpy.run_path(p, run_name="__main__")
    finally:
        report()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
