"""rul044 step 1: time-window text fix -> re-parse -> diff.
Trigger (question text only):
  W1  '(from|between) A s (and|to) B' whose second number lacks the seconds unit  ('between 0.20s and 0.30 in meters')
  W2  '(from|between) A (and|to) B' with both numbers bare decimals followed by '[,] in <unit>'  ('from 0.5 to 5.53 in m/s')
Both are rewritten to '<from|between> As <and|to> Bs'; geo/parse_v2.py (imported read-only, its OUT redirected) then
re-parses the split.  A row is a rul044 candidate when its parse changes ONLY in q.t1 / q.t2 (None -> number, t2 > t1)
plus the question-target string (the window is no longer glued to it); the effective target phrase, kind, unit, priors,
depth keys stay identical.
usage: python winparse.py -> analysis/v3/ag/rul044/out/parse_{split}_v2_win.jsonl, out/cand_{split}.json"""
import os, sys, re, json, io, contextlib
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
OD = f"{R}/analysis/v3/ag/rul044/out"
os.makedirs(OD, exist_ok=True)
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import parse_v2 as PV
import pandas as pd
LOAD0 = PV.load_split

W1 = re.compile(r"(?i)\b(between|from)\s+(\d*\.?\d+)\s*s\s+(and|to)\s+(\d*\.?\d+)(?![\d.])(?!\s*s(?:ec|econds?)?\b)")
W2 = re.compile(r"(?i)\b(between|from)\s+(\d*\.\d+)\s+(and|to)\s+(\d*\.\d+)(?=\s*,?\s*in\s+[a-z])")


def norm_win(s):
    if s is None or (isinstance(s, float) and s != s):
        return s
    s = str(s)
    s = W1.sub(r"\1 \2s \3 \4s", s)
    s = W2.sub(r"\1 \2s \3 \4s", s)
    return s


def run(split):
    df0 = LOAD0(split)
    df = df0.copy()
    df["question"] = df.question.map(norm_win)
    tmp = f"{OD}/_pv_{split}"
    os.makedirs(tmp, exist_ok=True)
    for f in (f"parse_{split}.jsonl", f"llm_parse_{split}.jsonl"):
        if not os.path.exists(f"{tmp}/{f}"):
            os.symlink(f"{R}/geo/out/{f}", f"{tmp}/{f}")
    PV.OUT = tmp
    PV.load_split = lambda sp: df
    sys.argv = ["parse_v2", "--split", split]
    with contextlib.redirect_stdout(io.StringIO()):
        PV.main()
    os.replace(f"{tmp}/parse_{split}_v2.jsonl", f"{OD}/parse_{split}_v2_win.jsonl")
    return {int(r["Unnamed: 0"]): (r.question, df.loc[i, "question"]) for i, r in df0.iterrows()}


def sig(r):
    q = r["q"]
    d = dict(kind=q.get("kind"), unit=q.get("unit"), t=q.get("t"), t1=q.get("t1"), t2=q.get("t2"), when=q.get("when"),
             qtarget=q.get("target"), qtarget2=q.get("target2"), dim=q.get("dim"), ok=q.get("ok"),
             target=(r.get("target") or {}).get("phrase"), where=tuple((r.get("target") or {}).get("where") or []),
             target2=(r.get("target2") or {}).get("phrase"), prior_obj=(r.get("prior_obj") or {}).get("phrase"),
             tdk=tuple(r.get("target_dkeys") or []), t2dk=tuple(r.get("target2_dkeys") or []), pdk=tuple(r.get("prior_dkeys") or []))
    d["depth"] = tuple((x["key"], x["t"], x["d"]) for x in (r.get("depth") or []))
    for j, p in enumerate(r.get("priors") or []):
        for k in ("value_si", "obj", "kind", "dim", "t", "t1", "t2", "gravity"):
            d[f"p{j}.{k}"] = p.get(k)
    return d


if __name__ == "__main__":
    for sp in ("test", "val"):
        txt = run(sp)
        cur = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl")}
        new = {json.loads(l)["id"]: json.loads(l) for l in open(f"{OD}/parse_{sp}_v2_win.jsonl")}
        ntxt = sum(1 for v in txt.values() if v[0] != v[1])
        cand, other = {}, []
        for i in sorted(cur):
            a, b = sig(cur[i]), sig(new[i])
            diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
            if not diff:
                if txt[i][0] != txt[i][1]:
                    print(f"  {sp}:{i} text changed, parse identical: {txt[i][0].strip()!r}")
                continue
            ok = (set(diff) <= {"t1", "t2", "qtarget"} and "t1" in diff and "t2" in diff and diff["t1"][0] is None and diff["t2"][0] is None
                  and diff["t1"][1] is not None and diff["t2"][1] is not None and float(diff["t2"][1]) > float(diff["t1"][1]))
            rec = dict(vid=cur[i]["video_id"], cat=cur[i]["cat"], q_old=txt[i][0], q_new=txt[i][1], diff={k: list(v) for k, v in diff.items()})
            if ok:
                cand[i] = rec
            else:
                other.append((i, rec))
        print(f"{sp}: questions rewritten {ntxt}, candidates (t1/t2 only) {len(cand)}, other parse changes {len(other)}")
        for i, c in cand.items():
            print(f"  CAND {sp}:{i} {c['vid']} {c['cat']} t1={c['diff']['t1'][1]} t2={c['diff']['t2'][1]} | {c['q_old'].strip()!r}")
        for i, c in other:
            print(f"  OTHER {sp}:{i} {c['vid']} {c['diff']} | {c['q_old'].strip()!r}")
        json.dump({str(k): v for k, v in cand.items()}, open(f"{OD}/cand_{sp}.json", "w"), indent=1, default=str)
