"""rul330: whole-clip path of the PRIOR OBJECT under a constant-speed prior = v x T (kin vT with a wider object match).

Rows: simulation / segmentation videos; prior 'speed of X = v' (pkind speed, not windowed, no time clause); question asks
for the distance / path / travel / displacement of the prior object with no time (t, t1, t2, when all None, no target2);
current answer an agreement hedge (agree_gm / agree_wgm, i.e. the pipeline's kin vT branch did not fire because
K.obj(prior) != K.obj(question): 'chess' vs 'chess piece', 'bird' vs 'total length of the bird', 'walking speed' vs 'man').
Physics: geo's ruler for this row IS the prior (metres per pixel = v / pixel speed of the prior object), so when the
subject is the prior object, geo's path = v x (track time span) exactly; pixels cancel. A constant-speed object whose
track spans the whole clip travels v x T. The hedge with CaW only adds the VLM's error to an identity.
Gates (all code on inputs):
  * subject = prior object: parse target's track obj == prior_obj's track obj; or head nouns match (one phrase's head is
    a word of the other); or the parse target is None / unresolved (ok False or best_score < 0.4) and the question
    contains the prior object's head noun (walking-speed prior: a person noun counts);
  * identity |geo - v (N_track - 1)/fps| <= 5 % geo on the mover track (fps = CSV fps, never the container);
  * constant motion on the mover track: 0.25 s chord pixel speeds have min >= 0.5 x median and CV <= 0.3;
  * meas_mode 2d_ratio / 3d_calib.
Output: geo (the kin vT value; --kin ...:vT takes kin rows as is, no gm).
writes cache/tmp/ag_pathvt_patch_{test,val}.json"""
import sys, json, re, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/cache/tmp")
import kinematic as K_
import track_common as tc
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 400); pd.set_option('display.max_colwidth', 70)
SUB = os.environ.get("PV_SUB", "sub_xt_Nzl")
RAWT = "geo_v4ntrj5sQ2srngsr_d_vw"
QRE = re.compile(r"distance|path|travel|moved|walked|displacement", re.I)
WIN_RE = re.compile(r"\b(?:at|after|before|between|during|from|until|when|within)\b|\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?)\b", re.I)
PERSON = re.compile(r"\b(?:man|woman|person|people|pedestrian|walker|boy|girl|guy|lady|men|women)\b", re.I)
IDTOL, CVMAX, MINF = 0.05, 0.30, 0.5
QSET = os.environ.get("PV_QSET", "queries")       # queries (cfg e) or queries_d
REQ_SRC = os.environ.get("PV_REQ_SRC", "1") == "1"
TRK_V2 = os.environ.get("PV_TRK", "v2sc")
GUARD = os.environ.get("PV_GUARD", "1") == "1"
IDTOL = float(os.environ.get("PV_IDTOL", IDTOL)); CVMAX = float(os.environ.get("PV_CV", CVMAX)); MINF = float(os.environ.get("PV_MIN", MINF))


def words(p):
    return [tc.sing(w) for w in re.findall(r"[a-z]+", (p or "").lower())]


def heads_match(pa, pb):
    if not pa or not pb:
        return False
    ha, hb = tc.head_noun(pa), tc.head_noun(pb)
    return bool(ha and hb and (ha == hb or ha in words(pb) or hb in words(pa)))


def vkey(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def track_stats(T, q, fps):
    """frames the track exists on, span (frames), 0.25 s chord pixel speeds"""
    o = q.get("obj") if q else None
    if o is None:
        return None
    lst = T["objs"][str(o)]
    fidx = T["fidx"]
    pts = [(fidx[k], lst[k]["st"]["c"]) for k in range(len(lst)) if lst[k] and lst[k].get("st") and lst[k]["st"].get("c") is not None]
    if len(pts) < 4:
        return None
    f = np.array([p[0] for p in pts], float); c = np.array([p[1] for p in pts], float)
    step = max(1, int(round(0.25 * fps)))
    fmap = {int(ff): cc for ff, cc in zip(f, c)}
    sp = []
    fr = int(f[0])
    while fr + step <= int(f[-1]):
        if fr in fmap and fr + step in fmap:
            sp.append(float(np.hypot(*(fmap[fr + step] - fmap[fr]))) / (step / fps))
        fr += step
    return dict(n=len(pts), span=float(f[-1] - f[0]), sp=np.array(sp), f0=int(f[0]), f1=int(f[-1]), Nvid=int(T.get("N") or len(fidx)))


def run(split, verbose_all=False):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWT}_raw.csv", index_col=0)
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[int(j["id"])] = j
    rows, patch = [], {}
    for i in det.index:
        r = raw.loc[i]; pj = P.get(int(i), {})
        q = str(r.question); pr = str(r.ground_truth_prior)
        src = str(r.video_source).lower()
        if not src.startswith(("simulation", "segment")):
            continue
        qq = pj.get("q") or {}
        pri = (pj.get("priors") or [{}])[0]
        if K_.pkind(pr) != "speed" or K_.prior_windowed(pr) or any(pri.get(k) is not None for k in ("t", "t1", "t2")) or WIN_RE.search(pr.split("=")[0]):
            continue
        if "gravit" in pr.lower():
            continue
        if not QRE.search(q):
            continue
        z = dict(id=i, vid=r.video_id, cat=det.cat[i], src=det.answer_source[i], ans=det.answer[i], g=det.g[i], cs=det.cs[i], mode=r.meas_mode)
        fail = []
        if any(qq.get(k) is not None for k in ("t", "t1", "t2")) or qq.get("target2") or pj.get("target2"):
            fail.append("q_time_or_t2")
        if qq.get("when") is not None:
            fail.append("q_when")
        if REQ_SRC and z["src"] not in ("agree_gm", "agree_wgm"):
            fail.append("src")
        if str(r.meas_mode) not in ("2d_ratio", "3d_calib"):
            fail.append("mode")
        # subject = prior object
        tv = str(r.meas_tracks) if isinstance(r.meas_tracks, str) else "v2"
        tv = TRK_V2 if tv == "v2" else tv                  # the current v2 track set is v2sc; v1 rows keep v1 tracks
        tdir = f"{R}/geo/out/tracks_{split}_{tv}" if tv != "v1" else f"{R}/geo/out/tracks_{split}"
        if not os.path.exists(f"{tdir}/{r.video_id}.json"):
            tdir = f"{R}/geo/out/tracks_{split}_v2sc"
        T = json.load(open(f"{tdir}/{r.video_id}.json"))
        qs = {qd["key"]: qd for qd in (T.get(QSET) or T["queries"])}
        tg, po = pj.get("target"), pj.get("prior_obj")
        tq, pq = qs.get(vkey(tg)), qs.get(vkey(po))
        tph, pph = (tg or {}).get("phrase"), (po or {}).get("phrase")
        t_obj = tq.get("obj") if tq else None
        p_obj = pq.get("obj") if pq else None
        t_bs = float(tq.get("best_score") or 0) if tq else 0.0
        ph = tc.head_noun(pph) if pph else ""
        ql = q.lower()
        walking = "walk" in pr.lower()
        q_has_prior = bool(ph) and (ph in words(ql) or (walking and ph in ("person", "human", "pedestrian") and bool(PERSON.search(ql))))
        unresolved = (tg is None) or (qq.get("ok") is False) or (t_bs < 0.4)
        how = None
        if t_obj is not None and p_obj is not None and t_obj == p_obj:
            how = "same_obj"
        elif heads_match(tph, pph):
            how = "heads"
        elif unresolved and q_has_prior:
            how = "unresolved+prior_noun"
        if how is None:
            fail.append("subject")
        # instance guard (added): two resolved tracks that differ, or a qualifier conflict (big / small, colour, left /
        # right ...) between the question / target phrase and the prior phrase, mean a different instance of the class
        p_bs = float(pq.get("best_score") or 0) if pq else 0.0
        if GUARD:
            if t_obj is not None and p_obj is not None and t_obj != p_obj and t_bs >= 0.4 and p_bs >= 0.4:
                fail.append("diff_obj")
            wq = set(re.findall(r"[a-z]+", (ql if unresolved or not tph else tph).lower()))
            wp = set(re.findall(r"[a-z]+", (pph or "") + " " + pr.split("=")[0].lower()))
            if tc.qualifier_conflict(wq, wp):
                fail.append("qual_conflict")
        # mover track: the parse target if resolved, else the query whose head noun is in the question
        mq = None
        if tq is not None and t_bs >= 0.4 and tq.get("obj") is not None:
            mq = tq
        else:
            for k_, qd in qs.items():
                hn = tc.head_noun(qd.get("phrase"))
                if hn and (hn in words(ql) or (walking and hn == "person" and PERSON.search(ql))) and qd.get("obj") is not None:
                    if mq is None or float(qd.get("best_score") or 0) > float(mq.get("best_score") or 0):
                        mq = qd
        fps = float(r.fps)
        st = track_stats(T, mq, fps) if mq is not None else None
        v_si = pri.get("value_si")
        if v_si is None:
            vv, uu = K_.prior_value(pr)
            v_si = abs(vv) * K_.to_si(uu) if vv is not None else np.nan
        g = float(det.g[i]) if pd.notna(det.g[i]) else np.nan
        z.update(how=how, tph=tph, pph=pph, t_obj=t_obj, p_obj=p_obj, t_bs=round(t_bs, 2), mover=(mq or {}).get("key"), fps=fps, v=v_si)
        if st is None:
            fail.append("no_track")
        else:
            vT = v_si * (st["n"] - 1) / fps / K_.q_scale(q)
            vTclip = v_si * (st["Nvid"] - 1) / fps / K_.q_scale(q)
            sp = st["sp"]
            med = float(np.median(sp)) if len(sp) else np.nan
            cv = float(np.std(sp) / np.mean(sp)) if len(sp) and np.mean(sp) > 0 else np.nan
            mn = float(sp.min() / med) if len(sp) and med > 0 else np.nan
            z.update(n=st["n"], span=st["span"], Nvid=st["Nvid"], vT=vT, vTclip=vTclip, idr=g / vT if vT > 0 else np.nan, cv=cv, minr=mn, nsp=len(sp))
            if not (np.isfinite(g) and abs(g - vT) <= IDTOL * g):
                fail.append("identity")
            if not (len(sp) >= 3 and np.isfinite(cv) and cv <= CVMAX and mn >= MINF):
                fail.append("constant")
        z["fail"] = ",".join(fail)
        z["q"] = q[:70]; z["prior"] = pr[:45]
        rows.append(z)
        if not fail:
            patch[str(int(i))] = dict(value=float(g), why=f"rul330: whole-clip path = v·T (prior object, {how}; geo {g:.4g} vs v(N-1)/fps {vT:.4g})")
    Z = pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()
    return Z, patch, det


if __name__ == "__main__":
    for split in ("test", "val"):
        Z, patch, det = run(split)
        print(f"===== {split}: {len(Z)} candidate rows (sim, timeless speed prior, path-like question)")
        if len(Z):
            print(Z.drop(columns=["q", "prior"]).round(4).to_string())
            print(Z[["q", "prior", "fail"]].to_string())
        out = f"{R}/cache/tmp/ag_pathvt_patch_{split}.json"
        json.dump(patch, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"[{split}] {len(patch)} rows changed -> {out}")
        for k, v in patch.items():
            print(" ", k, det.answer[int(k)], "->", v)
