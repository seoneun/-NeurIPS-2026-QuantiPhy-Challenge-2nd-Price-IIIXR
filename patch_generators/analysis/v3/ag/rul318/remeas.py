"""rul318: re-measure caw_nogeo rows (no geo, answer = CaW alone) whose geo failed for a parse / detection-query reason
that a code rule can repair from tracks that already exist.  geo/measure_v3.py is imported read-only (through the rul329
helper, which sets the same QP_* environment as the rj5sQ chain); nothing is re-detected.

Rules (each a code condition on parse + track files; labels are never read here):
  ALIAS   target_no_track: the target key has no object, and exactly one OTHER query of the same pipeline track file
          (same config) names the same head noun with a superset of the target's words and has an object
          ('yellow car' -> 'turning yellow car').  The question names the object more briefly than another question did.
  V3PAIR  two-instance distance ('the distance between the two X', target2 '#second' / the same key) whose pipeline
          objects collapsed (one object for both, or none for the second): the VLM-anchored v3nt file of the same frames
          has two different objects for the two keys.  Both must be localized instances: >= 80 % frame coverage, area
          ratio <= 2, area < 10 % of the frame, boxes clipped by at most one frame border, median centroid distance
          > 1.5 x the instance size, and neither box inside the other.  Both v3nt objects are grafted.
  PSOLO   part_target where the part word IS the object: the target phrase's head noun is a PART word, the question names
          no owner (no "of the ...", no possessive, no compound "<noun> <part>"), and the pipeline has its own object for
          that phrase (distinct from the prior object's).  ('the diameter of the handle' in air hockey = the mallet).
  LOWDET  target_low_det confirmed by the independent VLM-anchored grounding: the v3nt object of the same key overlaps
          the pipeline object (median IoU >= 0.5 over the shared frames, >= 80 % of frames) -> detection accepted.
usage: python remeas.py <split> [rules]   -> cache/tmp/ag_ngparse_meas_<split>.json"""
import os, sys, json, copy, re
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
import importlib.util
_spec = importlib.util.spec_from_file_location("pw_remeas", f"{R}/analysis/v3/ag/rul329/remeas.py")
PW = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PW)   # sets the QP_* env, imports geo/measure_v3 (read-only)
MV = PW.MV
import numpy as np, pandas as pd
sys.path.insert(0, f"{R}/geo")
import track_common as tc

RAWN = "geo_v4ntrj5sQ2srngsr_d_vw"
SUB = "sub_xt_Nzi"
NEVER = r"(?!x)x"


def words(p):
    return set(re.findall(r"[a-z]+", (p or "").lower())) - {"the", "a", "an", "of"}


def oinfo(vj, o):
    if o is None:
        return None
    fr = vj["objs"].get(str(o))
    if fr is None:
        return None
    ok = [j for j, f in enumerate(fr) if f and f.get("st")]
    if not ok:
        return None
    B = np.array([fr[j]["st"]["box"] for j in ok], float)
    A = np.array([fr[j]["st"]["area"] for j in ok], float)
    C = np.array([fr[j]["st"]["c"] for j in ok], float)
    return dict(ok=ok, cov=len(ok) / len(fr), B=B, A=A, C=C, area=float(np.median(A)), size=float(np.sqrt(np.median(A))))


def qobj(vj, key):
    for q in vj.get("queries", []):
        if q.get("key") == key:
            return q.get("obj"), q
    return None, None


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def borders(b, W, H, m=2):
    return int(b[0] <= m) + int(b[1] <= m) + int(b[2] >= W - m) + int(b[3] >= H - m)


def localized(vj, I):
    W, H = vj["W"], vj["H"]
    nb = np.median([borders(b, W, H) for b in I["B"]])
    return I["cov"] >= 0.8 and I["area"] < 0.10 * W * H and nb <= 1


def inside(I1, I2):
    """median fraction of box 1 inside box 2"""
    fr = []
    for b1, b2 in zip(I1["B"], I2["B"]):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1]); x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        fr.append(max(0, x2 - x1) * max(0, y2 - y1) / a1 if a1 > 0 else 0)
    return float(np.median(fr)) if fr else 0.0


def owned(question, part):
    """does the question name an owner of the part word? ('of the X' after it, a possessive or a noun before it)"""
    ql = question.lower().replace("’", "'")
    if re.search(r"\b" + part + r"s?\s+(?:on|of)\s+(?:the|a|an)\s+(?!left\b|right\b|top\b|bottom\b|front\b|back\b)[a-z]", ql):
        return True
    if re.search(r"[a-z]'s\s+(?:[a-z]+\s+)?" + part + r"\b", ql):
        return True
    for m in re.finditer(r"\b(?:the|a|an)\s+([a-z]+)\s+" + part + r"s?\b", ql):     # compound '<noun> <part>' ('door handle')
        if not re.fullmatch(r"left|right|top|bottom|front|back|small|big|large|red|blue|green|white|black|yellow|two|both|other", m.group(1)):
            return True
    return False


def run(split, rules=("ALIAS", "V3PAIR"), force_ids=None, out_tag=""):
    """PSOLO and LOWDET are kept for the record but are off by default: PSOLO's one test row (281, simulation_0126) has
    a 'handle' track on the score display (and 'puck' on a mallet), LOWDET confirms nothing (IoU 0.47 / 0.00)."""
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        r = json.loads(l); P[r["id"]] = r
    suf = "_detail.csv" if split == "test" else "_valdetail.csv"
    d = pd.read_csv(f"{R}/submissions/{SUB}{suf}").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWN}_raw.csv").set_index("Unnamed: 0")
    src = d.answer_source.astype(str).str.replace("^mo_guard:", "", regex=True)
    why = raw.meas_why.reindex(d.index).astype(str).map(lambda s: s.split(":")[0].split("(")[0])
    MY = {"unparsed_or_other", "part_target", "target_px_fail", "target_low_det", "target_no_track", "target2_no_track", "target=target2"}
    cand = [int(i) for i in d.index[(src == "caw_nogeo") & why.isin(MY)]] if force_ids is None else [i for i in force_ids if i in d.index]
    if force_ids is not None:
        why = why.where(~why.index.isin(force_ids), "forced")
    vids = sorted({P[i]["video_id"] for i in cand})
    fmed = PW.setup(split, set(vids))
    byv, vjs = MV.load_split_tracks(split, "e")
    res = {}
    for v in vids:
        ids = [i for i in cand if P[i]["video_id"] == v]
        vj = vjs.get(v)
        if vj is None or vj.get("version") != "v2":
            continue
        p3 = f"{R}/geo/out/tracks_{split}_v3nt/{v}.json"
        vj3 = json.load(open(p3)) if os.path.exists(p3) else None
        same_frames = vj3 is not None and vj3.get("fidx") == vj.get("fidx")
        rs = byv[v]
        rmap = {r["id"]: r for r in rs}
        for i in ids:
            if i not in rmap:
                continue
            w = why[i]
            r0 = rmap[i]
            base = copy.deepcopy(vj)
            rs2 = copy.deepcopy(rs)
            r = [x for x in rs2 if x["id"] == i][0]
            V0 = MV.Video(base, "e")
            tk = V0.key(r["target"]) if r.get("target") else None
            rule, note, part_off = None, "", False
            # ------------------------------------------------------------------ ALIAS
            if "ALIAS" in rules and w in ("target_no_track", "forced") and tk:
                o, _ = qobj(base, tk)
                tw = words(r["target"]["phrase"]); hn = tc.head_noun(r["target"]["phrase"])
                al = {}
                for q in base["queries"]:
                    if q.get("key") == tk or q.get("obj") is None:
                        continue
                    ph = q.get("phrase") or q["key"].split("|")[0]
                    if tc.head_noun(ph) == hn and words(ph) > tw and oinfo(base, q["obj"]) is not None:
                        al.setdefault(q["obj"], []).append(q["key"])
                # a bare head noun ('car', 'robot') may name any instance: in the v2sc files 7 of 61 short/superset
                # pairs were different objects, all of them single-word short phrases; with a modifier 0 of 54 were
                if o is None and len(al) == 1 and len(tw) >= 2:
                    ob, ks = list(al.items())[0]
                    ph = ks[0].split("|")[0]
                    r["target"] = dict(r["target"], phrase=ph, raw=ph, where=[])
                    rule, note = "ALIAS", f"'{tk}' has no track; the one superset query '{ks[0]}' (obj {ob})"
                else:
                    note = f"alias candidates {al}"
            # ------------------------------------------------------------------ V3PAIR
            if rule is None and "V3PAIR" in rules and r["q"].get("kind") == "dist2" and r.get("target2") and tk and same_frames:
                k2 = V0.key(r["target2"])
                o1, _ = qobj(base, tk); o2, _ = qobj(base, k2)
                coll = (o2 is None) or (o1 == o2) or (k2 == tk) or force_ids is not None
                second = (r["target2"].get("where") == ["#second"]) or (k2 == tk) or (
                    tc.head_noun(r["target"]["phrase"]) == tc.head_noun(r["target2"]["phrase"]))
                if coll and second:
                    a1, q1 = qobj(vj3, tk); a2, q2 = qobj(vj3, k2)
                    I1, I2 = oinfo(vj3, a1), oinfo(vj3, a2)
                    ok = (a1 is not None and a2 is not None and a1 != a2 and I1 and I2)
                    if ok:
                        dist = float(np.median(np.linalg.norm(I1["C"][:min(len(I1["C"]), len(I2["C"]))] - I2["C"][:min(len(I1["C"]), len(I2["C"]))], axis=1)))
                        sz = max(I1["size"], I2["size"])
                        conds = dict(loc1=localized(vj3, I1), loc2=localized(vj3, I2), arat=max(I1["area"], I2["area"]) / min(I1["area"], I2["area"]) <= 2,
                                     sep=dist > 1.5 * sz, nest=max(inside(I1, I2), inside(I2, I1)) < 0.5)
                        if all(conds.values()):
                            for kk, aa, qq, tag in ((tk, a1, q1, 20000), (k2, a2, q2, 21000)):
                                oid = str(tag + int(aa)); base["objs"][oid] = vj3["objs"][str(aa)]
                                hit = False
                                for qs in ("queries", "queries_d"):
                                    for q in base.get(qs, []):
                                        if q.get("key") == kk:
                                            q["obj"] = int(oid); hit = True
                                if not hit:
                                    qn = dict(qq, key=kk, obj=int(oid)); base["queries"].append(qn)
                            if k2 == tk:                  # one key for both: give target2 its own key
                                k2n = tk + "#v3second"
                                oid = str(21000 + int(a2))
                                base["queries"].append(dict(q2, key=k2n, obj=int(oid)))
                                r["target2"] = dict(r["target2"], where=list(r["target2"].get("where") or []) + ["#v3second"])
                            rule, note = "V3PAIR", f"pipeline objs {o1}/{o2} collapsed; v3nt objs {a1}/{a2} dist {dist:.0f}px size {sz:.0f}px"
                        else:
                            note = f"v3pair rejected {conds}"
                    else:
                        note = f"v3pair: v3nt objs {a1}/{a2}"
            # ------------------------------------------------------------------ PSOLO
            if rule is None and "PSOLO" in rules and w in ("part_target", "forced") and tk and r["q"].get("kind") == "size":
                hn = tc.head_noun(r["target"]["phrase"])
                if re.fullmatch(MV.PART, hn) or re.fullmatch(MV.PART, hn + "s"):
                    o, _ = qobj(base, tk)
                    po, _ = qobj(base, V0.key(r["prior_obj"])) if r["prior_obj"].get("phrase") else (None, None)
                    if not owned(r["question"], hn) and o is not None and o != po and oinfo(base, o):
                        rule, note, part_off = "PSOLO", f"part word '{hn}' names the object itself (no owner); obj {o}", True
                    else:
                        note = f"psolo: owned={owned(r['question'], hn)} obj {o} prior obj {po}"
            # ------------------------------------------------------------------ LOWDET
            if rule is None and "LOWDET" in rules and w in ("target_low_det", "forced") and tk and vj3 is not None and same_frames:
                o, q0 = qobj(base, tk); a, q3 = qobj(vj3, tk)
                I0, I3 = oinfo(base, o), oinfo(vj3, a)
                if I0 and I3:
                    fr0, fr3 = base["objs"][str(o)], vj3["objs"][str(a)]
                    ious = [iou(f0["st"]["box"], f3["st"]["box"]) for f0, f3 in zip(fr0, fr3) if f0 and f0.get("st") and f3 and f3.get("st")]
                    mi = float(np.median(ious)) if ious else 0.0
                    shared = len(ious) / len(fr0)
                    if mi >= 0.5 and shared >= 0.8:
                        for qs in ("queries", "queries_d"):
                            for q in base.get(qs, []):
                                if q.get("key") == tk:
                                    q["best_gd"] = max(q.get("best_gd") or 0, 0.3)
                        rule, note = "LOWDET", f"low-score detection (gd {q0.get('best_gd')}) confirmed by v3nt obj {a}: median IoU {mi:.2f} on {shared:.0%} of frames"
                    else:
                        note = f"lowdet: v3nt obj {a} IoU {mi:.2f} shared {shared:.0%}"
                else:
                    note = f"lowdet: v3nt obj {a}"
            if rule is None:
                print(f"  {i} {v} [{w}] no rule: {note}")
                continue
            if part_off:
                p0 = MV.PART; MV.PART = NEVER
            try:
                o, _V = PW.measure(rs2, base, fmed)
            finally:
                if part_off:
                    MV.PART = p0
            x = o[i]
            print(f"  {i} {v} [{w}] {rule}: new {x['value']:.4g} {x['mode']} why={x['why']} | CaW {d.cs[i]:.4g} ans {d.answer[i]:.4g} | {note}")
            res[str(i)] = dict(new=x["value"], mode=x["mode"], f=x["f"], why=x["why"], rule=rule, note=note, vid=v)
        # siblings reproduce the raw?
    json.dump(dict(res=res), open(f"{R}/cache/tmp/ag_ngparse_meas{out_tag}_{split}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    sp = sys.argv[1]
    rules = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("ALIAS", "V3PAIR")
    run(sp, rules)
