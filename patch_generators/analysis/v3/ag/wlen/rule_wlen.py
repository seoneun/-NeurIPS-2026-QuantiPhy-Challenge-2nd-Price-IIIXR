"""ag wlen (audit6 rule A5): WIDTH asked in a video that also asks the LENGTH of the same target.
If CaW's width is >= 0.8 x our final length answer for that target, CaW has answered the long side (the labelled widths of
non-winged objects are all <= 0.8 x their lengths), so CaW carries no information about the width and is dropped; the answer
becomes geo's width alone (geo measures a width as the rotated-rect MINOR axis of the target's track, i.e. the short side).
Gates (all code-computable, labels only used for the evaluation at the bottom):
  - 2D categories only (S2/D2): in 3D lab scenes the oblique view foreshortens the short side (captured_0021 table: geo 46-53
    vs truth 60, the only labelled rows the literal A5 rule changes, all made worse);
  - the target is not winged (wingspan rows belong to the wing patch) and the row is plain (no patch:*, var_ans_plain,
    patch_copy, mo_guard source; variant copies follow the plain row in submit_v3);
  - geo is self-consistent about the object being elongated: 0.15 <= g_width / g_length (geo's own length of the same
    target) <= 0.7, and where both rows also have a direct track measurement (meas_value) its w/l ratio agrees with
    g_w/g_l within 15% (internet_0022: vlm-ratio geo says 0.53, the track says 0.73 -> skipped);
  - the change is not trivial (|log(new/old)| > 1%).
Writes cache/tmp/ag_wlen_patch_{test,val}.json"""
import json, os, re, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + "/cache/tmp"); sys.path.insert(0, R + "/caw"); sys.path.insert(0, R + "/analysis/v3")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nza"
RAWN = "geo_v4ntrj5sQ2srngsr_d_vw"
ONLY2D = os.environ.get("WLEN_ALLCAT", "0") != "1"
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
WINGED = re.compile(r"\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft|hawk|crow|pigeon|parrot|"
                    r"duck|goose|swan|moth|airliner|glider|wing|wings)s?\b")
SKIP_SRC = re.compile(r"^(patch|var_ans_plain|mo_guard)")


def norm(p):
    p = (p or "").lower().strip()
    return re.sub(r"^(the|a|an)\s+", "", p)


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index("Unnamed: 0")
    raw = pd.read_csv(f"{R}/preds/{split}_{RAWN}_raw.csv").set_index("Unnamed: 0")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        o = json.loads(l); P[int(o["id"])] = o
    byv = {}
    for i, o in P.items():
        byv.setdefault(o["video_id"], []).append(i)
    patch, log = {}, []
    for v, ids in byv.items():
        for i in ids:
            o = P[i]; q = o.get("q") or {}
            if q.get("kind") != "size" or q.get("dim") != "width" or i not in det.index:
                continue
            tp = norm((o.get("target") or {}).get("phrase") or q.get("target"))
            sibs = [j for j in ids if j != i and j in det.index and (P[j].get("q") or {}).get("kind") == "size" and P[j]["q"].get("dim") == "length"
                    and norm((P[j].get("target") or {}).get("phrase") or P[j]["q"].get("target")) == tp]
            if not sibs:
                continue
            j = sibs[0]; d = det.loc[i]; dj = det.loc[j]
            cat = d["cat"]; src = str(d.answer_source)
            k = USI.get(P[j]["q"].get("unit") or "m", 1.0) / USI.get(q.get("unit") or "m", 1.0)   # length unit -> width unit
            lans, lg = float(dj.answer) * k, float(dj.g) * k
            g, cs, ans = float(d.g), float(d.cs), float(d.answer)
            mw = float(raw.loc[i, "meas_value"]) if i in raw.index else np.nan
            ml = float(raw.loc[j, "meas_value"]) * k if j in raw.index else np.nan
            rg = mw / ml if (np.isfinite(mw) and np.isfinite(ml) and ml > 0) else np.nan
            rec = dict(split=split, id=i, vid=v, cat=cat, tp=tp, ans=ans, src=src, g=g, cs=cs, lid=j, lans=lans, lg=lg, meas_wl=rg,
                       cw_l=cs / lans if lans > 0 else np.nan, g_lg=g / lg if lg > 0 else np.nan)
            why = None
            if not (cs >= 0.8 * lans):
                why = "caw_ok"
            elif ONLY2D and cat not in ("S2", "D2"):
                why = "3d"
            elif WINGED.search(tp):
                why = "winged"
            elif SKIP_SRC.search(src) or "patch" in src:
                why = "src"
            elif not (np.isfinite(g) and g > 0 and np.isfinite(lg) and lg > 0):
                why = "nogeo"
            elif not (0.15 <= g / lg <= 0.7):
                why = "geo_not_elongated"
            elif np.isfinite(rg) and abs(np.log(rg / (g / lg))) > np.log(1.15):
                why = "geo_ratio_inconsistent"
            elif abs(np.log(g / ans)) <= 0.01:
                why = "same"
            rec["gate"] = why or "FIRE"
            log.append(rec)
            if why is None:
                patch[str(i)] = dict(value=g, why=f"wlen: CaW width {cs:.3g} >= 0.8 x length answer {lans:.3g} of the same {tp} "
                                                   f"(CaW gave the long side) -> geo short side {g:.3g} alone (geo w/l {g / lg:.2f}); was {ans:.3g} {src}")
    return patch, pd.DataFrame(log)


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40); pd.set_option("display.max_rows", 300)
    tag = os.environ.get("WLEN_TAG", "wlen")
    allp = {}
    for split in ("test", "val"):
        p, lg_ = run(split)
        allp[split] = p
        json.dump(p, open(f"{R}/cache/tmp/ag_{tag}_patch_{split}.json", "w"), indent=1)
        print(f"[{split}] pairs {len(lg_)} gate counts", lg_.gate.value_counts().to_dict(), "-> patched", len(p))
        if len(lg_):
            print(lg_[lg_.gate.isin(["FIRE", "geo_not_elongated", "geo_ratio_inconsistent", "same"])].round(3).to_string())
