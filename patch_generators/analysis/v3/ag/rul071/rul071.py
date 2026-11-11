"""rul071: regulation net length on a regulation court (coverage gap of submit_v3 --std_court).

std_court answers only the court's other regulation dimension.  A question 'length of the net' on a court whose stated prior
matches a regulation court dimension (within 3 %) is answered with the regulation post-to-post net length of that sport:
  pickleball 22 ft = 6.706 m (posts 12 in outside each 20 ft sideline), tennis 42 ft = 12.80 m (post centres 3 ft outside the
  doubles sidelines), badminton 6.10 m (posts on the doubles sidelines), volleyball 9.50 m.
Why: in sunny top-down clips the net mask merges with its cast shadow (internet_0074: white tape 88-430 px between the posts,
shadow band 27-445 px; sruler measured 416 px = 8.45 m while the posts give 6.95 m).  Code conditions only: question text,
prior text, the detail answer/source.  Labels are not read.
usage: python rul071.py [write]   env C44_SUB (default sub_xt_Nzz2)
"""
import sys, re, json, os
import pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C44_SUB", "sub_xt_Nzz2")
TAG = "rul071"
COURT = {"pickleball": (6.10, 13.41), "badminton": (6.10, 13.40), "volleyball": (9.0, 18.0),
         "tennis": (10.97, 23.77), "basketball": (15.0, 28.0)}
NET = {"pickleball": 6.706, "tennis": 12.80, "badminton": 6.10, "volleyball": 9.50}
SI = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "feet": 0.3048}
UNIT = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "ft": 1 / 0.3048}


def norm(x):
    return str(x).lower().replace("pickle ball", "pickleball").replace("basket ball", "basketball")


def run(split):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    PP = {int(p["id"]): p for p in map(json.loads, open(f"{R}/geo/out/parse_{split}_v2.jsonl"))}
    out = {}
    for i in det.index:
        q, p = norm(raw.question.get(i, "")), norm(raw.ground_truth_prior.get(i, ""))
        if not re.search(r"\b(?:length of the net|how long is the net)\b", q):
            continue
        sp_ = "|".join(COURT)
        mp = re.search(r"\b(length|width)\b[^=]*\b(" + sp_ + r")\s*(?:ball\s*)?court\b[^=]*=\s*([\d.]+)\s*([a-z]+)", p)
        if not mp or mp.group(4) not in SI or mp.group(2) not in NET:
            print(f"[{split}] {i} skip: prior '{p}' not a regulation court with a net table entry"); continue
        w, l = COURT[mp.group(2)]
        pv = float(mp.group(3)) * SI[mp.group(4)]
        ref = w if mp.group(1) == "width" else l
        if abs(pv / ref - 1) > 0.03:
            print(f"[{split}] {i} skip: prior {pv} m is not regulation ({ref})"); continue
        unit = ((PP.get(int(i)) or {}).get("q") or {}).get("unit") or "m"
        if unit not in UNIT:
            print(f"[{split}] {i} skip unit {unit}"); continue
        val = NET[mp.group(2)] * UNIT[unit]
        a0 = float(det.answer[i])
        if abs(a0 / val - 1) <= 0.03 or str(det.answer_source[i]).startswith("patch:"):
            print(f"[{split}] {i} skip: answer {a0:.4g} ({det.answer_source[i]})"); continue
        out[str(int(i))] = {"value": float(val), "why": f"{TAG}: regulation {mp.group(2)} net length {NET[mp.group(2)]} m "
                            f"(court prior {pv:g} m regulation); was {a0:.4g} ({det.answer_source[i]}), geo {det.g[i]:.4g}, CaW {det.cs[i]:.4g}"}
        print(f"[{split}] {i} {det.video_id[i]} '{raw.question.get(i, '')}' {a0:.4g} -> {val:.4g}")
    return out


if __name__ == "__main__":
    res = {sp: run(sp) for sp in ("test", "val")}
    if "write" in sys.argv[1:]:
        for sp, o in res.items():
            json.dump(o, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
        print("wrote", {sp: len(o) for sp, o in res.items()})
