#!/usr/bin/env python
import sys, os, json, re
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
TAG = "pxwalk"
CATS = ("S2", "D2", "D3")
LO, HI, V = 1.0, 1.6, 1.30
PERSON = re.compile(r"\b(person|pedestrian|human|man|woman|walker)\b")
NOT = re.compile(r"bicycle|bike|cycl|car\b|ride|riding|rider|scoot|skat|astronaut|run|jog|swim|boat|horse|accel")


def main():
    BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t064"
    CUR = sys.argv[2] if len(sys.argv) > 2 else "pred_t054"
    for split in ("test", "val"):
        det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
        cur = pd.read_csv(f"{R}/submissions/{CUR}_id1.csv").parsed_value.values if split == "test" else det.answer.values
        P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
        assert [int(x["id"]) for x in P] == [int(k) for k in det.index] and len(cur) == len(P)
        out = {}
        for i, x in enumerate(P):
            q = str(x["question"]).lower()
            if x.get("cat") not in CATS or not str(x["video_id"]).startswith("simulation"):
                continue
            if not re.search(r"speed|velocity", q) or not PERSON.search(q) or NOT.search(q):
                continue
            a = float(cur[i])
            if np.isfinite(a) and LO <= a <= HI:
                continue
            out[str(int(det.index[i]))] = dict(value=V, why=f"{TAG}: walking-person speed {a:.4g} m/s outside the typical adult "
                                                            f"range {LO}-{HI} m/s -> {V} m/s")
        json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
        print(split, "written", len(out), sorted(int(k) for k in out))


if __name__ == "__main__":
    main()
