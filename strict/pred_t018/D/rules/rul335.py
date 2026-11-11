#!/usr/bin/env python
import sys, os, json, importlib.util
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
TAG = "rul335"
_sp = importlib.util.spec_from_file_location("rul336", f"{R}/analysis/v3/compliance/rules/rul336.py")
PQ = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(PQ)


def main():
    BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t064"
    CUR = sys.argv[2] if len(sys.argv) > 2 else "pred_t152"
    for split in ("test",):
        det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
        cur = pd.read_csv(f"{R}/submissions/{CUR}_id1.csv").parsed_value.values if split == "test" else det.answer.values
        P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
        assert [int(x["id"]) for x in P] == [int(k) for k in det.index] and len(cur) == len(P)
        out = {}
        for i, x in enumerate(P):
            e, cp = PQ.lookup(x["question"], x)
            if e is None:
                continue
            f = PQ.si_factor(x)
            a = float(cur[i]) * f
            cls, rx, nx, kd, lo, hi, src = e
            if not np.isfinite(a) or a <= 0 or lo <= a <= hi or PQ.flag(a, lo, hi) or PQ.prior_scaled(x):
                continue
            c = min(max(a, lo), hi)
            out[str(int(det.index[i]))] = dict(value=c / f, why=f"{TAG}: {cls} {x['q'].get('dim') or kd[0]} answer {a:.4g} SI outside the "
                                                                f"range {lo}-{hi} ({src}) by {max(a / hi, lo / a):.3f}x (< {PQ.MARGIN}x) -> "
                                                                f"nearest bound {c:.4g}; core phrase '{cp}'")
        json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
        print(split, "written", len(out))


if __name__ == "__main__":
    main()
