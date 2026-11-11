import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = ["pred_t002", "pred_t046", "pred_t047", "pred_t048", "pred_t049", "pred_t050", "pred_t051", "pred_t052", "pred_t034", "pred_t035", "pred_t036", "pred_t037",
         "pred_t038", "pred_t039", "pred_t040", "pred_t013", "pred_t041", "pred_t042", "pred_t043", "pred_t044", "pred_t006", "pred_t007",
         "pred_t014", "pred_t009", "pred_t010", "pred_t011", "pred_t003"]


def main(chain, out, c15f):
    T = pd.read_csv(Path(__file__).resolve().parent / "teamline_rows.csv")
    files = {}
    pre = []
    for r in T.itertuples():
        k = STEPS.index(r.step)
        if k == 0:
            pre.append(np.nan)
            continue
        if k not in files:
            files[k] = pd.to_numeric(pd.read_csv(Path(chain) / f"{STEPS[k - 1]}_id1.csv").parsed_value)
        pre.append(float(files[k][r.id0]))
    c15 = pd.read_csv(c15f).set_index("id0").value[T.id0].values
    pd.DataFrame(dict(id0=T.id0, c15_eq_pre=np.abs(c15 / np.array(pre) - 1) < 0.03)).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
