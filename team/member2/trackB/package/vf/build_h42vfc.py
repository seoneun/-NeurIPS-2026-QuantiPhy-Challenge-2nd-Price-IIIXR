import sys

import pandas as pd

ROWS = [66, 518, 526, 654, 762, 1213, 1217, 1218, 1823, 2623]


def main():
    out = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    z = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    for i in ROWS:
        out.loc[i, "parsed_value"] = z.parsed_value[i]
    out.to_csv(sys.argv[3], index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
