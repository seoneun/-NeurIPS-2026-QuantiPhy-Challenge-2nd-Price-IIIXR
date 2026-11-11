import sys, glob, pandas as pd, json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
pat, out, n_expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
fs = sorted(f for f in glob.glob(pat) if not f.endswith("_raw.csv"))
d = pd.concat([pd.read_csv(f) for f in fs]).sort_values("Unnamed: 0")
raw = pd.concat([pd.read_csv(f.replace(".csv", "_raw.csv")) for f in fs]).sort_values("Unnamed: 0")
assert len(d) == n_expected and d["Unnamed: 0"].is_unique
d.to_csv(out, index=False); raw.to_csv(out.replace(".csv", "_raw.csv"), index=False)
infos = [json.load(open(f.replace(".csv", "_info.json"))) for f in fs]
inv = int(d.parsed_value.isna().sum() + (d.parsed_value == 0).sum())
summ = {"files": fs, "rows": len(d), "invalid": inv, "invalid_pct": round(100 * inv / len(d), 2),
        "per_shard_total_s": [i["total_s"] for i in infos], "per_shard_gen_s": [i["gen_s"] for i in infos]}
json.dump(summ, open(out.replace(".csv", "_info.json"), "w"), indent=1); print(json.dumps(summ))
