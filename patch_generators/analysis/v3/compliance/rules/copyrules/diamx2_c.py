"""diamx2_c: runnable builder for the rul267 patch (copyrules, 2026-10-17, agent jumpffg).

rul267 has no single builder. Production (scripts/pipeline/05_patch_rules.sh lines 122-127, round 4, base sub_xt_Nza) ran
  analysis/v3/ag/diamx/scan.py val; analysis/v3/ag/diamx/scan.py test   -> cache/tmp/ag_diamx_scan_{split}.csv
  analysis/v3/ag/diamx/rule.py sub_xt_Nza                                -> cache/tmp/ag_diamx_patch_{split}.json
  analysis/v3/ag/echo/rule.py ...; analysis/v3/ag/filter_c4.py           -> cache/tmp/ag_diamx2_patch_{split}.json
This wrapper runs the same chain in one process (rerun_shadow.sh resets the shadow cache/tmp after every TSV line):
  1. scan.py val / test (production file, unchanged);
  2. analysis/v3/compliance/rules/copyrules/diamx_rule_c.py BASE (rule.py with the cross-row copy pooling off unless
     DIAMX_POOL=1);
  3. the diamx -> rul267 branch of analysis/v3/ag/filter_c4.py, copied here (filter_c4 also needs the echo patch, which is
     not part of rul267): test rows whose raw merge_src starts with 'xg' or whose question names a wheel / tire are
     dropped; val rows are kept unchanged.
Both filters are properties of the row's own inputs. Prints the diamx guard log of every test row whose target or prior
is a diameter / radius, so the gate that stops a row can be read from the log.
usage: python diamx2_c.py [base]   (default sub_xt_Nza, the production base)"""
import os, sys, json, re, subprocess
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nza"
PY = sys.executable
for s in ("val", "test"):
    subprocess.run([PY, f"{R}/analysis/v3/ag/diamx/scan.py", s], check=True, cwd=R)
subprocess.run([PY, f"{HERE}/diamx_rule_c.py", BASE], check=True, cwd=R)

# ---- filter_c4.py, diamx -> rul267 branch (same keep test)
P = {}
for ln in open(f"{R}/geo/out/parse_test_v2.jsonl"):
    x = json.loads(ln); P[str(x["id"])] = x
raw = pd.read_csv(f"{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv").set_index("Unnamed: 0")
keep = lambda k, x: not str(raw.merge_src.get(int(k), "")).startswith("xg") and not re.search(r"\b(wheel|tire|tyre)s?\b", x.get("question", "").lower())
for s in ("test", "val"):
    d = json.load(open(f"{R}/cache/tmp/ag_diamx_patch_{s}.json"))
    o = {k: v for k, v in d.items() if (s == "val" and k not in P) or keep(k, P.get(k, {}))} if s == "test" else d
    json.dump(o, open(f"{R}/cache/tmp/ag_diamx2_patch_{s}.json", "w"), indent=0)
    print("diamx -> rul267", s, len(d), "->", len(o), "dropped", sorted(set(d) - set(o), key=int))

# ---- guard log (test)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 2000); pd.set_option("display.max_colwidth", 40)
LG = pd.read_csv(f"{R}/cache/tmp/ag_diamx_log.csv", index_col=0)
LG = LG[LG.split == "test"]
cols = [c for c in ("id", "side", "vid", "phrase", "cls", "old", "new", "guard", "bw", "bh", "maj", "src", "g", "gnew", "cs", "ans", "new_ans", "fac") if c in LG.columns]
print("== diamx guard log, test rows")
print(LG[cols].round(4).to_string())
