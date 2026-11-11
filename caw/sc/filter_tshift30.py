import csv, json, os

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
ti = {r[""]: r["question"] for r in csv.DictReader(open(f"{R}/caw/inputs/test_input.csv"))}
ts = {r[""]: r["question"] for r in csv.DictReader(open(f"{R}/caw/inputs/test_tshift.csv"))}
keep = {int(i) for i in ts if ts[i] != ti[i]}
os.makedirs(f"{R}/caw/sc/out/test_tshift30", exist_ok=True)
n = 0
with open(f"{R}/caw/sc/out/test_tshift30/g16.jsonl", "w") as fo:
    for ln in open(f"{R}/caw/sc/out/test_tshift/g16.jsonl"):
        if int(json.loads(ln)["original_id"]) in keep:
            fo.write(ln)
            n += 1
print('')
