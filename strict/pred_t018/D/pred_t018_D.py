import sys, json, csv
OV, OUT = sys.argv[1:3]
SRC = {61: "strict_final", 94: "rul335", 473: "rul335", 641: "rul442", 650: "bindmoverS3", 666: "bindmoverS3", 1058: "rul340",
       1066: "rul340", 1281: "rul336", 1294: "rul335", 1777: "rul335", 2818: "rul259", 2826: "rul259"}
sf = {int(r["id"]) - 1: float(r["parsed_value"]) for r in csv.DictReader(open(f"{OV}/submissions/pred_t058_id1.csv", newline="", encoding="utf-8"))}
P = {t: json.load(open(f"{OV}/cache/tmp/ag_{t}_patch_test.json")) for t in set(SRC.values()) if t != "strict_final"}
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id0", "value"])
    for i, t in sorted(SRC.items()):
        w.writerow([i, repr(sf[i] if t == "strict_final" else float(P[t][str(i)]["value"]))])
