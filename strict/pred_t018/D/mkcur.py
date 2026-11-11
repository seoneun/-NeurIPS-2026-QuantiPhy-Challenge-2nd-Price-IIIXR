import sys, json, csv
rows = list(csv.reader(open(sys.argv[1], newline="", encoding="utf-8")))
k = rows[0].index("parsed_value")
for key, e in json.load(open(sys.argv[2])).items():
    rows[int(key) + 1][k] = repr(float(e["value"]))
csv.writer(open(sys.argv[3], "w", newline="", encoding="utf-8")).writerows(rows)
