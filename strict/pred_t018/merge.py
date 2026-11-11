import sys, csv
val = {int(r["id"]) - 1: float(r["parsed_value"]) for r in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))}
for p in sys.argv[3:]:
    for r in csv.DictReader(open(p, newline="")):
        val[int(r["id0"])] = float(r["value"])
with open(sys.argv[2], "w", newline="") as f:
    f.write("id0,value\n")
    for i in sorted(val):
        f.write(f"{i},{val[i]!r}\n")
