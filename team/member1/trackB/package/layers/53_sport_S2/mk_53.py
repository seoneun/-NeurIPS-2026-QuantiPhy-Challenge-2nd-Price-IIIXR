import json, sys, csv
v = float(json.load(open(sys.argv[1]))['3266']['value'])
C = [(3266, v, 'tilepitchP'), (2967, 21.0, 'seg2dpriorP')]
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for i, x, r in C:
        w.writerow([i, repr(float(x)), r])
