import sys, math, pandas as pd
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/kin')
import gruler
out = []
for id0, vid in {2180: 'captured_0009s', 2185: 'captured_0009x'}.items():
    out.append((id0, gruler.est(vid)))
for _, p in pd.read_csv('analysis/v3/ag/r9_main/levers/kin/in/rul334_scan_test.csv').iterrows():
    L = 9.8 * p['T'] ** 2 / (4 * math.pi ** 2)
    for _ in range(100):
        th = math.asin(min(p.A_tot / L, 0.99)); K = 1 + th ** 2 / 16 + 11 * th ** 4 / 3072
        L = 9.8 * (p['T'] / K) ** 2 / (4 * math.pi ** 2)
    out.append((int(p.id), round(100 * (L - 0.0335), 2)))
pd.DataFrame(out, columns=['id0', 'value']).to_csv(sys.argv[1], index=False)
