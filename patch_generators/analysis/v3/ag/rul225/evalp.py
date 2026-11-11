import sys, json, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
from lab_rows import labelled
import analyze_caw as A
L = labelled('sub_xt_Nzz36')
for sp in ("test", "val"):
    P = json.load(open(f'{R}/cache/tmp/ag_d03occ2_vlmruler_patch_{sp}.json'))
    print(sp, len(P))
    for k, v in P.items():
        s = L[L.split == sp]
        if int(k) in s.index:
            y = s.y[int(k)]; print('  LAB', k, y, s.m[int(k)], A.row_mra(np.array([v['value']]), np.array([y]))[0])
        print('  ', k, v)
