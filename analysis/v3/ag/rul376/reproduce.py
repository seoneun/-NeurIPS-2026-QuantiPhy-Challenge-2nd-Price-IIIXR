import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import conv
T = conv.load(); A = conv.alternatives(T)
ids = sorted(A)
if len(sys.argv) > 1:
    ids = [i for b, _, _, rows in conv.BLOCKS if b == sys.argv[1] for i in rows]
for i in ids:
    print(i, repr(float(A[i][0])))
