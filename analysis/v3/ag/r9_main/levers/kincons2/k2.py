import sys, numpy as np
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/kincons')
from trk_kc import load
def series(rows):
    return np.array([(r[1], r[2], r[3]) for r in rows if not r[7]])
