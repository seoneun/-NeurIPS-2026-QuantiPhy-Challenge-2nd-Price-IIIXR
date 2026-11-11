import numpy as np
def kfit(S, t, H):
    m = np.abs(S[:, 0] - t) <= H + 1e-9
    if m.sum() < 5: return None
    T = S[m, 0] - t; px = np.polyfit(T, S[m, 1], 2); py = np.polyfit(T, S[m, 2], 2)
    v = np.array([px[1], py[1]]); a = np.array([2 * px[0], 2 * py[0]]); sp = np.linalg.norm(v)
    return sp, float(np.linalg.norm(a))
