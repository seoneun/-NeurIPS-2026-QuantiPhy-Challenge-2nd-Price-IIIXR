import sys, cv2, numpy as np, pandas as pd
cap = cv2.VideoCapture('data/test/simulation_0145.mp4'); fps = cap.get(cv2.CAP_PROP_FPS); F = []
while True:
    ok, fr = cap.read()
    if not ok: break
    F.append(fr.astype(np.int16))
T, S, L = [], [], []
for i in range(6, 22):
    mk = (np.abs(F[i] - F[0]).max(axis=2) > 40).astype(np.uint8); mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mk, 8)
    if n < 2: continue
    k = 1 + np.argmax(st[1:, 4]); ys, xs = np.where(lab == k); P = np.c_[xs, ys].astype(float); c = P.mean(0)
    vt = np.linalg.svd(P - c, full_matrices=False)[2]; d = vt[0] if vt[0][0] < 0 else -vt[0]
    pr = (P - c) @ vt[0]; T.append(i); S.append(P[np.argmax((P - c) @ d)]); L.append(pr.max() - pr.min())
S = np.array(S); s = np.hypot(*(S - S[0]).T); T = np.array(T, float)
sel = T >= 6; p = np.polyfit(T[sel], s[sel], 2); Lm = np.median([l for t, l in zip(T, L) if t >= 12])
pd.DataFrame([dict(id0=341, value=round(2 * p[0] * fps ** 2 * 2.9 / Lm, 4))]).to_csv(sys.argv[1], index=False)
