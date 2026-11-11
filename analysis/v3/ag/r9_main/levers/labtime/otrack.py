import sys, subprocess, numpy as np
from scipy import ndimage
vid, out = sys.argv[1], sys.argv[2]
W, H = 1280, 720
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'data/test/{vid}.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                     capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.int16)
L = ['n,cx,cy,top,bot,left,right,area']
for n, f in enumerate(fr):
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    m = (R > 120) & (R - G > 50) & (R - B > 70)
    lab, k = ndimage.label(m)
    if k == 0:
        L.append(f'{n},,,,,,,0'); continue
    sz = ndimage.sum(m, lab, range(1, k + 1)); i = int(np.argmax(sz)) + 1
    ys, xs = np.nonzero(lab == i)
    cy0, cx0 = ys.mean(), xs.mean()
    for j in range(1, k + 1):
        if j != i and sz[j - 1] > 30:
            yj, xj = np.nonzero(lab == j)
            if abs(yj.mean() - cy0) < 60 and abs(xj.mean() - cx0) < 60:
                ys = np.r_[ys, yj]; xs = np.r_[xs, xj]
    L.append(f'{n},{xs.mean():.1f},{ys.mean():.1f},{ys.min()},{ys.max()},{xs.min()},{xs.max()},{len(ys)}')
open(out, 'w').write('\n'.join(L) + '\n')
