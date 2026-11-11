import sys, subprocess, numpy as np
vid = sys.argv[1]
W, H = 1280, 720
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'data/test/{vid}.mp4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                     capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.int16)
from scipy import ndimage
print('n,cy,bot,top,area,cx')
for n, f in enumerate(fr):
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    m = (R > 120) & (R - G > 50) & (R - B > 70)
    lab, k = ndimage.label(m)
    if k == 0:
        print(f'{n},,,,0,'); continue
    sz = ndimage.sum(m, lab, range(1, k + 1)); i = int(np.argmax(sz)) + 1
    ys, xs = np.nonzero(lab == i)
    cy0, cx0 = ys.mean(), xs.mean()
    for j in range(1, k + 1):
        if j != i and sz[j - 1] > 30:
            yj, xj = np.nonzero(lab == j)
            if abs(yj.mean() - cy0) < 60 and abs(xj.mean() - cx0) < 60:
                ys = np.r_[ys, yj]; xs = np.r_[xs, xj]
    print(f'{n},{ys.mean():.1f},{ys.max()},{ys.min()},{len(ys)},{xs.mean():.1f}')
