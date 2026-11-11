import glob, os, cv2, numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
res=[]
for sp in ('test','val'):
    for p in sorted(glob.glob(R+f'/data/{sp}_480p/simulation_*.mp4')):
        cap=cv2.VideoCapture(p); fr=[]
        for _ in range(4):
            ok,im=cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float))
        if len(fr)<4: continue
        d=[np.abs(fr[i]-fr[i-1]).mean() for i in (1,2,3)]
        res.append((sp,os.path.basename(p)[:-4],*d))
import collections
c=collections.Counter()
for r in res:
    moving = r[3]>0.05 and r[4]>0.05
    if moving: c[('d01_static' if r[2]<0.02 else 'd01_moving')]+=1
print(c)
for r in res:
    if r[3]>0.05 and r[4]>0.05 and r[2]>=0.02: print('moving01', r[0], r[1], ['%.3f'%x for x in r[2:]])
for r in res:
    if r[2]<0.02 and r[3]>0.05: print('static01', r[0], r[1], ['%.3f'%x for x in r[2:]])
