import json, numpy as np, cv2, sys
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
J=json.load(open(R+'/geo/out/tracks_test_v2sc/simulation_0104.json'))
print({k:J[k] for k in ('N','stride','fps','H','W','t_sec')})
top=np.array([s['st']['box'][1] for s in J['objs']['1']]); f=np.array(J['fidx'])
m=(f>=2)&(f<=26); a,b=np.polyfit(f[m],top[m],1); print('slope px/f',a,'onset frame (top=%.1f)'%top[0], (top[0]-b)/a, 'end frame', (top[-1]-b)/a)
cap=cv2.VideoCapture(R+'/data/test_480p/simulation_0104.mp4'); fr=[]
while True:
    ok,im=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float))
print('nframes',len(fr), 'container fps', cap.get(cv2.CAP_PROP_FPS))
for i in range(1,min(8,len(fr))): print(i, 'meandiff prev', np.abs(fr[i]-fr[i-1]).mean().round(3))
for i in range(len(fr)-5,len(fr)): print(i, 'meandiff prev', np.abs(fr[i]-fr[i-1]).mean().round(3))
