import sys, cv2, numpy as np
sys.path.insert(0,'geo'); import track_common as tc
for vid in sys.argv[1:]:
    fr=tc.load_video(f'data/test_480p/{vid}.mp4'); n=len(fr)
    for f in [n//4,n//2,3*n//4]:
        g=cv2.cvtColor(fr[f],cv2.COLOR_RGB2GRAY)
        th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
        print(vid,f)
        for y in range(372,408,3):
            r=th[y]>35
            xs=np.nonzero(np.diff(r.astype(int))==1)[0]+1; xe=np.nonzero(np.diff(r.astype(int))==-1)[0]+1
            runs=[(int(a),int(b)) for a,b in zip(xs,xe) if b-a<60]
            print('  y',y,'nbright',int(r.sum()),runs[:12])
