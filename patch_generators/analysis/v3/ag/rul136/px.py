import sys, cv2, numpy as np
sys.path.insert(0,'geo'); import track_common as tc
for vid in ['simulation_0218','simulation_0220']:
    fr=tc.load_video(f'data/test_480p/{vid}.mp4'); n=len(fr)
    for f in [n//4,n//2,3*n//4]:
        g=cv2.cvtColor(fr[f],cv2.COLOR_RGB2GRAY)
        print(vid,f,'col150',g[355:415:2,150].tolist())
        print('   row389',g[389,0:854:30].tolist())
        print('   row300',g[300,0:854:30].tolist())
