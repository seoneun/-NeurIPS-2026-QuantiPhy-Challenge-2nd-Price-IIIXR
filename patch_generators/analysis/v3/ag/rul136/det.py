import sys, cv2, numpy as np
sys.path.insert(0,'geo'); import track_common as tc
cv2.setNumThreads(2)
def detect(fr):
    H,W=fr.shape[:2]
    g=cv2.cvtColor(fr,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(fr,cv2.COLOR_RGB2HSV)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    m=((th>35)&(hsv[...,1]<60)).astype(np.uint8)
    m[:H//2]=0
    L=cv2.HoughLinesP(m*255,1,np.pi/360,30,minLineLength=15,maxLineGap=10)
    hs,st=[],[]
    if L is None: return m,[],[]
    for x1,y1,x2,y2 in L.reshape(-1,4):
        ang=np.degrees(np.arctan2(y2-y1,x2-x1))%180
        ln=np.hypot(x2-x1,y2-y1)
        if min(ang,180-ang)<5: hs.append((x1,y1,x2,y2,ln))
        elif 20<ang<160: st.append((x1,y1,x2,y2,ln))
    return m,hs,st
for vid in sys.argv[1:]:
    fr=tc.load_video(f'data/test_480p/{vid}.mp4'); n=len(fr)
    for k,f in enumerate([n//4,n//2,3*n//4]):
        m,hs,st=detect(fr[f])
        print(vid,f,'H',sorted([(int(a),int(b),int(c),int(d)) for a,b,c,d,l in hs if l>150],key=lambda t:t[1])[:15])
        print('   S',sorted([(int(a),int(b),int(c),int(d)) for a,b,c,d,l in st],key=lambda t:t[0])[:20])
        if k==1: cv2.imwrite(f'cache/tmp/ag_c73pline_mask_{vid}.png',m*255)
