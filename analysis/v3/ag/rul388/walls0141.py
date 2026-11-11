import cv2, numpy as np, csv, sys, json
R='@QP_ROOT@'; O=f'{R}/cache/tmp/rul388'
cap=cv2.VideoCapture(f'{R}/data/test_480p/simulation_0141.mp4'); fr=[]
while True:
    ok,im=cap.read()
    if not ok: break
    fr.append(im)
bg=np.median(np.stack(fr),axis=0).astype(np.uint8); cv2.imwrite(f'{O}/walls0141_bg.png',bg)
g=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY); e=cv2.Canny(cv2.GaussianBlur(g,(3,3),0),40,120)
L=cv2.HoughLinesP(e,1,np.pi/360,60,minLineLength=60,maxLineGap=6)[:,0,:].astype(float)
ang=np.degrees(np.arctan2(L[:,3]-L[:,1],L[:,2]-L[:,0]))%180
L=L[(np.abs(ang-90)>8)&(np.minimum(ang,180-ang)>4)]
def line_h(l):
    p=np.array([l[0],l[1],1.]); q=np.array([l[2],l[3],1.]); h=np.cross(p,q); return h/np.linalg.norm(h[:2])
H=np.array([line_h(l) for l in L]); ln=np.hypot(L[:,2]-L[:,0],L[:,3]-L[:,1])
rng=np.random.default_rng(0); best=(0,None)
for _ in range(4000):
    i,j=rng.choice(len(L),2,replace=False); v=np.cross(H[i],H[j])
    if abs(v[2])<1e-9: continue
    v=v/v[2]
    if not (0<v[0]<853 and 100<v[1]<400): continue
    d=np.abs(H@v)
    s=ln[d<3].sum()
    if s>best[0]: best=(s,v)
vp=best[1]; d=np.abs(H@vp); inl=np.where(d<3)[0]
A=H[inl,:2]; b=-H[inl,2]; vp2=np.linalg.lstsq(A*ln[inl,None],b*ln[inl],rcond=None)[0]
print('VP ransac',vp[:2].round(1),'refined',vp2.round(1),'inliers',len(inl))
cand=[]
for k in inl:
    x0,y0,x1,y1=L[k]
    if max(y0,y1)<vp2[1]+15: continue
    yl=max(y0,y1); xl=x0 if y0>y1 else x1
    cand.append((k,xl,yl,ln[k]))
cand.sort(key=lambda c:-c[3]); cand=cand[:40]
im=bg.copy()
with open(f'{O}/walls0141_lines.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['idx','x_low','y_low','len','slope_dx_dy','x_at_456'])
    for n,(k,xl,yl,l) in enumerate(cand):
        s=(xl-vp2[0])/(yl-vp2[1]); x456=vp2[0]+s*(456-vp2[1])
        w.writerow([n,round(xl,1),round(yl,1),round(l,1),round(s,4),round(x456,1)])
        col=(0,0,255) if n%2==0 else (255,0,255)
        cv2.line(im,(int(vp2[0]),int(vp2[1])),(int(vp2[0]+s*(479-vp2[1])),479),col,1)
        cv2.putText(im,str(n),(int(np.clip(xl,5,840)),int(np.clip(yl,15,475))),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
cv2.circle(im,(int(vp2[0]),int(vp2[1])),5,(0,255,0),-1)
json.dump({'vp':vp2.tolist()},open(f'{O}/walls0141_vp.json','w'))
cv2.imwrite(f'{O}/walls0141_lines.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,85])
