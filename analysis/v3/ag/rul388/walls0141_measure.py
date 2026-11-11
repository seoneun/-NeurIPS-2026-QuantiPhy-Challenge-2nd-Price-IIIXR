import cv2, numpy as np, json, csv
R='@QP_ROOT@'; O=f'{R}/cache/tmp/rul388'
bg=cv2.imread(f'{O}/walls0141_bg.png'); g=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY).astype(float)
cx,cy=json.load(open(f'{O}/walls0141_vp.json'))['vp']
_lc=list(csv.DictReader(open(f'{O}/walls0141_lines.csv')))
S_WALL_L=float(np.mean([float(_lc[i]['slope_dx_dy']) for i in (0,3)]))
tr=[r for r in csv.DictReader(open(f'{O}/boxtracks_simulation_0141.csv')) if r['src']=='M']
def get(key,f): 
    r=[r for r in tr if r['key']==key and int(r['fidx'])==f][0]; return r
res={}
for f in (0,12,24,36,48,59):
    man=get('obj0:man|/person|',f); ft=get('obj10:foot|',f)
    yb=float(man['y1']); h=yb-float(man['y0']); xf=float(ft['cx']); yf=float(ft['y1'])
    hcA=lambda H: H*(yb-cy)/h
    sf=(xf-cx)/(yf-cy)
    res[f]=(sf,hcA(1.837),hcA(1.78))
    print(f'f{f}: foot ({xf:.1f},{yf:.0f}) s_f {sf:.4f} man h {h:.0f}px bottom {yb:.0f}  h_c(H=1.837) {hcA(1.837):.3f}  h_c(H=1.78) {hcA(1.78):.3f}')
gx=cv2.Sobel(cv2.GaussianBlur(g,(3,3),0),cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(cv2.GaussianBlur(g,(3,3),0),cv2.CV_64F,0,1,ksize=3)
ss=np.arange(-3.4,-0.9,0.002); prof=[]
for s in ss:
    ys=np.arange(400,479); xs=cx+s*(ys-cy); m=(xs>=1)&(xs<852)
    if m.sum()<20: prof.append(np.nan); continue
    nx,ny=1/np.hypot(1,s),-s/np.hypot(1,s)
    v=np.abs(gx[ys[m],xs[m].astype(int)]*nx+gy[ys[m],xs[m].astype(int)]*ny)
    prof.append(v.mean())
prof=np.array(prof); p=prof-np.nanmean(prof); p=np.nan_to_num(p)
ac=np.correlate(p,p,'full')[len(p)-1:]; ac/=ac[0]
lags=np.arange(len(ac))*0.002
lo=int(0.15/0.002); k=lo+np.argmax(ac[lo:int(0.9/0.002)])
ds=lags[k]
pk=[ss[i] for i in range(2,len(p)-2) if p[i]==max(p[i-2:i+3]) and p[i]>np.std(p)]
print('seam period ds %.3f (autocorr %.2f); profile peaks s:'%(ds,ac[k]), np.round(pk,3)[:25])
tile=0.6057; hcB=tile/ds
print('h_c from seams (tile 0.6057) %.3f'%hcB)
e=cv2.Canny(cv2.GaussianBlur(bg,(3,3),0),40,120)
L=cv2.HoughLinesP(e,1,np.pi/360,40,minLineLength=40,maxLineGap=4)[:,0,:].astype(float)
pil=[]
for x0,y0,x1,y1 in L:
    a=np.degrees(np.arctan2(abs(y1-y0),abs(x1-x0)))
    if a<84: continue
    xb,yb=(x0,y0) if y0>y1 else (x1,y1)
    if xb>cx+40 and yb>cy+20: pil.append(((xb-cx)/(yb-cy),xb,yb,abs(y1-y0)))
pil.sort()
print('pillar bottom candidates (s, x, y, len):',[(round(a,3),int(b),int(c),int(d)) for a,b,c,d in pil[:12]])
json.dump({'S_WALL_L':S_WALL_L,'res':res,'ds':ds,'hcB':hcB,'pillars':pil},open(f'{O}/walls0141_measure.json','w'))
sf,hc1,hc2=res[0]
print('LEFT  f0: A(H=1.837) %.3f  A(H=1.78) %.3f  B(seams) %.3f'%((sf-S_WALL_L)*hc1,(sf-S_WALL_L)*hc2,(sf-S_WALL_L)*hcB))
for f in (12,24,36,48,59):
    sf2,h1,h2=res[f]; print('   f%d: A %.3f'%(f,(sf2-S_WALL_L)*h1))
