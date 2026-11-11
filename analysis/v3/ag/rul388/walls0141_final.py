import json, numpy as np, cv2, csv, sys
R='@QP_ROOT@'; O=f'{R}/cache/tmp/rul388'
m=json.load(open(f'{O}/walls0141_measure.json')); cx,cy=json.load(open(f'{O}/walls0141_vp.json'))['vp']
sL=m['S_WALL_L']; pil=[p for p in m['pillars'] if 2.0<p[0]<3.0 and p[3]>=80]; sR=float(np.median([p[0] for p in pil]))
tr=[r for r in csv.DictReader(open(f'{O}/boxtracks_simulation_0141.csv')) if r['src']=='M']
def get(key,f): return [r for r in tr if r['key']==key and int(r['fidx'])==f][0]
L=[];Rr=[];hcs=[]
for f in (0,12,24,36,48,59):
    man=get('obj0:man|/person|',f); ft=get('obj10:foot|',f)
    yb=float(man['y1']); h=yb-float(man['y0']); sf=(float(ft['cx'])-cx)/(float(ft['y1'])-cy); hc=1.8*(yb-cy)/h
    L.append((sf-sL)*hc); Rr.append((sR-sf)*hc); hcs.append(hc)
left=float(np.median(L)); right=float(np.median(Rr)); hc=float(np.median(hcs))
sfm=float(np.median([(float(get('obj10:foot|',f)['cx'])-cx)/(float(get('obj10:foot|',f)['y1'])-cy) for f in (0,12,24,36,48,59)]))
if '--v' in sys.argv:
    print('pillars used',[(round(p[0],3),int(p[1]),int(p[2]),int(p[3])) for p in pil],'s_R %.3f s_L %.3f'%(sR,sL))
    print('per-frame left',np.round(L,3),'right',np.round(Rr,3),'h_c median %.3f'%hc)
    hcB=m['hcB']; print('check (seam scale h_c %.3f): left %.3f right %.3f'%(hcB,(sfm-sL)*hcB,(sR-sfm)*hcB))
    bg=cv2.imread(f'{O}/walls0141_bg.png')
    for s,c in ((sL,(0,255,0)),(sR,(0,128,255)),(sfm,(255,0,0))):
        cv2.line(bg,(int(cx),int(cy)),(int(cx+s*(479-cy)),479),c,2)
    for p in pil: cv2.circle(bg,(int(p[1]),int(p[2])),4,(0,0,255),-1)
    cv2.imwrite(f'{O}/walls0141_check.jpg',bg,[cv2.IMWRITE_JPEG_QUALITY,85])
print(1510, round(left,4)); print(1511, round(right,4))
