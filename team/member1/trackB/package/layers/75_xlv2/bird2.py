import cv2,numpy as np,re,csv,sys,json,os
LMODE='min'
R=os.environ.get('QP_R','.')
def blobs(video):
    c=cv2.VideoCapture(R+'/data/test/%s.mp4'%video); W=int(c.get(3)); fr=[]
    while True:
        ok,im=c.read()
        if not ok: break
        med=np.median(im.reshape(-1,3),0)
        d=np.abs(im.astype(np.int16)-med[None,None,:]).sum(2)
        m=cv2.morphologyEx((d>60).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        n,lab,st,cen=cv2.connectedComponentsWithStats(m,8)
        fr.append([dict(x=int(st[i][0]),y=int(st[i][1]),w=int(st[i][2]),h=int(st[i][3]),a=int(st[i][4]),border=int(st[i][0]==0 or st[i][0]+st[i][2]>=W)) for i in range(1,n) if st[i][4]>=20])
    return fr,W
def tracks(fr):
    T={'A':{}, 'B':{}}
    two=[j for j in range(1,len(fr)) if len(fr[j])==2 and len(fr[j-1])>=1 and min(abs(c['x']-p['x']) for c in fr[j] for p in fr[j-1])>2]
    if two:
        j0=two[0]; a,b=sorted(fr[j0],key=lambda c:-c['a']); T['A'][j0]=a; T['B'][j0]=b
    else:
        j0=next(j for j in range(1,len(fr)) if len(fr[j])>=1 and len(fr[j-1])>=1 and abs(fr[j][0]['x']-fr[j-1][0]['x'])>2); T={'A':{j0:max(fr[j0],key=lambda c:c['a'])}}
    for j in range(j0+1,len(fr)):
        C=[]
        for ci,c in enumerate(fr[j]):
            for k in T:
                pj=max(T[k])
                if pj!=j-1: continue
                p=T[k][pj]; dx=c['x']-p['x']
                C.append((abs(np.log(c['a']/p['a']))+abs(dx)/2000,ci,k))
        used=set(); 
        for cost,ci,k in sorted(C):
            if ci in used or j in T[k]: continue
            T[k][j]=fr[j][ci]; used.add(ci)
    return T,j0
def lead(c,dirn):
    return c['x'] if dirn<0 else c['x']+c['w']
def measure(video):
    fr,W=blobs(video); T,j0=tracks(fr)
    out={}
    for k,t in T.items():
        js=sorted(t); ok=[j for j in js if not t[j]['border']]
        dirn=np.sign(t[js[-1]]['x']-t[js[0]]['x'])
        dx={j:lead(t[j],dirn)-lead(t[j-1],dirn) for j in ok if j-1 in t and not t[j-1]['border']}
        out[k]=dict(area=float(np.median([t[j]['a'] for j in ok])),v_px=float(np.median(list(dx.values()))),dx=dx,
                    L=float(np.median([t[j]['w'] for j in ok])),Lmin=float(min(t[j]['w'] for j in ok)),Wmax=float(max(t[j]['h'] for j in ok)),n=len(ok))
    o=sorted(out,key=lambda k:-out[k]['area']); big,small=o[0],o[-1]
    return {'big':out[big],'small':out[small],'j0':j0}
def rule(q,prior,fps,M):
    pm=re.search(r'speed of the (big|small|) ?bird\s*=\s*([\d.]+)\s*m/s',prior)
    if not pm: return None,'prior_not_speed'
    tgt=re.search(r'(big|small) bird',q)
    if not tgt: return None,'no_target'
    if M['big'] is M['small']: return None,'one_bird'
    P=M[pm.group(1)]; Tg=M[tgt.group(1)]
    k=float(pm.group(2))/(abs(P['v_px'])*fps)
    if re.search(r'speed',q):
        tm=re.search(r'at ([\d.]+)\s*s',q); n=round(fps*float(tm.group(1))) if tm else None
        d=Tg['dx'].get(n,Tg['v_px'])
        return abs(d)*fps*k,'speed_bd n=%s'%n
    if re.search(r'wingspan|width',q): return Tg['Wmax']*k,'wingspan_max'
    if re.search(r'length',q): return (Tg['Lmin'] if LMODE=='min' else Tg['L'])*k,'length_'+LMODE
    return None,'dim?'
if __name__=='__main__':
    import random
    sub,outp=sys.argv[1],sys.argv[2]
    rows=[r for r in csv.DictReader(open(sub,encoding='utf-8')) if re.search(r'bird',r['question'])]
    cache={}; res=[]
    for r in rows:
        v=r['video_id']
        if v not in cache:
            try: cache[v]=measure(v)
            except Exception as e: cache[v]=None
        if cache[v] is None: continue
        val,why=rule(r['question'],r['ground_truth_prior'],float(r['fps']),cache[v])
        if val is None or not np.isfinite(val): continue
        res.append((int(r['id'])-1,float(val),'fly_d2:'+why))
    w=csv.writer(open(outp,'w',newline='')); w.writerow(['id0','value','rule'])
    for x in sorted(res): w.writerow([x[0],repr(x[1]),x[2]])
    print(outp,len(res))
